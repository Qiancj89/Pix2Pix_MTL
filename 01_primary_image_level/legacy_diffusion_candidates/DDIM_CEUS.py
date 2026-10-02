import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

       
config = {
    "timesteps": 1000,                
    "sampling_timesteps": 50,           
    "image_size": 128,              
    "batch_size": 4,                         
    "epochs": 500,                  
    "lr": 1e-4,                    
    "beta_start": 0.0001,             
    "beta_end": 0.02,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Using device: {config['device']}")

         
class UltrasoundDataset(Dataset):
    def __init__(self, bmode_dir, ceus_dir, transform=None):
        self.bmode_files = sorted([os.path.join(bmode_dir, f) for f in os.listdir(bmode_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.ceus_files = sorted([os.path.join(ceus_dir, f) for f in os.listdir(ceus_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.bmode_files) == len(self.ceus_files), \
            f"B-mode ({len(self.bmode_files)}) and CEUS ({len(self.ceus_files)}) image counts must match"
        self.transform = transform
        print(f"Loaded {len(self.bmode_files)} image pairs")

    def __len__(self):
        return len(self.bmode_files)

    def __getitem__(self, idx):
        try:
            bmode_img = Image.open(self.bmode_files[idx]).convert('L')
            ceus_img = Image.open(self.ceus_files[idx]).convert('RGB')
            
            seed = torch.randint(0, 2**32, (1,)).item()               
            torch.manual_seed(seed)
            
            if self.transform:
                bmode_img = self.transform(bmode_img)
                ceus_img = self.transform(ceus_img)
            
                         
            bmode_img = (bmode_img - 0.5) * 2
            ceus_img = (ceus_img - 0.5) * 2
            
            return bmode_img, ceus_img
        except Exception as e:
            print(f"Error loading {self.bmode_files[idx]} or {self.ceus_files[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        
transform = transforms.Compose([
    transforms.Resize(config['image_size']),
    transforms.CenterCrop(config['image_size']),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

       
class NoiseScheduler:
    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000):
                         
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cumprods = torch.cumprod(self.alphas, dim=0)
        self.timesteps = timesteps
        
    def to_device(self, device):
                           
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprods = self.alpha_cumprods.to(device)
        return self
        
    def add_noise(self, original, noise, t):
                      
        t = t.clamp(0, self.timesteps - 1)
        
                
        device = original.device
        
                                 
        alpha_cumprods = self.alpha_cumprods.to(device)
        
                
        sqrt_alpha_prod = torch.sqrt(alpha_cumprods[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_cumprods[t]).view(-1, 1, 1, 1)
        
        return sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise

            
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        
             
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        
             
        self.mid = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
             
        self.dec3 = self._block(256 + 256, 128, transpose=True)
        self.dec2 = self._block(128 + 128, 64, transpose=True)
        self.dec1 = self._block(64 + 64, 32, transpose=True)
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Tanh()
        )
        
                 
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_c, out_c, transpose=False):
        if transpose:
            conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1)
        else:
            conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, t, condition):
                
        x = torch.cat([x, condition], dim=1)
        
               
        enc1 = self.enc1(x)
        x = self.down(enc1)
        
        enc2 = self.enc2(x)
        x = self.down(enc2)
        
        enc3 = self.enc3(x)
        x = self.down(enc3)
        
             
        x = self.mid(x)
        
               
        x = self.up(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        return self.final(x)

         
class DDIMSampler:
    def __init__(self, model, scheduler, config):
        self.model = model
        self.scheduler = scheduler
        self.config = config
        
    @torch.no_grad()
    def sample(self, condition):
        model = self.model
        device = self.config['device']
        img_size = self.config['image_size']
        total_timesteps = self.config['timesteps']
        sampling_timesteps = self.config['sampling_timesteps']
        
                
        x = torch.randn((1, 3, img_size, img_size), device=device)
        condition = condition.unsqueeze(0).to(device)
        
                
        step_interval = total_timesteps // sampling_timesteps
        timesteps = torch.arange(0, total_timesteps, step_interval, device=device).flip(0)
        
                                 
        alpha_cumprods = self.scheduler.alpha_cumprods.to(device)
        
        for i, t in enumerate(timesteps):
                    
            time_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            
                  
            pred_noise = model(x, time_tensor, condition)
            
                      
            alpha_prod_t = alpha_cumprods[t]
            if i < len(timesteps)-1:
                next_t = timesteps[i+1]
                alpha_prod_t_prev = alpha_cumprods[next_t]
            else:
                alpha_prod_t_prev = torch.tensor(1.0, device=device)
            
                       
            pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * pred_noise) / torch.sqrt(alpha_prod_t)
            
                  
            dir_xt = torch.sqrt(1 - alpha_prod_t_prev) * pred_noise
            
                 
            x = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt
        
                         
        x = torch.clamp(x, -1, 1)
        return (x + 1) / 2              

      
def train(model, dataloader, optimizer, scheduler, config):
    model.train()
    device = config['device']
    mse_loss = nn.MSELoss()
    
               
    scheduler.to_device(device)
    
    for epoch in range(config['epochs']):
        total_loss = 0
        progress_bar = tqdm(dataloader)
        
        for batch_idx, (bmode, ceus) in enumerate(progress_bar):
            bmode = bmode.to(device)
            ceus = ceus.to(device)
            
                       
            t = torch.randint(0, config['timesteps'], (bmode.size(0),)).to(device)
            
                  
            noise = torch.randn_like(ceus).to(device)
            
                  
            noisy_images = scheduler.add_noise(ceus, noise, t)
            
                  
            pred_noise = model(noisy_images, t, bmode)
            
                  
            loss = mse_loss(pred_noise, noise)
            
                  
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)        
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Avg Loss: {avg_loss:.6f}")
        
                         
        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(model.state_dict(), f"DDIM_CEUS_model/model_epoch_{epoch+1}.pth")
            generate_examples(model, scheduler, dataloader, config, epoch)

        
def generate_examples(model, scheduler, dataloader, config, epoch):
    sampler = DDIMSampler(model, scheduler, config)
    model.eval()
    
                 
    bmode, ceus = next(iter(dataloader))
    bmode = bmode[0].to(config['device'])
    
          
    with torch.no_grad():
        generated = sampler.sample(bmode)
    
                   
    bmode_np = bmode.cpu().squeeze().numpy()
    generated_np = generated.squeeze().permute(1, 2, 0).cpu().numpy()
    ceus_np = ceus[0].permute(1, 2, 0).numpy()
    
          
    bmode_np = (bmode_np + 1) / 2  # [-1,1] -> [0,1]
    ceus_np = (ceus_np + 1) / 2
    
           
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title("Input B-mode")
    plt.imshow(bmode_np, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    plt.subplot(132)
    plt.title("Generated CEUS")
    plt.imshow(np.clip(generated_np, 0, 1), vmin=0, vmax=1)
    plt.axis('off')
    
    plt.subplot(133)
    plt.title("Real CEUS")
    plt.imshow(ceus_np, vmin=0, vmax=1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"result_epoch_{epoch+1}.png")
    plt.close()
    print(f"Saved example image for epoch {epoch+1}")

     
if __name__ == "__main__":
            
    dataset = UltrasoundDataset(
        bmode_dir="B_mode",
        ceus_dir="CEUS",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                           shuffle=True, num_workers=2, pin_memory=True)
    
    if not os.path.exists("DDIM_CEUS_model"):
        os.makedirs("DDIM_CEUS_model")
    
               
    model = SimpleUNet().to(config['device'])
    noise_scheduler = NoiseScheduler(
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        timesteps=config['timesteps']
    )
    
         
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
            
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    
           
    start_epoch = 0
    checkpoint_path = "DDIM_CEUS_model/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch+1}")
    
          
    for epoch in range(start_epoch, config['epochs']):
        train(model, dataloader, optimizer, noise_scheduler, config)
        
               
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
               
        lr_scheduler.step(train_loss)           
    
                 
    torch.save(model.state_dict(), "DDIM_CEUS_model/final_model.pth")