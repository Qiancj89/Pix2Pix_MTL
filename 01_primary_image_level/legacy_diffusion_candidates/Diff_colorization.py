import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
from pathlib import Path
import time

      
class Config:
    image_size = 256
    batch_size = 4
    lr = 1e-4
    timesteps = 1000         
    epochs = 200
    save_interval = 2
    dataset_path = "B_mode/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
            
    sampling_timesteps = 50              
    eta = 0.0          
    output_dir = "./results/"
    model_dir = "./models/"

          
os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(Config.model_dir, exist_ok=True)

         
class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        
                                                        
        for i in range(0, len(os.listdir(root_dir))):
            b_mode_path = root_dir+os.listdir(root_dir)[i]
            ceus_path = 'CEUS/'+os.listdir('CEUS')[i]
            self.image_pairs.append((b_mode_path, ceus_path))
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        b_mode_path, ceus_path = self.image_pairs[idx]
        
                     
        b_mode = Image.open(b_mode_path).convert('RGB')
        ceus = Image.open(ceus_path).convert('RGB')
        
        if self.transform:
            b_mode = self.transform(b_mode)
            ceus = self.transform(ceus)
        
                     
        b_mode_lab = rgb_to_lab(b_mode)
        ceus_lab = rgb_to_lab(ceus)
        
                                     
        return b_mode_lab[0:1, :, :], ceus_lab[1:, :, :], b_mode, ceus, b_mode_path, ceus_path

                
def rgb_to_lab(rgb_tensor):
    



       
                  
    rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
    
                
    rgb_np = (rgb_np * 255).astype(np.uint8)
    
               
    pil_img = Image.fromarray(rgb_np)
    lab_img = pil_img.convert('LAB')
    
           
    lab_np = np.array(lab_img)
    lab_np = lab_np.astype(np.float32)
    
            
    lab_np[..., 0] = lab_np[..., 0] * (100.0 / 255.0)        # L: [0, 100]
    lab_np[..., 1] = (lab_np[..., 1] - 128)                  # a: [-128, 127]
    lab_np[..., 2] = (lab_np[..., 2] - 128)                  # b: [-128, 127]
    
                
    lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()
    return lab_tensor

           
def lab_to_rgb(lab_tensor):
    



       
            
    if lab_tensor.dim() == 4:
              
        batch_size = lab_tensor.size(0)
        results = []
        for i in range(batch_size):
            results.append(lab_to_rgb_single(lab_tensor[i]))
        return torch.stack(results)
    else:
                
        return lab_to_rgb_single(lab_tensor)

def lab_to_rgb_single(lab_tensor):
    

       
    lab_np = lab_tensor.permute(1, 2, 0).detach().cpu().numpy()
    
            
    lab_np[..., 0] = lab_np[..., 0] * (255.0 / 100.0)        # L: [0, 255]
    lab_np[..., 1] = lab_np[..., 1] + 128                    # a: [0, 255]
    lab_np[..., 2] = lab_np[..., 2] + 128                    # b: [0, 255]
    
                
    lab_np = np.clip(lab_np, 0, 255).astype(np.uint8)
    
               
    pil_img = Image.fromarray(lab_np, mode='LAB')
    rgb_img = pil_img.convert('RGB')
    
           
    rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float()
    return rgb_tensor

       
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

               
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, time_dim=256):
        super().__init__()
        
              
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
             
        self.down1 = self._block(in_channels, 64)
        self.down2 = self._block(64, 128)
        self.down3 = self._block(128, 256)
        self.down4 = self._block(256, 512)
        
             
        self.bottleneck = self._block(512, 1024)
        
             
        self.up1 = self._block(1024 + 512, 512)
        self.up2 = self._block(512 + 256, 256)
        self.up3 = self._block(256 + 128, 128)
        self.up4 = self._block(128 + 64, 64)
        
             
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
               
        self.down_sample = nn.MaxPool2d(2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.GroupNorm(8, features),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(8, features),
            nn.GELU()
        )
    
    def forward(self, x, time):
              
        t = self.time_mlp(time)
        t = t.unsqueeze(-1).unsqueeze(-1)              
        
               
        d1 = self.down1(x)
        d1_pool = self.down_sample(d1)
        d2 = self.down2(d1_pool)
        d2_pool = self.down_sample(d2)
        d3 = self.down3(d2_pool)
        d3_pool = self.down_sample(d3)
        d4 = self.down4(d3_pool)
        d4_pool = self.down_sample(d4)
        
             
        bottleneck = self.bottleneck(d4_pool)
        
               
        u1 = self.up1(torch.cat([self.up_sample(bottleneck), d4], dim=1))
        u2 = self.up2(torch.cat([self.up_sample(u1), d3], dim=1))
        u3 = self.up3(torch.cat([self.up_sample(u2), d2], dim=1))
        u4 = self.up4(torch.cat([self.up_sample(u3), d1], dim=1))
        
            
        return self.out(u4)

      
class DiffusionModel:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
                
        self.betas = self._linear_beta_schedule(config.timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
                  
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
                   
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def _linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
                     
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
                      
    def _extract(self, arr, t, x_shape):
        batch_size = t.shape[0]
        out = arr.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
          
    def p_losses(self, denoise_model, x_start, cond, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
              
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
                              
        model_input = torch.cat([cond, x_noisy], dim=1)
        
              
        predicted_noise = denoise_model(model_input, t)
        
                              
        l1_loss = F.l1_loss(noise, predicted_noise)
        ssim_loss = 1 - ssim(noise, predicted_noise, data_range=2.0)                
        return l1_loss + 0.5 * ssim_loss
    
                   
    @torch.no_grad()
    def ddim_sample(self, model, x_l, image_size, batch_size=1):
                               
        model.eval()
                        
        img = torch.randn((batch_size, 2, image_size, image_size), device=self.device)
        
                          
        times = torch.linspace(0, self.config.timesteps - 1, steps=self.config.sampling_timesteps)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        for time, time_next in time_pairs:
                  
            t = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            next_t = torch.full((batch_size,), time_next, device=self.device, dtype=torch.long)
            
                           
            model_input = torch.cat([x_l, img], dim=1)
            
                  
            pred_noise = model(model_input, t)
            
                       
            alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_next = self.alphas_cumprod[next_t].view(-1, 1, 1, 1)
            
                       
            x0_t = (img - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            
                  
            sigma = self.config.eta * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = torch.sqrt(1 - alpha_next - sigma**2)
            
                  
            img = torch.sqrt(alpha_next) * x0_t + c * pred_noise + sigma * torch.randn_like(img)
        
        return img

          
def ssim(img1, img2, window_size=11, data_range=1.0):
              
    return structural_similarity_index_measure(img1, img2, data_range=data_range)

                 
def calculate_ssim(img1, img2):
    img1 = img1.unsqueeze(0) if img1.dim() == 3 else img1
    img2 = img2.unsqueeze(0) if img2.dim() == 3 else img2
    return structural_similarity_index_measure(img1, img2, data_range=1.0)

          
def save_comparison(b_mode_img, generated_ceus, real_ceus, ssim_score, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
          
    axes[0].imshow(b_mode_img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("B-mode Ultrasound")
    axes[0].axis('off')
    
             
    axes[1].imshow(generated_ceus.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(f"Generated CEUS\nSSIM: {ssim_score:.4f}")
    axes[1].axis('off')
    
            
    axes[2].imshow(real_ceus.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title("Real CEUS")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

      
def train():
    config = Config()
    
           
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    
               
    dataset = UltrasoundDataset(config.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
              
    model = UNet(in_channels=3, out_channels=2).to(config.device)                
    diffusion = DiffusionModel(config)
    optimizer = Adam(model.parameters(), lr=config.lr)
    
          
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            x_l, target_ab, b_mode_rgb, ceus_rgb, b_paths, c_paths = batch
            x_l = x_l.to(config.device)
            target_ab = target_ab.to(config.device)
            
                                     
            target_ab = (target_ab * 2.0) - 1.0
            
                     
            t = torch.randint(0, config.timesteps, (x_l.shape[0],), device=config.device)
            
                  
            optimizer.zero_grad()
            loss = diffusion.p_losses(model, target_ab, x_l, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
              
        if (epoch + 1) % config.save_interval == 0:
            model_path = os.path.join(config.model_dir, f"diffusion_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)
            print(f"Model saved to {model_path}")
            
                       
            validate_and_save(model, dataset, diffusion, config, epoch+1)
    
    print("Training completed!")

           
@torch.no_grad()
def validate_and_save(model, dataset, diffusion, config, epoch):
    model.eval()
                
    indices = torch.randint(0, len(dataset), (5,))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        x_l, target_ab, b_mode_rgb, ceus_rgb, b_path, c_path = sample
        
               
        x_l = x_l.unsqueeze(0).to(config.device)
        
                     
        generated_ab = diffusion.ddim_sample(model, x_l, config.image_size, batch_size=1)
        
                        
        generated_lab = torch.cat([x_l, generated_ab], dim=1)
        
                
        generated_rgb = lab_to_rgb(generated_lab)
        
                
        ssim_score = calculate_ssim(generated_rgb, ceus_rgb.unsqueeze(0))
        
                
        save_path = os.path.join(config.output_dir, f"epoch_{epoch}_sample_{i}.png")
        save_comparison(
            b_mode_rgb, 
            generated_rgb.squeeze(0), 
            ceus_rgb,
            ssim_score.item(),
            save_path
        )
    
    print(f"Saved validation samples for epoch {epoch}")

                
@torch.no_grad()
def generate_and_compare(model_path, b_mode_path, real_ceus_path=None):
    config = Config()
    
          
    model = UNet(in_channels=3, out_channels=2).to(config.device)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
             
    diffusion = DiffusionModel(config)
    
             
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    
            
    b_mode = transform(Image.open(b_mode_path).convert('RGB'))
    b_mode_lab = rgb_to_lab(b_mode).unsqueeze(0).to(config.device)  # [1,3,H,W]
    x_l = b_mode_lab[:, 0:1, :, :]                   
    
            
    start_time = time.time()
    generated_ab = diffusion.ddim_sample(model, x_l, config.image_size, batch_size=1)
    gen_time = time.time() - start_time
    
                    
    generated_lab = torch.cat([x_l, generated_ab], dim=1)
    
            
    generated_rgb = lab_to_rgb(generated_lab)
    
                      
    real_ceus_rgb = None
    if real_ceus_path and os.path.exists(real_ceus_path):
        real_ceus = transform(Image.open(real_ceus_path).convert('RGB'))
        real_ceus_rgb = real_ceus
                
        ssim_score = calculate_ssim(generated_rgb.unsqueeze(0), real_ceus.unsqueeze(0))
    else:
        ssim_score = torch.tensor(0.0)
    
          
    save_path = os.path.join(config.output_dir, f"inference_result_{int(time.time())}.png")
    save_comparison(
        b_mode,
        generated_rgb.squeeze(0),
        real_ceus_rgb if real_ceus_rgb is not None else torch.zeros_like(b_mode),
        ssim_score.item(),
        save_path
    )
    
    print(f"Generated CEUS in {gen_time:.3f} seconds")
    print(f"SSIM score: {ssim_score.item():.4f}")
    print(f"Result saved to {save_path}")
    
          
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(b_mode.permute(1, 2, 0).numpy())
    plt.title("B-mode Ultrasound")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(generated_rgb.permute(1, 2, 0).numpy())
    plt.title("Generated CEUS")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(real_ceus_rgb.permute(1, 2, 0).numpy())
    plt.title("Generated CEUS")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return generated_rgb

      
if __name__ == "__main__":
          
    train()
    
          
    # b_mode_path = "path/to/b_mode_image.png"
    # model_path = "diffusion_model_epoch_200.pth"
    # ceus_image = generate_and_compare(b_mode_path, model_path)
    # ceus_image.save("generated_ceus.png")