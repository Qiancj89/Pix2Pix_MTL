import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from tqdm import tqdm, trange
from pytorch_msssim import ssim, SSIM

                       
class AttentionFusion(nn.Module):
    def __init__(self, channels, reduction=8, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
               
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
               
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
              
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
              
        if use_residual:
            self.res_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
               
        channel_att = self.channel_att(x)
        x_att = x * channel_att
        
               
        spatial_input = torch.cat([
            torch.mean(x_att, dim=1, keepdim=True),
            torch.std(x_att, dim=1, keepdim=True)
        ], dim=1)
        spatial_att = self.spatial_att(spatial_input)
        
                 
        x_spatial = x_att * spatial_att
        
                   
        gated = self.gate(x_spatial)
        fused = gated * x_spatial
        
              
        if self.use_residual:
            fused = fused + self.res_conv(fused)
            
        return fused

                          
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, sample_size=256):
        super().__init__()
                      
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=1,                
            out_channels=1,                
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", 
                "AttnUpBlock2D",
                "UpBlock2D", 
                "UpBlock2D"
            ),
            cross_attention_dim=256           
        )
        
                         
        self.condition_encoder = nn.Sequential(
                                
            nn.Conv2d(2, 64, kernel_size=3, padding=1),              
            nn.ReLU(),
            nn.MaxPool2d(2),
            AttentionFusion(64),             
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, noisy_images, timesteps, condition_images, mask_images):
                         
        condition_input = torch.cat([condition_images, mask_images], dim=1)
        
                
        condition_embeds = self.condition_encoder(condition_input)
        condition_embeds = condition_embeds.unsqueeze(1)  # (batch, 1, 256)
        
              
        return self.unet(noisy_images, timesteps, encoder_hidden_states=condition_embeds).sample

                      
class UltrasoundPairDataset(Dataset):
    def __init__(self, b_mode_paths, mask_paths, ceus_paths, transform=None):
        



           
        self.b_mode_paths = b_mode_paths
        self.mask_paths = mask_paths
        self.ceus_paths = ceus_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
                         
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.b_mode_paths)

    def __getitem__(self, idx):
        try:
            b_mode_img = Image.open(self.b_mode_paths[idx]).convert('L')
            mask_img = Image.open(self.mask_paths[idx]).convert('L')
            ceus_img = Image.open(self.ceus_paths[idx]).convert('L')
            
            if self.transform:
                b_mode_tensor = self.transform(b_mode_img)
                mask_tensor = self.mask_transform(mask_img)
                ceus_tensor = self.transform(ceus_img)
                
            return {
                'b_mode': b_mode_tensor,
                'mask': mask_tensor,
                'ceus': ceus_tensor,
                'b_mode_path': self.b_mode_paths[idx]
            }
        except Exception as e:
            print(f"Error loading image {self.b_mode_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

             
def forward_diffusion(x0, t, noise_scheduler):
                   
    noise = torch.randn_like(x0)
    noisy_image = noise_scheduler.add_noise(x0, noise, t)
    return noisy_image, noise

                     
def train_model(model, dataloader, optimizer, scheduler, noise_scheduler, device, epochs=100, save_dir="models"):
    model.train()
    model.to(device)
    
            
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
          
    train_log = {
        "epoch": [],
        "step1_loss": [],
        "step2_loss": [],
        "learning_rate": [],
        "time_per_epoch": []
    }
    
           
    for epoch in range(epochs):
        epoch_start_time = time.time()
        step1_losses = []
        step2_losses = []
        
                       
        print(f"Epoch {epoch+1}/{epochs} - Step 1: Training mask region")
        model.train()
        for batch in tqdm(dataloader, desc="Step 1: Mask Region"):
            b_mode = batch['b_mode'].to(device)
            mask = batch['mask'].to(device)
            ceus = batch['ceus'].to(device)
            
                              
            masked_ceus = ceus * mask
            
                     
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (b_mode.size(0),), device=device
            ).long()
            
                  
            noisy_images, noise = forward_diffusion(masked_ceus, timesteps, noise_scheduler)
            
                  
            noise_pred = model(noisy_images, timesteps, b_mode, mask)
            
                             
            loss = F.mse_loss(noise_pred * mask, noise * mask)
            
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step1_losses.append(loss.item())
        
                       
        print(f"Epoch {epoch+1}/{epochs} - Step 2: Training non-mask region")
        model.train()
        for batch in tqdm(dataloader, desc="Step 2: Non-Mask Region"):
            b_mode = batch['b_mode'].to(device)
            mask = batch['mask'].to(device)
            ceus = batch['ceus'].to(device)
            
                               
            non_masked_ceus = ceus * (1 - mask)
            
                     
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (b_mode.size(0),), device=device
            ).long()
            
                  
            noisy_images, noise = forward_diffusion(non_masked_ceus, timesteps, noise_scheduler)
            
                              
            noise_pred = model(noisy_images, timesteps, b_mode, 1 - mask)
            
                              
            loss = F.mse_loss(noise_pred * (1 - mask), noise * (1 - mask))
            
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step2_losses.append(loss.item())
        
                
        avg_step1_loss = np.mean(step1_losses)
        avg_step2_loss = np.mean(step2_losses)
        epoch_time = time.time() - epoch_start_time
        
                
        train_log["epoch"].append(epoch+1)
        train_log["step1_loss"].append(avg_step1_loss)
        train_log["step2_loss"].append(avg_step2_loss)
        train_log["learning_rate"].append(scheduler.get_last_lr()[0])
        train_log["time_per_epoch"].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Step1 Loss: {avg_step1_loss:.6f} | "
              f"Step2 Loss: {avg_step2_loss:.6f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
                       
        model_path = os.path.join(save_dir, f"diffusion_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
            
              
        generate_samples(model, dataloader, noise_scheduler, device, epoch+1)
    
            
    final_model_path = os.path.join(save_dir, "final_diffusion_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
            
    np.save(os.path.join(save_dir, "train_log.npy"), train_log)
    
            
    plt.figure(figsize=(12, 6))
    plt.plot(train_log["epoch"], train_log["step1_loss"], label="Step1 Loss (Mask Region)")
    plt.plot(train_log["epoch"], train_log["step2_loss"], label="Step2 Loss (Non-Mask Region)")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()
    
    return model, train_log

                         
@torch.no_grad()
def generate_samples(model, dataloader, noise_scheduler, device, epoch, num_samples=4):
    model.eval()
    batch = next(iter(dataloader))
    b_mode = batch['b_mode'][:num_samples].to(device)
    mask = batch['mask'][:num_samples].to(device)
    
                        
    print("Generating mask region...")
    mask_region_ceus = generate_region(model, noise_scheduler, device, b_mode, mask, mask_type='mask')
    
                        
    print("Generating non-mask region...")
    non_mask_region_ceus = generate_region(model, noise_scheduler, device, b_mode, mask, mask_type='non-mask')
    
             
    generated_ceus = mask_region_ceus * mask + non_mask_region_ceus * (1 - mask)
    
         
    generated_ceus = torch.clamp((generated_ceus + 1) / 2, 0, 1).cpu()
    
           
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
              
        plt.subplot(3, num_samples, i+1)
        plt.imshow(batch['b_mode'][i].squeeze(), cmap='gray')
        plt.title("Input B-mode")
        plt.axis('off')
        
                
        plt.subplot(3, num_samples, i+num_samples+1)
        plt.imshow(batch['mask'][i].squeeze(), cmap='gray')
        plt.title("Input Mask")
        plt.axis('off')
        
                 
        plt.subplot(3, num_samples, i+2*num_samples+1)
        plt.imshow(generated_ceus[i].squeeze(), cmap='viridis')
        plt.title(f"Generated CEUS")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"samples/generated_samples_epoch_{epoch}.png")
    plt.close()

             
def generate_region(model, noise_scheduler, device, b_mode, mask, mask_type='mask', num_steps=1000):
                       
    sample_shape = (b_mode.size(0), 1, 256, 256)
    noisy_images = torch.randn(sample_shape, device=device)
    
                 
    if mask_type == 'mask':
        region_mask = mask
    else:  # non-mask
        region_mask = 1 - mask
    
          
    step_interval = noise_scheduler.config.num_train_timesteps // num_steps
    timesteps = list(reversed(range(0, noise_scheduler.config.num_train_timesteps, step_interval)))
    
          
    for t in tqdm(timesteps, desc=f"Generating {mask_type} region"):
        timestep = torch.full((b_mode.size(0),), t, device=device, dtype=torch.long)
        
              
        noise_pred = model(noisy_images, timestep, b_mode, region_mask)
        
                  
        masked_noise_pred = noise_pred * region_mask
        
                
        noisy_images = noise_scheduler.step(masked_noise_pred, t, noisy_images).prev_sample
    
    return noisy_images

        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 4
    learning_rate = 1e-4
    epochs = 200
    sample_size = 256
    
            
    b_mode_paths = []
    mask_paths = []
    ceus_paths = []
    
                                
    for i in range(0, len(os.listdir('B_mode'))):
        b_mode_paths.append('B_mode/'+os.listdir('B_mode')[i])
        mask_paths.append('Mask/'+os.listdir('Mask')[i])
        ceus_paths.append('CEUS/'+os.listdir('CEUS')[i])
    
           
    dataset = UltrasoundPairDataset(b_mode_paths, mask_paths, ceus_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
           
    model = ConditionalDiffusionModel(sample_size=sample_size)
    
           
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        clip_sample=False
    )
    
                
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * epochs * 2                      
    )
    
          
    print(f"Starting training for {epochs} epochs...")
    trained_model, train_log = train_model(
        model, dataloader, optimizer, lr_scheduler, 
        noise_scheduler, device, epochs, save_dir="models"
    )
    
    print("Training completed.")

         
def generate_from_bmode_and_mask(b_mode_path, mask_path, model_path, save_path, device="cuda", num_steps=100):
          
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU instead")
    
          
    model = ConditionalDiffusionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
         
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    b_mode_img = Image.open(b_mode_path).convert('L')
    mask_img = Image.open(mask_path).convert('L')
    
    b_mode_tensor = transform(b_mode_img).unsqueeze(0).to(device)
    mask_tensor = mask_transform(mask_img).unsqueeze(0).to(device)
    
                  
    mask_region = generate_region(model, DDPMScheduler(num_train_timesteps=1000), 
                                device, b_mode_tensor, mask_tensor, 'mask', num_steps)
    
                   
    non_mask_region = generate_region(model, DDPMScheduler(num_train_timesteps=1000), 
                                    device, b_mode_tensor, mask_tensor, 'non-mask', num_steps)
    
          
    generated_ceus = mask_region * mask_tensor + non_mask_region * (1 - mask_tensor)
    generated_ceus = torch.clamp((generated_ceus.squeeze().cpu().detach() + 1) / 2, 0, 1).numpy()
    
             
    original_size = b_mode_img.size
    resized_img = cv2.resize(generated_ceus, original_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(save_path, (resized_img * 255).astype(np.uint8))

if __name__ == "__main__":
    main()