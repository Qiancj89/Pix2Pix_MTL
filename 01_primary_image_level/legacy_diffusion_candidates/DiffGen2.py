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
from typing import Optional, Tuple

                       
class UltrasoundPairDataset(Dataset):
    def __init__(self, b_mode_paths, mask_paths, ceus_paths, transform=None, mask_transform=None):
        



           
        self.b_mode_paths = b_mode_paths
        self.mask_paths = mask_paths
        self.ceus_paths = ceus_paths
        
               
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
                           
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.b_mode_paths)

    def __getitem__(self, idx):
        try:
            b_mode_img = Image.open(self.b_mode_paths[idx]).convert('L')        
            mask_img = Image.open(self.mask_paths[idx]).convert('L')               
            ceus_img = Image.open(self.ceus_paths[idx]).convert('L')               
            
                  
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

         
class AttentionFusion(nn.Module):
                                  
    def __init__(self, branch_channels: int, reduction: int = 8):
        

           
        super().__init__()
                        
        self.input_channels = branch_channels * 2
        
               
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
                          
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.input_channels, self.input_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_channels // reduction, self.input_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
                               
        self.fuse_conv = nn.Conv2d(self.input_channels, branch_channels, kernel_size=3, padding=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
              
        fused = torch.cat([x1, x2], dim=1)
        
               
        spatial_avg = torch.mean(fused, dim=1, keepdim=True)
        spatial_max, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.spatial_att(spatial_concat)
        
               
        channel_att = self.channel_att(fused)
        
               
        att = spatial_att * channel_att
        att_x = fused * att
        
                         
        fused_out = self.fuse_conv(att_x)
        return fused_out + x1                 

              
class MaskCrossAttention(nn.Module):
                                     
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        N = height * width         
        
                  
        query = self.query(x).view(batch, channels, N).permute(0, 2, 1)  # (batch, N, channels)
        key = self.key(x).view(batch, channels, N)  # (batch, channels, N)
        value = self.value(x).view(batch, channels, N).permute(0, 2, 1)  # (batch, N, channels)
        
                 
        attention = torch.bmm(query, key)  # (batch, N, N)
        attention = self.softmax(attention)
        
                         
        mask = F.interpolate(mask, size=(height, width), mode='nearest')
        
                      
        mask_flat = mask.view(batch, 1, -1)  # (batch, 1, N)
        col_mask = mask_flat.permute(0, 2, 1)  # (batch, N, 1)
        
              
        assert attention.shape == (batch, N, N), f"注意力形状错误: {attention.shape}，应为({batch}, {N}, {N})"
        assert col_mask.shape == (batch, N, 1), f"掩码形状错误: {col_mask.shape}，应为({batch}, {N}, 1)"
        
                              
        attention = attention * col_mask
        
               
        out = torch.bmm(attention, value)  # (batch, N, channels)
        out = out.permute(0, 2, 1).view(batch, channels, height, width)
        
              
        return x + self.gamma * out

        
class GatedFusion(nn.Module):
                
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x1, x2], dim=1)
        gate = self.gate(combined)
        fused = self.conv(combined)
        return gate * fused + (1 - gate) * x1

        
class DualBranchEncoder(nn.Module):
                                       
    def __init__(self, base_channels: int = 64, levels: int = 4):
        super().__init__()
        self.levels = levels
        
                
        self.b_mode_encoder = nn.ModuleList()
                           
        self.mask_encoder = nn.ModuleList()
              
        self.fusions = nn.ModuleList()
        
                               
        b_in_channels = 1
        m_in_channels = 1
        
        for i in range(levels):
                       
            out_channels = base_channels * (2**i)
            
                  
            b_conv = nn.Sequential(
                nn.Conv2d(b_in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.b_mode_encoder.append(b_conv)
            
                                  
            m_conv = nn.Sequential(
                nn.Conv2d(m_in_channels, out_channels, kernel_size=3, padding=1),                      
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.mask_encoder.append(m_conv)
            
                               
            if i < 2:      
                fusion = AttentionFusion(out_channels)
            else:          
                fusion = GatedFusion(out_channels)
            self.fusions.append(fusion)
            
                         
            b_in_channels = out_channels
            m_in_channels = out_channels
        
              
        self.final_fusion = MaskCrossAttention(base_channels * (2**(levels-1)))

    def forward(self, b_mode: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, list]:
        b_features = []
        m_features = []
        fused_features = []
        
        for i in range(self.levels):
                
            b_feat = self.b_mode_encoder[i](b_mode)
            m_feat = self.mask_encoder[i](mask)
            
                          
            if i < self.levels - 1:
                b_feat = F.max_pool2d(b_feat, 2)
                m_feat = F.max_pool2d(m_feat, 2)
            
                  
            fused = self.fusions[i](b_feat, m_feat)
            
                  
            b_features.append(b_feat)
            m_features.append(m_feat)
            fused_features.append(fused)
            
                      
            b_mode = b_feat
            mask = m_feat
        
                             
        return fused, fused_features

                       
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
        
                  
        self.condition_encoder = DualBranchEncoder(base_channels=64, levels=4)
        
                        
        self.condition_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),  # 512 = 64 * 8 (base_channels * 2^3)
            nn.ReLU()
        )

    def forward(self, noisy_images, timesteps, b_mode, mask):
                
        condition_embeds, _ = self.condition_encoder(b_mode, mask)
        condition_embeds = self.condition_proj(condition_embeds)
        condition_embeds = condition_embeds.unsqueeze(1)  # (batch, 1, 256)
        
              
        return self.unet(noisy_images, timesteps, encoder_hidden_states=condition_embeds).sample

             
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
        "epoch_loss": [],
        "mask_loss": [],
        "background_loss": [],
        "total_loss": [],
        "learning_rate": [],
        "time_per_epoch": [],
        "mask_weight": [],
        "mask_ratio": []
    }
    
            
    epoch_pbar = trange(epochs, desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        total_loss = 0.0
        total_mask_loss = 0.0
        total_bg_loss = 0.0
        total_mask_ratio = 0.0
        
                                
        base_mask_weight = 1.0        
        mask_weight = min(5.0, base_mask_weight + epoch * 0.1)
        
                 
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, batch in enumerate(batch_pbar):
            b_mode = batch['b_mode'].to(device)
            mask = batch['mask'].to(device)
            ceus = batch['ceus'].to(device)
            
                     
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (b_mode.size(0),), device=device
            ).long()
            
                  
            noisy_images, noise = forward_diffusion(ceus, timesteps, noise_scheduler)
            
                  
            noise_pred = model(noisy_images, timesteps, b_mode, mask)
            
                       
            base_loss = F.mse_loss(noise_pred, noise, reduction='none')
            
                              
            weight_map = torch.ones_like(base_loss)
            
                                              
            expanded_mask = F.interpolate(
                mask, 
                size=noise_pred.shape[-2:], 
                mode='nearest'
            )
            
                           
            mask_positive_ratio = (expanded_mask > 0.5).float().mean().item()
            total_mask_ratio += mask_positive_ratio
            
                        
            if mask_positive_ratio > 0:
                weight_map[expanded_mask > 0.5] = mask_weight
            else:
                                   
                pass
            
                  
            weighted_loss = base_loss * weight_map
            loss = weighted_loss.mean()
            
                              
            if mask_positive_ratio > 0:
                mask_loss = base_loss[expanded_mask > 0.5].mean().item()
            else:
                mask_loss = 0.0
                
            bg_ratio = 1 - mask_positive_ratio
            if bg_ratio > 0:
                bg_loss = base_loss[expanded_mask <= 0.5].mean().item()
            else:
                bg_loss = 0.0
                
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
                  
            total_loss += loss.item()
            total_mask_loss += mask_loss
            total_bg_loss += bg_loss
            
            train_log["total_loss"].append(loss.item())
            train_log["mask_loss"].append(mask_loss)
            train_log["background_loss"].append(bg_loss)
            
            batch_pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "mask_l": f"{mask_loss:.6f}",
                "bg_l": f"{bg_loss:.6f}",
                "mask_w": f"{mask_weight:.2f}",
                "mask_r": f"{mask_positive_ratio:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
                
        avg_loss = total_loss / len(dataloader)
        avg_mask_loss = total_mask_loss / len(dataloader)
        avg_bg_loss = total_bg_loss / len(dataloader)
        avg_mask_ratio = total_mask_ratio / len(dataloader)

        epoch_time = time.time() - epoch_start_time
        
                
        train_log["epoch_loss"].append(avg_loss)
        train_log["mask_loss"].append(avg_mask_loss)
        train_log["background_loss"].append(avg_bg_loss)
        train_log["learning_rate"].append(scheduler.get_last_lr()[0])
        train_log["time_per_epoch"].append(epoch_time)
        train_log["mask_weight"].append(mask_weight)
        train_log["mask_ratio"].append(avg_mask_ratio)
        
                
        epoch_pbar.set_postfix({
            "loss": f"{avg_loss:.6f}", 
            "mask_l": f"{avg_mask_loss:.6f}",
            "bg_l": f"{avg_bg_loss:.6f}",
            "mask_w": f"{mask_weight:.2f}",
            "mask_r": f"{avg_mask_ratio:.4f}",
            "time": f"{epoch_time:.2f}s"
        })
        
                             
        if (epoch + 1) % 2 == 0:
            model_path = os.path.join(save_dir, f"diffusion_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            
                  
            generate_samples(model, dataloader, noise_scheduler, device, epoch+1)
    
            
    final_model_path = os.path.join(save_dir, "final_diffusion_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
            
    np.save(os.path.join(save_dir, "train_log.npy"), train_log)
    
            
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    plt.plot(train_log["epoch_loss"])
    plt.title("Total Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(3, 2, 2)
    plt.plot(train_log["mask_loss"], label="Mask Loss")
    plt.plot(train_log["background_loss"], label="Background Loss")
    plt.title("Mask vs Background Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(3, 2, 3)
    plt.plot(train_log["mask_weight"])
    plt.title("Mask Weight Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    
    plt.subplot(3, 2, 4)
    plt.plot(train_log["mask_ratio"])
    plt.title("Average Mask Ratio per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    
    plt.subplot(3, 2, 5)
    plt.plot(train_log["learning_rate"])
    plt.title("Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close()
    
    return model, train_log

           
@torch.no_grad()
def generate_samples(model, dataloader, noise_scheduler, device, epoch, num_samples=4):
    model.eval()
    batch = next(iter(dataloader))
    b_mode = batch['b_mode'][:num_samples].to(device)
    mask = batch['mask'][:num_samples].to(device)
    
            
    sample_shape = (num_samples, 1, 256, 256)
    noisy_images = torch.randn(sample_shape, device=device)
    
             
    timesteps = list(reversed(range(0, noise_scheduler.config.num_train_timesteps)))
    denoise_pbar = tqdm(timesteps, desc="Denoising", leave=False)

              
    with torch.inference_mode():
        for t in denoise_pbar:
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(noisy_images, timestep, b_mode, mask)
            noisy_images = noise_scheduler.step(noise_pred, t, noisy_images).prev_sample
            denoise_pbar.set_postfix({"step": t})
    
                       
    generated_ceus = torch.clamp((noisy_images + 1) / 2, 0, 1).cpu()

           
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
              
        plt.subplot(3, num_samples, i+1)
        plt.imshow(batch['b_mode'][i].squeeze(), cmap='gray')
        plt.title("Input B-mode")
        plt.axis('off')
        
        # Mask
        plt.subplot(3, num_samples, i+num_samples+1)
        plt.imshow(batch['mask'][i].squeeze(), cmap='gray')
        plt.title("Target Mask")
        plt.axis('off')
        
                 
        plt.subplot(3, num_samples, i+num_samples*2+1)
        plt.imshow(generated_ceus[i].squeeze(), cmap='viridis')
        plt.title("Generated CEUS")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"samples/generated_samples_epoch_{epoch}.png")
    plt.close()

                   
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
    
          
    valid_triplets = []
    for b_path, m_path, c_path in zip(b_mode_paths, mask_paths, ceus_paths):
        if os.path.exists(b_path) and os.path.exists(m_path) and os.path.exists(c_path):
            valid_triplets.append((b_path, m_path, c_path))
    
    print(f"Found {len(valid_triplets)} valid image triplets")
    if len(valid_triplets) == 0:
        raise ValueError("No valid image triplets found. Please check your data paths.")
    
                 
    dataset = UltrasoundPairDataset(
        [p[0] for p in valid_triplets],
        [p[1] for p in valid_triplets],
        [p[2] for p in valid_triplets]
    )
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
        num_training_steps=len(dataloader) * epochs
    )
    
          
    print(f"Starting training for {epochs} epochs...")
    trained_model, train_log = train_model(
        model, dataloader, optimizer, lr_scheduler, 
        noise_scheduler, device, epochs, save_dir="models"
    )
    
    print("Training completed.")

         
def generate_from_bmode(datapath, savepath, model_path="models/final_diffusion_model.pth", device="cuda", num_steps=1000):
          
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
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    image_paths = os.listdir(datapath)
    for img_name in image_paths:
        b_mode_image_path = os.path.join(datapath, img_name)
        mask_image_path = os.path.join("Mask", img_name)                   
        
              
        b_mode_img = Image.open(b_mode_image_path).convert('L')
        mask_img = Image.open(mask_image_path).convert('L')
        
              
        b_mode_tensor = transform(b_mode_img).unsqueeze(0).to(device)
        mask_tensor = mask_transform(mask_img).unsqueeze(0).to(device)
        
               
        sample_shape = (1, 1, 256, 256)
        noisy_image = torch.randn(sample_shape, device=device)
        
               
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        
              
        step_interval = noise_scheduler.config.num_train_timesteps // num_steps
        timesteps = list(reversed(range(0, noise_scheduler.config.num_train_timesteps, step_interval)))
        
                 
        denoise_pbar = tqdm(timesteps, desc="Generating CEUS image")
        
                  
        with torch.inference_mode():
            for t in denoise_pbar:
                timestep = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model(noisy_image, timestep, b_mode_tensor, mask_tensor)
                noisy_image = noise_scheduler.step(noise_pred, t, noisy_image).prev_sample
                denoise_pbar.set_postfix({"step": t})
        
             
        generated_ceus = (noisy_image.squeeze().cpu().detach().numpy() + 1) / 2
        generated_ceus = np.clip(generated_ceus, 0, 1)

              
        resized_img = cv2.resize(generated_ceus, b_mode_img.size, interpolation=cv2.INTER_LINEAR)
        resized_image_rgb = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(savepath, img_name), np.uint8(resized_image_rgb * 255))

if __name__ == "__main__":
    main()