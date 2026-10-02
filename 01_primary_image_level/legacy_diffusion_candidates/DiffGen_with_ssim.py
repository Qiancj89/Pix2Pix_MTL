import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDPMPipeline, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights        
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import cv2

                             
class UltrasoundPairDataset(Dataset):
    def __init__(self, b_mode_paths, ceus_paths, cache_size=50):
        self.b_mode_paths = b_mode_paths
        self.ceus_paths = ceus_paths
        
                             
        self.b_mode_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
                            
            transforms.Normalize([0.5], [0.5])
        ])
        
                              
        self.ceus_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
                              
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.b_mode_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        try:
                        
            b_mode_img = Image.open(self.b_mode_paths[idx])
            if b_mode_img.mode != 'L':
                b_mode_img = b_mode_img.convert('L')
            b_mode_tensor = self.b_mode_transform(b_mode_img)
            
                          
            ceus_img = Image.open(self.ceus_paths[idx])
            if ceus_img.mode != 'RGB':
                ceus_img = ceus_img.convert('RGB')
            ceus_tensor = self.ceus_transform(ceus_img)
            
            item = {'b_mode': b_mode_tensor, 'ceus': ceus_tensor}
            
            if len(self.cache) < self.cache_size:
                self.cache[idx] = item
            
            return item
        except Exception as e:
            print(f"Error loading image {self.b_mode_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

                         
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, sample_size=256):
        super().__init__()
                                        
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=3,                     
            out_channels=3,                      
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", 
                "AttnDownBlock2D",
                "DownBlock2D", 
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
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
                                              
        try:
                                        
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        except TypeError:
                   
            self.vgg = vgg16(pretrained=True).features[:16].eval()
            
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, noisy_images, timesteps, condition_images):
                    
        condition_embeds = self.condition_encoder(condition_images).unsqueeze(1)
        return self.unet(noisy_images, timesteps, encoder_hidden_states=condition_embeds).sample
    
    def perceptual_loss(self, x, y):
                        
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return F.l1_loss(x_features, y_features)

                             
def train_model(model, dataloader, optimizer, scheduler, noise_scheduler, device, epochs=100, save_dir="models"):
    model.train()
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    scaler = torch.amp.GradScaler('cuda')
    epoch_losses = []
    epoch_times = []
    
    pbar = tqdm(total=epochs * len(dataloader), desc="Training Progress", unit="batch")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            b_mode = batch['b_mode'].to(device)       
            ceus = batch['ceus'].to(device)          
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                     (b_mode.size(0),), device=device).long()
            
            with torch.autocast('cuda'):
                              
                noise = torch.randn_like(ceus)
                noisy_images = noise_scheduler.add_noise(ceus, noise, timesteps)
                
                              
                noise_pred = model(noisy_images, timesteps, b_mode)
                
                           
                mse_loss = F.mse_loss(noise_pred, noise)
                
                                
                           
                with torch.no_grad():
                    denoised = noise_scheduler.step(noise_pred, timesteps[0], noisy_images).prev_sample
                
                        
                percep_loss = model.perceptual_loss(denoised, ceus)
                
                      
                loss = mse_loss + percep_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            pbar.update(1)
            pbar.set_postfix({
                "epoch": f"{epoch+1}/{epochs}",
                "batch_loss": f"{loss.item():.4f}",
                "mse_loss": f"{mse_loss.item():.4f}",
                "percep_loss": f"{percep_loss.item():.4f}"
            })
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        
                             
        if (epoch + 1) % 2 == 0:
            model_path = os.path.join(save_dir, f"diffusion_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            
                  
            generate_samples(model, dataloader, noise_scheduler, device, epoch+1, num_samples=4, use_ddim=True)
    
    pbar.close()
    
    final_path = os.path.join(save_dir, "final_diffusion_model.pth")
    torch.save(model.state_dict(), final_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    
    return model

                     
@torch.no_grad()
def generate_samples(model, dataloader, noise_scheduler, device, epoch, num_samples=4, use_ddim=True):
    model.eval()
    batch = next(iter(dataloader))
    b_mode = batch['b_mode'][:num_samples].to(device)
    real_ceus = batch['ceus'][:num_samples]
    
                   
    if use_ddim:
                         
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
                
        steps = 50
        ddim_scheduler.set_timesteps(steps, device=device)
        scheduler = ddim_scheduler
    else:
        scheduler = noise_scheduler
                
        steps = noise_scheduler.config.num_train_timesteps
        scheduler.set_timesteps(steps, device=device)
    
              
    noisy_images = torch.randn((num_samples, 3, 256, 256), device=device)

           
    pbar = tqdm(total=steps, desc="Generating Samples", leave=False)
    
                   
    for t in scheduler.timesteps:
                   
        timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
              
        noise_pred = model(noisy_images, timestep, b_mode)
        
              
        noisy_images = scheduler.step(noise_pred, t, noisy_images).prev_sample
        
               
        pbar.update(1)
    
    pbar.close()
    
                          
    generated_ceus = noisy_images.cpu()
    
            
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
                    
    generated_ceus = denormalize(generated_ceus, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
                   
    real_ceus_denorm = denormalize(real_ceus.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
               
    b_mode_denorm = denormalize(batch['b_mode'][:num_samples].clone(), [0.5], [0.5])
    
           
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
                    
        plt.subplot(3, num_samples, i+1)
        plt.imshow(b_mode_denorm[i].squeeze(), cmap='gray')
        plt.title("Input B-mode")
        plt.axis('off')
        
                       
        plt.subplot(3, num_samples, i+num_samples+1)
        ceus_img = generated_ceus[i].permute(1, 2, 0).numpy()
        plt.imshow(ceus_img)
        plt.title("Generated CEUS")
        plt.axis('off')
        
                      
        plt.subplot(3, num_samples, i+2*num_samples+1)
        real_img = real_ceus_denorm[i].permute(1, 2, 0).numpy()
        plt.imshow(real_img)
        plt.title("Real CEUS")
        plt.axis('off')
    
    plt.suptitle(f"Epoch {epoch} Comparison")
    plt.tight_layout()
    plt.savefig(f"samples/comparison_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
                   
    for i in range(num_samples):
        ceus_img = generated_ceus[i].permute(1, 2, 0).numpy() * 255
        ceus_img = ceus_img.astype(np.uint8)
        Image.fromarray(ceus_img).save(f"samples/generated_ceus_{epoch}_{i}.png")

        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
                    
    batch_size = 4
    learning_rate = 2e-4
    epochs = 200
    sample_size = 256
    
          
    b_mode_paths = []
    ceus_paths = []
    for i in range(0, len(os.listdir('B_mode'))):
        b_mode_paths.append('B_mode/'+os.listdir('B_mode')[i])
        ceus_paths.append('CEUS/'+os.listdir('CEUS')[i])
    b_mode_paths = np.array(b_mode_paths, dtype='<U32')
    ceus_paths = np.array(ceus_paths, dtype='<U32')
    
           
    valid_pairs = []
    for b_path, c_path in zip(b_mode_paths, ceus_paths):
        if os.path.exists(b_path) and os.path.exists(c_path):
            valid_pairs.append((b_path, c_path))
    
    print(f"Found {len(valid_pairs)} valid image pairs")
    if not valid_pairs:
        raise ValueError("No valid image pairs found. Please check data paths.")
    
           
    dataset = UltrasoundPairDataset(
        [p[0] for p in valid_pairs],
        [p[1] for p in valid_pairs],
        cache_size=100
    )
    
           
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
           
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
    
        
    print(f"Starting training for B-mode to CEUS conversion...")
    trained_model = train_model(
        model, dataloader, optimizer, lr_scheduler, 
        noise_scheduler, device, epochs
    )

    print("Training completed and pipeline saved.")

                      
def generate_from_bmode(b_mode_image_path, model_path="models/final_diffusion_model.pth", device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU instead")
    
    model = ConditionalDiffusionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
                 
    b_mode_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
                                                             
    ])
    
    b_mode_img = Image.open(b_mode_image_path)
    if b_mode_img.mode != 'L':
        b_mode_img = b_mode_img.convert('L')
    b_mode_tensor = b_mode_transform(b_mode_img).unsqueeze(0).to(device)
    
              
    noisy_image = torch.randn((1, 3, 256, 256), device=device)
    
                
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=False
    )
    
                    
    steps = 50
    for t in tqdm(reversed(range(0, ddim_scheduler.config.num_train_timesteps, 
                               ddim_scheduler.config.num_train_timesteps // steps)), 
                 desc="Generating CEUS"):
        timestep = torch.full((1,), t, device=device, dtype=torch.long)
        noise_pred = model(noisy_image, timestep, b_mode_tensor)
        noisy_image = ddim_scheduler.step(noise_pred, t, noisy_image).prev_sample
    
                
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
    generated_ceus = denormalize(noisy_image.squeeze(0).cpu(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    generated_ceus = generated_ceus.permute(1, 2, 0).numpy()
    
         
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(b_mode_img, cmap='gray')
    plt.title("Input B-mode")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(generated_ceus)
    plt.title("Generated CEUS")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("b_mode_to_ceus_result.png", dpi=300, bbox_inches='tight')
    
                 
    generated_ceus_img = (generated_ceus * 255).astype(np.uint8)
    Image.fromarray(generated_ceus_img).save("generated_ceus.png")
    print("Generated CEUS image saved as generated_ceus.png")
    
    return generated_ceus

if __name__ == "__main__":
    main()
                      
    # generate_from_bmode("new_bmode_image.png")