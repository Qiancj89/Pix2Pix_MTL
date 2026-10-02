import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
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

                       
class MaskedContrastiveDiffusionModel(nn.Module):
    def __init__(self, sample_size=256, projection_dim=128):
        super().__init__()
        
                            
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
        
                        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
        
                            
        self.condition_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
                  
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
        
                
        try:
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        except:
            self.vgg = vgg16(pretrained=True).features[:16].eval()
            
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, noisy_images, timesteps, condition_images):
                    
        condition_embeds = self.condition_encoder(condition_images).unsqueeze(1)
        return self.unet(noisy_images, timesteps, encoder_hidden_states=condition_embeds).sample
    
    def encode_condition(self, condition_images):
                
        features = self.condition_encoder(condition_images)
        return self.condition_projection(features)
    
    def encode_image(self, images):
              
        features = self.image_encoder(images)
        return self.image_projection(features)
    
    def perceptual_loss_with_mask(self, x, y, mask, inside_weight=1.0, outside_weight=0.5):
        




           
              
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        
                          
        _, _, h, w = x_features.shape
        mask_resized = F.interpolate(mask, size=(h, w), mode='nearest')
        
                     
        l1_loss = F.l1_loss(x_features, y_features, reduction='none')
        
                          
        inside_loss = (l1_loss * mask_resized).sum() / (mask_resized.sum() + 1e-8)
        outside_loss = (l1_loss * (1 - mask_resized)).sum() / ((1 - mask_resized).sum() + 1e-8)
        
              
        total_loss = inside_weight * inside_loss + outside_weight * outside_loss
        
        return total_loss, inside_loss, outside_loss
    
    def contrastive_loss(self, condition_embeds, image_embeds, temperature=0.1):
                
        batch_size = condition_embeds.size(0)
        
                 
        condition_embeds = F.normalize(condition_embeds, p=2, dim=1)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        
                 
        logits = torch.mm(condition_embeds, image_embeds.t()) / temperature
        
                          
        labels = torch.arange(batch_size, device=condition_embeds.device)
        
                 
        loss_cond = F.cross_entropy(logits, labels)
        loss_img = F.cross_entropy(logits.t(), labels)
        
        return (loss_cond + loss_img) / 2

                     
class MaskedUltrasoundPairDataset(Dataset):
    def __init__(self, b_mode_paths, ceus_paths, mask_paths, cache_size=50):
        self.b_mode_paths = b_mode_paths
        self.ceus_paths = ceus_paths
        self.mask_paths = mask_paths
        
                     
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
        
                
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
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
            
                    
            mask_img = Image.open(self.mask_paths[idx])
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
            mask_tensor = self.mask_transform(mask_img)
            
            item = {
                'b_mode': b_mode_tensor, 
                'ceus': ceus_tensor,
                'mask': mask_tensor
            }
            
            if len(self.cache) < self.cache_size:
                self.cache[idx] = item
            
            return item
        except Exception as e:
            print(f"Error loading image {self.b_mode_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

         
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
        total_diff_loss = 0.0
        total_contrast_loss = 0.0
        total_percep_loss = 0.0
        total_inside_loss = 0.0
        total_outside_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            b_mode = batch['b_mode'].to(device)       
            ceus = batch['ceus'].to(device)          
            mask = batch['mask'].to(device)     # mask
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                     (b_mode.size(0),), device=device).long()
            
            with torch.autocast('cuda'):
                              
                noise = torch.randn_like(ceus)
                noisy_images = noise_scheduler.add_noise(ceus, noise, timesteps)
                
                              
                noise_pred = model(noisy_images, timesteps, b_mode)
                
                           
                mse_loss = F.mse_loss(noise_pred, noise)
                
                        
                condition_embeds = model.encode_condition(b_mode)
                image_embeds = model.encode_image(ceus)
                contrast_loss = model.contrastive_loss(condition_embeds, image_embeds)
                
                              
                with torch.no_grad():
                    denoised = noise_scheduler.step(noise_pred, timesteps[0], noisy_images).prev_sample
                
                percep_loss, inside_loss, outside_loss = model.perceptual_loss_with_mask(denoised, ceus, mask)
                
                      
                loss = mse_loss + 0.5 * contrast_loss + 0.1 * percep_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_diff_loss += mse_loss.item()
            total_contrast_loss += contrast_loss.item()
            total_percep_loss += percep_loss.item()
            total_inside_loss += inside_loss.item()
            total_outside_loss += outside_loss.item()
            batch_count += 1
            
            pbar.update(1)
            pbar.set_postfix({
                "epoch": f"{epoch+1}/{epochs}",
                "loss": f"{loss.item():.4f}",
                "diff": f"{mse_loss.item():.4f}",
                "contrast": f"{contrast_loss.item():.4f}",
                "percep": f"{percep_loss.item():.4f}",
                "inside": f"{inside_loss.item():.4f}",
                "outside": f"{outside_loss.item():.4f}"
            })
        
                      
        avg_loss = total_loss / len(dataloader)
        avg_diff_loss = total_diff_loss / len(dataloader)
        avg_contrast_loss = total_contrast_loss / len(dataloader)
        avg_percep_loss = total_percep_loss / len(dataloader)
        avg_inside_loss = total_inside_loss / len(dataloader)
        avg_outside_loss = total_outside_loss / len(dataloader)
        
        epoch_time = time.time() - epoch_start
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        
                             
        if (epoch + 1) % 2 == 0:
            model_path = os.path.join(save_dir, f"masked_contrastive_diffusion_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            
                  
            generate_samples(model, dataloader, noise_scheduler, device, epoch+1, num_samples=4, use_ddim=True)
    
    pbar.close()
    
    final_path = os.path.join(save_dir, "masked_contrastive_diffusion_final.pth")
    torch.save(model.state_dict(), final_path)
    
            
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(epoch_losses)
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 3, 2)
    plt.plot([avg_diff_loss for _ in range(len(epoch_losses))])
    plt.title("Diffusion Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 3, 3)
    plt.plot([avg_contrast_loss for _ in range(len(epoch_losses))])
    plt.title("Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 3, 4)
    plt.plot([avg_percep_loss for _ in range(len(epoch_losses))])
    plt.title("Perceptual Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 3, 5)
    plt.plot([avg_inside_loss for _ in range(len(epoch_losses))])
    plt.title("Inside Mask Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 3, 6)
    plt.plot([avg_outside_loss for _ in range(len(epoch_losses))])
    plt.title("Outside Mask Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    
    return model

           
@torch.no_grad()
def generate_samples(model, dataloader, noise_scheduler, device, epoch, num_samples=4, use_ddim=True):
    model.eval()
    batch = next(iter(dataloader))
    b_mode = batch['b_mode'][:num_samples].to(device)
    real_ceus = batch['ceus'][:num_samples]
    masks = batch['mask'][:num_samples]
    
                   
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
    
           
    plt.figure(figsize=(20, 15))
    for i in range(num_samples):
                    
        plt.subplot(4, num_samples, i+1)
        plt.imshow(b_mode_denorm[i].squeeze(), cmap='gray')
        plt.title("Input B-mode")
        plt.axis('off')
        
        # Mask
        plt.subplot(4, num_samples, i+num_samples+1)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Mask")
        plt.axis('off')
        
                       
        plt.subplot(4, num_samples, i+2*num_samples+1)
        ceus_img = generated_ceus[i].permute(1, 2, 0).numpy()
        plt.imshow(ceus_img)
        plt.title("Generated CEUS")
        plt.axis('off')
        
                      
        plt.subplot(4, num_samples, i+3*num_samples+1)
        real_img = real_ceus_denorm[i].permute(1, 2, 0).numpy()
        plt.imshow(real_img)
        plt.title("Real CEUS")
        plt.axis('off')
    
    plt.suptitle(f"Epoch {epoch} Comparison - Masked Contrastive Diffusion")
    plt.tight_layout()
    plt.savefig(f"samples/masked_contrastive_comparison_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
                   
    for i in range(num_samples):
        ceus_img = generated_ceus[i].permute(1, 2, 0).numpy() * 255
        ceus_img = ceus_img.astype(np.uint8)
        Image.fromarray(ceus_img).save(f"samples/masked_contrastive_generated_ceus_{epoch}_{i}.png")

           
@torch.no_grad()
def generate_from_bmode(b_mode_image_path, mask_image_path, model_path="models/masked_contrastive_diffusion_final.pth", device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU instead")
    
          
    model = MaskedContrastiveDiffusionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
                 
    b_mode_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
             
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    b_mode_img = Image.open(b_mode_image_path)
    if b_mode_img.mode != 'L':
        b_mode_img = b_mode_img.convert('L')
    b_mode_tensor = b_mode_transform(b_mode_img).unsqueeze(0).to(device)
    
    mask_img = Image.open(mask_image_path)
    if mask_img.mode != 'L':
        mask_img = mask_img.convert('L')
    mask_tensor = mask_transform(mask_img).unsqueeze(0).to(device)
    
              
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
    
         
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(b_mode_img, cmap='gray')
    plt.title("Input B-mode")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(mask_tensor.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Mask")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(generated_ceus)
    plt.title("Generated CEUS")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
                 
    masked_generated = generated_ceus * mask_tensor.squeeze().cpu().numpy()[:, :, np.newaxis]
    plt.imshow(masked_generated)
    plt.title("Masked Generated CEUS")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("masked_contrastive_b_mode_to_ceus_result.png", dpi=300, bbox_inches='tight')
    
                 
    generated_ceus_img = (generated_ceus * 255).astype(np.uint8)
    Image.fromarray(generated_ceus_img).save("masked_contrastive_generated_ceus.png")
    print("Generated CEUS image saved as masked_contrastive_generated_ceus.png")
    
    return generated_ceus

        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
                    
    batch_size = 4
    learning_rate = 2e-4
    epochs = 200
    sample_size = 256
    
          
    b_mode_paths = []
    ceus_paths = []
    mask_paths = []
    for i in range(0, len(os.listdir('B_mode'))):
        b_mode_paths.append('B_mode/'+os.listdir('B_mode')[i])
        ceus_paths.append('CEUS/'+os.listdir('CEUS')[i])
        mask_paths.append('Mask/'+os.listdir('Mask')[i])
    b_mode_paths = np.array(b_mode_paths, dtype='<U32')
    ceus_paths = np.array(ceus_paths, dtype='<U32')
    mask_paths = np.array(mask_paths, dtype='<U32')
    
           
    valid_pairs = []
    for b_path, c_path, m_path in zip(b_mode_paths, ceus_paths, mask_paths):
        if os.path.exists(b_path) and os.path.exists(c_path) and os.path.exists(m_path):
            valid_pairs.append((b_path, c_path, m_path))
    
    print(f"Found {len(valid_pairs)} valid image pairs with masks")
    if not valid_pairs:
        raise ValueError("No valid image pairs with masks found. Please check data paths.")
    
           
    dataset = MaskedUltrasoundPairDataset(
        [p[0] for p in valid_pairs],
        [p[1] for p in valid_pairs],
        [p[2] for p in valid_pairs],
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
    
           
    model = MaskedContrastiveDiffusionModel(sample_size=sample_size)
    
           
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
    
        
    print(f"Starting training for Masked Contrastive Diffusion Model...")
    trained_model = train_model(
        model, dataloader, optimizer, lr_scheduler, 
        noise_scheduler, device, epochs
    )
    
    print("Training completed. Model saved.")
    




       

if __name__ == "__main__":
    main()