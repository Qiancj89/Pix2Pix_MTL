import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from tqdm import tqdm
import random
import math

# Optimizations for RTX 4090
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

class AdvancedMedicalAugmentation:
    """Advanced medical image augmentation with edge preservation"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, bmode_img: Image.Image, ceus_img: Image.Image):
        # Apply same transformation to both images
        if random.random() < self.prob:
            # Random rotation (-5 to 5 degrees) - reduced for better edge preservation
            angle = random.uniform(-5, 5)
            bmode_img = bmode_img.rotate(angle, fillcolor=0, resample=Image.BICUBIC)
            ceus_img = ceus_img.rotate(angle, fillcolor=0, resample=Image.BICUBIC)
        
        if random.random() < self.prob:
            # Random horizontal flip
            bmode_img = bmode_img.transpose(Image.FLIP_LEFT_RIGHT)
            ceus_img = ceus_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if random.random() < self.prob:
            # Brightness adjustment (more conservative for medical images)
            factor = random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Brightness(bmode_img)
            bmode_img = enhancer.enhance(factor)
            enhancer = ImageEnhance.Brightness(ceus_img)
            ceus_img = enhancer.enhance(factor)
        
        if random.random() < self.prob:
            # Contrast adjustment (more conservative)
            factor = random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Contrast(bmode_img)
            bmode_img = enhancer.enhance(factor)
            enhancer = ImageEnhance.Contrast(ceus_img)
            ceus_img = enhancer.enhance(factor)
        
        if random.random() < self.prob:
            # Random crop and resize (less aggressive)
            width, height = bmode_img.size
            crop_size = random.uniform(0.9, 1.0)
            new_w, new_h = int(width * crop_size), int(height * crop_size)
            
            left = random.randint(0, width - new_w)
            top = random.randint(0, height - new_h)
            
            bmode_img = bmode_img.crop((left, top, left + new_w, top + new_h))
            ceus_img = ceus_img.crop((left, top, left + new_w, top + new_h))
            
            # Use high-quality resampling
            bmode_img = bmode_img.resize((width, height), Image.LANCZOS)
            ceus_img = ceus_img.resize((width, height), Image.LANCZOS)
        
        return bmode_img, ceus_img

class UltrasoundDatasetAugmented(Dataset):
    """Enhanced dataset with medical image augmentation"""
    def __init__(self, bmode_dir: str, ceus_dir: str, size: tuple = (256, 256), 
                 augment: bool = True, augment_factor: int = 4):
        self.bmode_dir = bmode_dir
        self.ceus_dir = ceus_dir
        self.size = size
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1
        
        self.image_names = [f for f in os.listdir(bmode_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        # Medical augmentation
        self.med_aug = AdvancedMedicalAugmentation(prob=0.7)
        
        # Transform for B-mode (grayscale input)
        self.bmode_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Transform for CEUS (3-channel output)
        self.ceus_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_names) * self.augment_factor
    
    def __getitem__(self, idx):
        # Get base image index
        base_idx = idx // self.augment_factor
        img_name = self.image_names[base_idx]
        
        # Load images
        bmode_img = Image.open(os.path.join(self.bmode_dir, img_name))
        ceus_img = Image.open(os.path.join(self.ceus_dir, img_name))

        # Ensure CEUS is RGB (3-channel)
        if ceus_img.mode != 'RGB':
            ceus_img = ceus_img.convert('RGB')
        
        # Apply augmentation
        if self.augment:
            bmode_img, ceus_img = self.med_aug(bmode_img, ceus_img)
        
        # Convert to tensors with different transforms
        bmode_tensor = self.bmode_transform(bmode_img)  # [1, H, W]
        ceus_tensor = self.ceus_transform(ceus_img)     # [3, H, W]
        
        return bmode_tensor, ceus_tensor

# Spectral Normalization for stable training
def spectral_norm(module, use_spectral_norm=True):
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module

# Self-Attention Module for better feature capture
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Queries, keys, values
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        proj_value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

# Enhanced Generator with attention and residual connections
class EnhancedGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Encoder with residual blocks
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect")),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.down1 = self._residual_block(features, features * 2, 4, 2, 1)
        self.down2 = self._residual_block(features * 2, features * 4, 4, 2, 1)
        self.down3 = self._residual_block(features * 4, features * 8, 4, 2, 1)
        self.down4 = self._residual_block(features * 8, features * 8, 4, 2, 1)
        self.down5 = self._residual_block(features * 8, features * 8, 4, 2, 1)
        self.down6 = self._residual_block(features * 8, features * 8, 4, 2, 1)
        
        # Self-attention at the bottleneck
        if use_attention:
            self.attention = SelfAttention(features * 8)
        
        # Bottleneck with residual connection
        self.bottleneck = nn.Sequential(
            spectral_norm(nn.Conv2d(features * 8, features * 8, 4, 2, 1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(features * 8, features * 8, 3, 1, 1)),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with residual blocks
        self.up1 = self._residual_upblock(features * 8, features * 8, 4, 2, 1, dropout=True)
        self.up2 = self._residual_upblock(features * 8 * 2, features * 8, 4, 2, 1, dropout=True)
        self.up3 = self._residual_upblock(features * 8 * 2, features * 8, 4, 2, 1, dropout=True)
        self.up4 = self._residual_upblock(features * 8 * 2, features * 8, 4, 2, 1)
        self.up5 = self._residual_upblock(features * 8 * 2, features * 4, 4, 2, 1)
        self.up6 = self._residual_upblock(features * 4 * 2, features * 2, 4, 2, 1)
        self.up7 = self._residual_upblock(features * 2 * 2, features, 4, 2, 1)
        
        # Enhanced final layer with edge enhancement
        self.final = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(features * 2, features, 4, 2, 1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(features, out_channels, 3, 1, 1)),
            nn.Tanh()
        )

    def _residual_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                                  bias=False, padding_mode="reflect")),
            nn.InstanceNorm2d(out_channels),  # Instance norm often better than batch norm for images
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, 
                                  bias=False, padding_mode="reflect")),
            nn.InstanceNorm2d(out_channels)
        )

    def _residual_upblock(self, in_channels, out_channels, kernel_size, stride, padding, dropout=False):
        layers = [
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.3))  # Reduced dropout
        
        layers.extend([
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, padding_mode="reflect")),
            nn.InstanceNorm2d(out_channels)
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder with skip connections
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        bottleneck = self.bottleneck(d7)
        
        # Apply self-attention at bottleneck
        if self.use_attention:
            bottleneck = self.attention(bottleneck)
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final(torch.cat([u7, d1], 1))

# Enhanced Discriminator with spectral normalization
class EnhancedDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect")),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv1 = self._block(features, features * 2, 4, 2, 1)
        self.conv2 = self._block(features * 2, features * 4, 4, 2, 1)
        self.conv3 = self._block(features * 4, features * 8, 4, 1, 1)
        
        # Self-attention for better discrimination
        if use_attention:
            self.attention = SelfAttention(features * 8)
        
        self.conv4 = self._block(features * 8, features * 8, 4, 1, 1)
        self.final = spectral_norm(nn.Conv2d(features * 8, 1, 4, 1, 1))

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                                  bias=False, padding_mode="reflect")),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        x = self.conv4(x)
        return self.final(x)

# Perceptual Loss for better image quality
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pre-trained VGG for perceptual loss
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()

    def forward(self, fake, real):
        # Convert to 3-channel if needed
        if fake.size(1) == 1:
            fake = fake.repeat(1, 3, 1, 1)
        if real.size(1) == 1:
            real = real.repeat(1, 3, 1, 1)
            
        fake_features = self.feature_extractor(fake)
        real_features = self.feature_extractor(real)
        
        return self.mse_loss(fake_features, real_features)

# Edge Loss for preserving sharp details
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.mse_loss = nn.MSELoss()

    def forward(self, fake, real):
        # Compute gradients
        fake_grad_x = F.conv2d(fake.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        fake_grad_y = F.conv2d(fake.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        fake_grad = torch.sqrt(fake_grad_x**2 + fake_grad_y**2 + 1e-6)
        
        real_grad_x = F.conv2d(real.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        real_grad_y = F.conv2d(real.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        real_grad = torch.sqrt(real_grad_x**2 + real_grad_y**2 + 1e-6)
        
        return self.mse_loss(fake_grad, real_grad)

# SSIM Metric for evaluation (higher is better)
def compute_ssim_metric(img1, img2, window_size=11, val_range=2.0):
    """Compute SSIM metric (returns value between 0 and 1, higher is better)"""
    _, channel, height, width = img1.size()
    
    # Handle different channel sizes
    if img1.size(1) != img2.size(1):
        if img1.size(1) == 3 and img2.size(1) == 1:
            img2 = img2.repeat(1, 3, 1, 1)
        elif img1.size(1) == 1 and img2.size(1) == 3:
            img1 = img1.repeat(1, 3, 1, 1)
    
    # Create Gaussian window
    def _gaussian(window_size, sigma=1.5):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    
    _1D_window = _gaussian(window_size).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(img1.device).type_as(img1)
    
    # SSIM calculation
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()  # Return SSIM metric (higher is better)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})
            return True
        return False

# Enhanced training function with early stopping
def train_enhanced_pix2pix(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 200,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    lambda_l1: float = 100,
    lambda_perceptual: float = 1,
    lambda_edge: float = 10,
    save_path: str = "models/enhanced_pix2pix_model_wo_ssim.pth",
    patience: int = 30
):
    """Enhanced training with early stopping (SSIM loss removed)"""
    
    # Optimizers
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 ** ((epoch - warmup_epochs) // 50)
    
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda)
    
    # Loss functions
    adversarial_loss = nn.MSELoss()  # LSGAN loss for more stable training
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    edge_loss = EdgeLoss().to(device)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, restore_best_weights=True)
    
    best_val_ssim = -1.0
    
    print(f"Starting training with early stopping (patience={patience})...")
    print(f"Loss weights: L1={lambda_l1}, Perceptual={lambda_perceptual}, Edge={lambda_edge}")
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        gen_losses, disc_losses = [], []
        l1_losses, perceptual_losses, edge_losses = [], [], []
        
        # Training loop
        for i, (bmode, ceus) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            bmode, ceus = bmode.to(device), ceus.to(device)
            
            # Train Discriminator less frequently for better generator training
            if i % 2 == 0:
                opt_disc.zero_grad()
                
                with torch.autocast('cuda'):
                    # Real pairs
                    real_pred = discriminator(bmode, ceus)
                    real_labels = torch.ones_like(real_pred) * 0.9  # Label smoothing
                    real_loss = adversarial_loss(real_pred, real_labels)
                    
                    # Fake pairs
                    fake_ceus = generator(bmode)
                    fake_pred = discriminator(bmode, fake_ceus.detach())
                    fake_labels = torch.zeros_like(fake_pred) + 0.1  # Label smoothing
                    fake_loss = adversarial_loss(fake_pred, fake_labels)
                    
                    disc_loss = (real_loss + fake_loss) / 2
                
                scaler.scale(disc_loss).backward()
                scaler.step(opt_disc)
                disc_losses.append(disc_loss.item())
            
            # Train Generator
            opt_gen.zero_grad()
            
            with torch.autocast('cuda'):
                fake_ceus = generator(bmode)
                gen_pred = discriminator(bmode, fake_ceus)
                gen_labels = torch.ones_like(gen_pred)
                
                # Multiple loss components (without SSIM)
                adversarial_gen_loss = adversarial_loss(gen_pred, gen_labels)
                l1_gen_loss = l1_loss(fake_ceus, ceus)
                perceptual_gen_loss = perceptual_loss(fake_ceus, ceus)
                edge_gen_loss = edge_loss(fake_ceus, ceus)
                
                # Combined generator loss (SSIM loss removed)
                gen_loss = (adversarial_gen_loss + 
                           lambda_l1 * l1_gen_loss +
                           lambda_perceptual * perceptual_gen_loss +
                           lambda_edge * edge_gen_loss)
            
            scaler.scale(gen_loss).backward()
            scaler.step(opt_gen)
            scaler.update()
            
            # Log losses
            gen_losses.append(gen_loss.item())
            l1_losses.append(l1_gen_loss.item())
            perceptual_losses.append(perceptual_gen_loss.item())
            edge_losses.append(edge_gen_loss.item())
        
        # Validation
        generator.eval()
        val_l1_losses = []
        val_ssim_metrics = []
        
        with torch.no_grad():
            for bmode, ceus in val_loader:
                bmode, ceus = bmode.to(device), ceus.to(device)
                
                with torch.autocast('cuda'):
                    fake_ceus = generator(bmode)
                    val_l1 = l1_loss(fake_ceus, ceus)
                    val_ssim_metric = compute_ssim_metric(fake_ceus, ceus)
                
                val_l1_losses.append(val_l1.item())
                val_ssim_metrics.append(val_ssim_metric.item())
        
        # Compute averages
        avg_gen_loss = np.mean(gen_losses)
        avg_disc_loss = np.mean(disc_losses) if disc_losses else 0.0
        avg_val_l1 = np.mean(val_l1_losses)
        avg_val_ssim = np.mean(val_ssim_metrics)
        
        # Print comprehensive loss information
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Gen: {avg_gen_loss:.4f}, Disc: {avg_disc_loss:.4f}")
        print(f"  Train - L1: {np.mean(l1_losses):.4f}, Perceptual: {np.mean(perceptual_losses):.4f}")
        print(f"  Train - Edge: {np.mean(edge_losses):.4f}")
        print(f"  Val - L1: {avg_val_l1:.4f}, SSIM Metric: {avg_val_ssim:.4f}")
        print(f"  LR: {scheduler_g.get_last_lr()[0]:.6f}")
        
        # Update learning rates
        scheduler_g.step()
        scheduler_d.step()
        
        # Save best model based on SSIM metric (higher is better)
        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch,
                'val_ssim': avg_val_ssim,
                'val_l1': avg_val_l1
            }, save_path)
            print(f"  New best model saved! SSIM: {avg_val_ssim:.4f}")
        
        # Early stopping check (use validation L1 loss for early stopping)
        val_loss_for_stopping = avg_val_l1  # Use L1 loss for early stopping
        if early_stopping(val_loss_for_stopping, generator):
            print(f"Early stopping triggered after epoch {epoch+1}")
            print(f"Best SSIM achieved: {best_val_ssim:.4f}")
            break

# Post-processing for sharper images
@torch.no_grad()
def generate_sharp_ceus(generator, bmode_path: str, output_path: str, apply_sharpening: bool = True):
    """Generate CEUS with post-processing for sharpness"""
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load and preprocess
    bmode_img = Image.open(bmode_path)
    bmode_tensor = transform(bmode_img).unsqueeze(0).to(device)
    
    # Generate
    generator.eval()
    with torch.autocast('cuda'):
        fake_ceus = generator(bmode_tensor)
    
    # Post-processing - proper denormalization
    fake_ceus = fake_ceus.squeeze(0).cpu()
    fake_ceus = (fake_ceus * 0.5) + 0.5  # Denormalize from [-1, 1] to [0, 1]
    fake_ceus = torch.clamp(fake_ceus, 0, 1)
    fake_ceus_img = transforms.ToPILImage()(fake_ceus)
    
    # Apply sharpening filter if requested
    if apply_sharpening:
        enhancer = ImageEnhance.Sharpness(fake_ceus_img)
        fake_ceus_img = enhancer.enhance(1.1)  # Gentle sharpening
    
    fake_ceus_img.save(output_path, quality=95)
    return fake_ceus_img

# Single image inference function
@torch.no_grad()
def inference_single_image(model_path: str, bmode_path: str, output_path: str, apply_sharpening: bool = True):
    """
    Load model and perform inference on a single B-mode image
    
    Args:
        model_path: Path to the saved model checkpoint
        bmode_path: Path to the input B-mode image
        output_path: Path to save the generated CEUS image
        apply_sharpening: Whether to apply post-processing sharpening
    """
    # Initialize generator
    generator = EnhancedGenerator(in_channels=1, out_channels=3, use_attention=True).to(device)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Validation SSIM: {checkpoint.get('val_ssim', 'N/A'):.4f}")
    print(f"Validation L1: {checkpoint.get('val_l1', 'N/A'):.4f}")
    
    # Generate CEUS image
    generated_img = generate_sharp_ceus(generator, bmode_path, output_path, apply_sharpening)
    
    print(f"Generated CEUS image saved to: {output_path}")
    return generated_img

# Evaluation function to compute metrics
@torch.no_grad()
def evaluate_model(generator, test_loader, device):
    """Comprehensive model evaluation with multiple metrics"""
    generator.eval()
    
    metrics = {
        'ssim': [],
        'l1': [],
        'mse': [],
        'psnr': []
    }
    
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    
    for bmode, ceus in tqdm(test_loader, desc="Evaluating"):
        bmode, ceus = bmode.to(device), ceus.to(device)
        
        with torch.autocast('cuda'):
            fake_ceus = generator(bmode)
        
        # Compute metrics
        ssim_val = compute_ssim_metric(fake_ceus, ceus).item()
        l1_val = l1_loss(fake_ceus, ceus).item()
        mse_val = mse_loss(fake_ceus, ceus).item()
        psnr_val = -10 * math.log10(mse_val + 1e-8)
        
        metrics['ssim'].append(ssim_val)
        metrics['l1'].append(l1_val)
        metrics['mse'].append(mse_val)
        metrics['psnr'].append(psnr_val)
    
    # Compute averages
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    print(f"\nEvaluation Results:")
    print(f"  SSIM: {avg_metrics['ssim']:.4f}")
    print(f"  L1 Loss: {avg_metrics['l1']:.4f}")
    print(f"  MSE: {avg_metrics['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
    
    return avg_metrics

# Example usage with train/validation split, no augmentation, batch size 4, and early stopping
if __name__ == "__main__":
    print(f"Using device: {device}")
    '''
    # Create full dataset from training directory only (with augmentation)
    full_dataset = UltrasoundDatasetAugmented(
        "data/train/bmode", 
        "data/train/ceus",
        augment=True,
        augment_factor=8  # 8x augmentation for better training
    )
    
    # Split dataset into train and validation (80/20 split)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    print(f"Total dataset size: {total_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create data loaders with batch size 4
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize enhanced models
    generator = EnhancedGenerator(
        in_channels=1, 
        out_channels=3, 
        use_attention=True
    ).to(device)
    
    discriminator = EnhancedDiscriminator(
        in_channels=4, 
        use_attention=True
    ).to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train with early stopping (SSIM loss removed)
    train_enhanced_pix2pix(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,
        lr_g=2e-4,
        lr_d=2e-4,
        lambda_l1=100,
        lambda_perceptual=1,
        lambda_edge=10,
        save_path="models/enhanced_pix2pix_model_wo_ssim.pth",
        patience=30  # Early stopping patience
    )
    
    # Example of single image inference after training
    print("\nExample single image inference:")
    print("inference_single_image('models/enhanced_pix2pix_model_wo_ssim.pth', 'data/val/bmode/160_ZhuWenling1.jpg', 'test_images/output_ceus_wo_ssim.png')")
    '''
    # Uncomment the following lines to test inference with a specific image
    img_files = os.listdir('TestData/bmode/')
    os.makedirs('TestData/enhanced_pix2pix_model_wo_ssim', exist_ok=True)
    for i in range(0, len(img_files)):
        inference_single_image(
            model_path='models/enhanced_pix2pix_model_wo_ssim.pth', 
            bmode_path='TestData/bmode/'+img_files[i],  
            output_path='TestData/enhanced_pix2pix_model_wo_ssim/'+img_files[i],
            apply_sharpening=True
        )
    
    # Example evaluation after training
    # print("\nEvaluating model performance...")
    # final_metrics = evaluate_model(generator, val_loader, device)