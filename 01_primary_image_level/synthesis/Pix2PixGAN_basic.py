import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
import random

# Optimizations for RTX 4090
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=20, min_delta=0.0, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
            verbose: Print messages when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best validation loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop

class MedicalAugmentation:
    """Medical image specific augmentation"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, bmode_img: Image.Image, ceus_img: Image.Image):
        # Apply same transformation to both images
        if random.random() < self.prob:
            # Random rotation (-10 to 10 degrees)
            angle = random.uniform(-10, 10)
            bmode_img = bmode_img.rotate(angle, fillcolor=0)
            ceus_img = ceus_img.rotate(angle, fillcolor=0)
        
        if random.random() < self.prob:
            # Random horizontal flip
            bmode_img = bmode_img.transpose(Image.FLIP_LEFT_RIGHT)
            ceus_img = ceus_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if random.random() < self.prob:
            # Brightness adjustment (medical images are sensitive)
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(bmode_img)
            bmode_img = enhancer.enhance(factor)
            enhancer = ImageEnhance.Brightness(ceus_img)
            ceus_img = enhancer.enhance(factor)
        
        if random.random() < self.prob:
            # Contrast adjustment
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(bmode_img)
            bmode_img = enhancer.enhance(factor)
            enhancer = ImageEnhance.Contrast(ceus_img)
            ceus_img = enhancer.enhance(factor)
        
        if random.random() < self.prob:
            # Random crop and resize (simulate different probe positions)
            width, height = bmode_img.size
            crop_size = random.uniform(0.85, 1.0)
            new_w, new_h = int(width * crop_size), int(height * crop_size)
            
            left = random.randint(0, width - new_w)
            top = random.randint(0, height - new_h)
            
            bmode_img = bmode_img.crop((left, top, left + new_w, top + new_h))
            ceus_img = ceus_img.crop((left, top, left + new_w, top + new_h))
            
            bmode_img = bmode_img.resize((width, height), Image.LANCZOS)
            ceus_img = ceus_img.resize((width, height), Image.LANCZOS)
        
        return bmode_img, ceus_img

class UltrasoundDatasetAugmented(Dataset):
    """Enhanced dataset with aggressive augmentation for small datasets"""
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
        self.med_aug = MedicalAugmentation(prob=0.7)
        
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

# Pix2Pix Generator (U-Net with skip connections)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super().__init__()
        
        # Encoder
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = self._block(features, features * 2, 4, 2, 1)
        self.down2 = self._block(features * 2, features * 4, 4, 2, 1)
        self.down3 = self._block(features * 4, features * 8, 4, 2, 1)
        self.down4 = self._block(features * 8, features * 8, 4, 2, 1)
        self.down5 = self._block(features * 8, features * 8, 4, 2, 1)
        self.down6 = self._block(features * 8, features * 8, 4, 2, 1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )
        
        # Decoder
        self.up1 = self._upblock(features * 8, features * 8, 4, 2, 1, dropout=True)
        self.up2 = self._upblock(features * 8 * 2, features * 8, 4, 2, 1, dropout=True)
        self.up3 = self._upblock(features * 8 * 2, features * 8, 4, 2, 1, dropout=True)
        self.up4 = self._upblock(features * 8 * 2, features * 8, 4, 2, 1)
        self.up5 = self._upblock(features * 8 * 2, features * 4, 4, 2, 1)
        self.up6 = self._upblock(features * 4 * 2, features * 2, 4, 2, 1)
        self.up7 = self._upblock(features * 2 * 2, features, 4, 2, 1)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _upblock(self, in_channels, out_channels, kernel_size, stride, padding, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
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
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final(torch.cat([u7, d1], 1))

# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, in_channels=4, features=64):  # 2 channels: B-mode + CEUS
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        self.conv1 = self._block(features, features * 2, 4, 2, 1)
        self.conv2 = self._block(features * 2, features * 4, 4, 2, 1)
        self.conv3 = self._block(features * 4, features * 8, 4, 1, 1)
        
        self.final = nn.Conv2d(features * 8, 1, 4, 1, 1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                     bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)  # Concatenate B-mode and CEUS
        x = self.initial(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.final(x)

# Training function optimized for small datasets with early stopping
def train_pix2pix(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 200,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    lambda_l1: float = 100,
    save_path: str = "models/pix2pix_model.pth",
    patience: int = 30,  # Early stopping patience
    min_delta: float = 0.001  # Minimum improvement threshold
):
    """Training with techniques for small datasets and early stopping"""
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    
    # Optimizers with weight decay for regularization
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999), weight_decay=1e-4)
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=50, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=50, gamma=0.5)
    
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        gen_losses, disc_losses = [], []
        
        for bmode, ceus in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            bmode, ceus = bmode.to(device), ceus.to(device)
            batch_size = bmode.size(0)
            
            # Train Discriminator
            opt_disc.zero_grad()
            
            with torch.autocast('cuda'):
                # Real pairs
                real_pred = discriminator(bmode, ceus)
                real_labels = torch.ones_like(real_pred)
                real_loss = adversarial_loss(real_pred, real_labels)
                
                # Fake pairs
                fake_ceus = generator(bmode)
                fake_pred = discriminator(bmode, fake_ceus.detach())
                fake_labels = torch.zeros_like(fake_pred)
                fake_loss = adversarial_loss(fake_pred, fake_labels)
                
                disc_loss = (real_loss + fake_loss) / 2
            
            scaler.scale(disc_loss).backward()
            scaler.step(opt_disc)
            
            # Train Generator
            opt_gen.zero_grad()
            
            with torch.autocast('cuda'):
                fake_ceus = generator(bmode)
                gen_pred = discriminator(bmode, fake_ceus)
                gen_labels = torch.ones_like(gen_pred)
                
                adversarial_gen_loss = adversarial_loss(gen_pred, gen_labels)
                l1_gen_loss = l1_loss(fake_ceus, ceus)
                
                gen_loss = adversarial_gen_loss + lambda_l1 * l1_gen_loss
            
            scaler.scale(gen_loss).backward()
            scaler.step(opt_gen)
            scaler.update()
            
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())
        
        # Validation
        generator.eval()
        val_losses = []
        
        with torch.no_grad():
            for bmode, ceus in val_loader:
                bmode, ceus = bmode.to(device), ceus.to(device)
                
                with torch.autocast('cuda'):
                    fake_ceus = generator(bmode)
                    val_loss = l1_loss(fake_ceus, ceus)
                
                val_losses.append(val_loss.item())
        
        avg_gen_loss = np.mean(gen_losses)
        avg_disc_loss = np.mean(disc_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}: Gen Loss: {avg_gen_loss:.4f}, "
              f"Disc Loss: {avg_disc_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler_g.step()
        scheduler_d.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"✓ New best model saved with validation loss: {avg_val_loss:.4f}")
        
        # Check early stopping
        if early_stopping(avg_val_loss, epoch):
            print(f"\n{'='*60}")
            print(f"Training stopped early at epoch {epoch+1}")
            print(f"Best model was at epoch {early_stopping.best_epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            print(f"{'='*60}\n")
            break
    
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")

# Fast inference function
@torch.no_grad()
def generate_ceus_fast(generator, bmode_path: str, output_path: str):
    """Ultra-fast inference for Pix2Pix"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load and preprocess
    bmode_img = Image.open(bmode_path)
    bmode_tensor = transform(bmode_img).unsqueeze(0).to(device)
    
    # Generate (single forward pass - very fast)
    generator.eval()
    fake_ceus = generator(bmode_tensor)
    
    # Post-process and save
    fake_ceus = (fake_ceus.squeeze(0).cpu() + 1) / 2  # Denormalize: [3, H, W]
    fake_ceus = torch.clamp(fake_ceus, 0, 1)
    fake_ceus = transforms.ToPILImage()(fake_ceus)  # Converts to RGB PIL Image
    fake_ceus.save(output_path)
    
    return fake_ceus

# Single image inference function
@torch.no_grad()
def inference_single_image(model_path: str, bmode_path: str, output_path: str):
    """
    Load model and perform inference on a single B-mode image
    
    Args:
        model_path: Path to the saved model checkpoint
        bmode_path: Path to the input B-mode image
        output_path: Path to save the generated CEUS image
    """
    # Initialize generator
    generator = Generator().to(device)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    
    # Generate CEUS image
    generated_img = generate_ceus_fast(generator, bmode_path, output_path)
    
    print(f"Generated CEUS image saved to: {output_path}")
    return generated_img

# Example usage for 250 pairs with train/validation split
if __name__ == "__main__":
    print(f"Using device: {device}")
    '''
    # Create full dataset from training directory only
    full_dataset = UltrasoundDatasetAugmented(
        "data/train/bmode", 
        "data/train/ceus",
        augment=True,
        augment_factor=8  # 250 * 8 = 2000 training samples
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train with early stopping
    train_pix2pix(
        generator, 
        discriminator, 
        train_loader, 
        val_loader,
        num_epochs=2000,
        lambda_l1=100,
        patience=30,  # Stop if no improvement for 30 epochs
        min_delta=0.001  # Minimum improvement threshold
    )
    '''
    # Example of single image inference after training
    img_files = os.listdir('TestData/bmode/')
    os.makedirs('TestData/Pix2Pix_basic_ceus', exist_ok=True)
    for i in range(0, len(img_files)):
        inference_single_image('models/pix2pix_model.pth', 'TestData/bmode/'+img_files[i], 'TestData/Pix2Pix_basic_ceus/'+img_files[i])