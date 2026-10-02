import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
import os
from tqdm import tqdm
import json
import shutil
from pathlib import Path

# ===========================
# 1. Data Augmentation (Same as before)
# ===========================
def augment_dataset(csv_path, b_mode_dir, ceus_dir, output_dir, 
                    train_split, random_seed):
    """Augment training dataset"""
    print("="*60)
    print("Starting Data Augmentation Pipeline")
    print("="*60)
    
    np.random.seed(random_seed)
    df = pd.read_csv(csv_path)
    print(f"\nTotal samples: {len(df)}")
    
    train_df, val_df = train_test_split(
        df, test_size=1-train_split, 
        random_state=random_seed, 
        stratify=df['Malignant']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    output_path = Path(output_dir)
    train_b_mode_dir = output_path / 'train' / 'B_mode'
    train_ceus_dir = output_path / 'train' / 'CEUS'
    val_b_mode_dir = output_path / 'val' / 'B_mode'
    val_ceus_dir = output_path / 'val' / 'CEUS'
    
    for dir_path in [train_b_mode_dir, train_ceus_dir, val_b_mode_dir, val_ceus_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Validation set - no augmentation
    print("\nProcessing Validation Set")
    val_records = []
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        name = row['English_Name']
        b_mode_src = find_image(b_mode_dir, name)
        ceus_src = find_image(ceus_dir, name)
        
        if b_mode_src is None or ceus_src is None:
            continue
        
        ext = Path(b_mode_src).suffix
        shutil.copy2(b_mode_src, val_b_mode_dir / f"{name}{ext}")
        shutil.copy2(ceus_src, val_ceus_dir / f"{name}{ext}")
        val_records.append(row.to_dict())
    
    val_df_final = pd.DataFrame(val_records)
    val_df_final.to_csv(output_path / 'val_clinical_data.csv', index=False)
    
    # Training set - with augmentation
    print("\nProcessing Training Set with Augmentation")
    train_records = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        name = row['English_Name']
        b_mode_src = find_image(b_mode_dir, name)
        ceus_src = find_image(ceus_dir, name)
        
        if b_mode_src is None or ceus_src is None:
            continue
        
        b_mode_img = Image.open(b_mode_src).convert('RGB')
        ceus_img = Image.open(ceus_src).convert('RGB')
        ext = Path(b_mode_src).suffix
        
        # Original
        new_name = f"{name}_original"
        b_mode_img.save(train_b_mode_dir / f"{new_name}{ext}")
        ceus_img.save(train_ceus_dir / f"{new_name}{ext}")
        record = row.to_dict()
        record['English_Name'] = new_name
        train_records.append(record)
        
        # Rotations
        for angle in [-30, -20, -10, 10, 20, 30]:
            b_mode_rotated = b_mode_img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            ceus_rotated = ceus_img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            new_name = f"{name}_rot{angle:+03d}"
            b_mode_rotated.save(train_b_mode_dir / f"{new_name}{ext}")
            ceus_rotated.save(train_ceus_dir / f"{new_name}{ext}")
            record = row.to_dict()
            record['English_Name'] = new_name
            train_records.append(record)
        
        # Flips
        selected_angles = np.random.choice([-30, -20, -10, 10, 20, 30], size=3, replace=False)
        for angle in selected_angles:
            rotated_name = f"{name}_rot{angle:+03d}"
            b_mode_src_rot = find_image(train_b_mode_dir, rotated_name)
            ceus_src_rot = find_image(train_ceus_dir, rotated_name)
            
            if b_mode_src_rot and ceus_src_rot:
                b_mode_img_rot = Image.open(b_mode_src_rot).convert('RGB')
                ceus_img_rot = Image.open(ceus_src_rot).convert('RGB')
                
                b_mode_flipped = b_mode_img_rot.transpose(Image.FLIP_LEFT_RIGHT)
                ceus_flipped = ceus_img_rot.transpose(Image.FLIP_LEFT_RIGHT)
                
                new_name = f"{rotated_name}_flip"
                b_mode_flipped.save(train_b_mode_dir / f"{new_name}{ext}")
                ceus_flipped.save(train_ceus_dir / f"{new_name}{ext}")
                record = row.to_dict()
                record['English_Name'] = new_name
                train_records.append(record)
    
    train_df_final = pd.DataFrame(train_records)
    train_df_final.to_csv(output_path / 'train_clinical_data.csv', index=False)
    
    print(f"\nAugmentation Complete!")
    print(f"Training: {len(train_records)} images")
    print(f"Validation: {len(val_records)} images")
    
    return train_df_final, val_df_final

def find_image(image_dir, name):
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        image_path = Path(image_dir) / f"{name}{ext}"
        if image_path.exists():
            return str(image_path)
    return None

# ===========================
# 2. Simple Baseline: Fine-tuned Pre-trained Model
# ===========================
class SimpleUltrasoundDataset(Dataset):
    def __init__(self, data_df, b_mode_dir, ceus_dir, transform=None):
        self.data_df = data_df.reset_index(drop=True)
        self.b_mode_dir = b_mode_dir
        self.ceus_dir = ceus_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        name = row['English_Name']
        
        b_mode_path = find_image(self.b_mode_dir, name)
        ceus_path = find_image(self.ceus_dir, name)
        
        if b_mode_path is None or ceus_path is None:
            raise FileNotFoundError(f"Images not found for {name}")
        
        b_mode_img = Image.open(b_mode_path).convert('RGB')
        ceus_img = Image.open(ceus_path).convert('RGB')
        
        if self.transform:
            b_mode_img = self.transform(b_mode_img)
            ceus_img = self.transform(ceus_img)
        
        # Concatenate along channel dimension
        combined_img = torch.cat([b_mode_img, ceus_img], dim=0)
        
        labels = {
            'Internal_Echo': int(row['Internal_Echo']),
            'Morphology': float(row['Morphology']),
            'Boundary': float(row['Boundary']),
            'Solid': float(row['Solid']),
            'Separation': float(row['Separation']),
            'Nipple': float(row['Nipple']),
            'Blood_Flow': float(row['Blood_Flow']),
            'Malignant': float(row['Malignant'])
        }
        
        return combined_img, labels, name

class SimplePretrainedModel(nn.Module):
    """Unified model supporting multiple pre-trained backbones"""
    
    def __init__(self, dropout=0.5, backbone='vit_l_32'):
        """
        Args:
            dropout: Dropout rate
            backbone: One of: 'efficientnet-b0/b1/b2/b3/b4/b5/b6/b7',
                     'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                     'vgg16', 'vgg19',
                     'resnet18/34/50/101/152',
                     'densenet121/169/201',
                     'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'
        """
        super(SimplePretrainedModel, self).__init__()
        
        self.backbone_name = backbone
        self.features = None
        
        # EfficientNet
        if 'efficientnet' in backbone:
            try:
                from efficientnet_pytorch import EfficientNet
            except ImportError:
                raise ImportError("Install: pip install efficientnet-pytorch")
            
            base_model = EfficientNet.from_pretrained(backbone)
            old_conv = base_model._conv_stem
            self.conv1 = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                  stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                self.conv1.weight[:, :3] = old_conv.weight
                self.conv1.weight[:, 3:] = old_conv.weight
            base_model._conv_stem = self.conv1
            self.features = base_model
            
            feature_dims = {
                'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                'efficientnet-b6': 2304, 'efficientnet-b7': 2560
            }
            feature_dim = feature_dims.get(backbone, 1280)
            self.forward_type = 'efficientnet'
            
        # MobileNet
        elif 'mobilenet' in backbone:
            if backbone == 'mobilenet_v2':
                base_model = models.mobilenet_v2(pretrained=True)
                feature_dim = 1280
            elif backbone == 'mobilenet_v3_small':
                base_model = models.mobilenet_v3_small(pretrained=True)
                feature_dim = 576
            elif backbone == 'mobilenet_v3_large':
                base_model = models.mobilenet_v3_large(pretrained=True)
                feature_dim = 960
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            # Modify first conv
            old_conv = base_model.features[0][0]
            new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight
            base_model.features[0][0] = new_conv
            
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.forward_type = 'mobilenet'
            
        # VGG
        elif 'vgg' in backbone:
            if backbone == 'vgg16':
                base_model = models.vgg16(pretrained=True)
            elif backbone == 'vgg19':
                base_model = models.vgg19(pretrained=True)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            feature_dim = 4096  # VGG classifier output
            
            # Modify first conv
            old_conv = base_model.features[0]
            new_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight
                new_conv.bias = old_conv.bias
            base_model.features[0] = new_conv
            
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.vgg_classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
            self.forward_type = 'vgg'
            
        # ResNet
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                base_model = models.resnet18(pretrained=True)
                feature_dim = 512
            elif backbone == 'resnet34':
                base_model = models.resnet34(pretrained=True)
                feature_dim = 512
            elif backbone == 'resnet50':
                base_model = models.resnet50(pretrained=True)
                feature_dim = 2048
            elif backbone == 'resnet101':
                base_model = models.resnet101(pretrained=True)
                feature_dim = 2048
            elif backbone == 'resnet152':
                base_model = models.resnet152(pretrained=True)
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.conv1.weight[:, :3] = base_model.conv1.weight
                self.conv1.weight[:, 3:] = base_model.conv1.weight
            
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool
            self.forward_type = 'resnet'
            
        # DenseNet
        elif 'densenet' in backbone:
            if backbone == 'densenet121':
                base_model = models.densenet121(pretrained=True)
                feature_dim = 1024
            elif backbone == 'densenet169':
                base_model = models.densenet169(pretrained=True)
                feature_dim = 1664
            elif backbone == 'densenet201':
                base_model = models.densenet201(pretrained=True)
                feature_dim = 1920
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            # Modify first conv
            old_conv = base_model.features.conv0
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight
            base_model.features.conv0 = new_conv
            
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.forward_type = 'densenet'
            
        # Vision Transformer (ViT)
        elif 'vit' in backbone:
            if backbone == 'vit_b_16':
                base_model = models.vit_b_16(pretrained=True)
                feature_dim = 768
            elif backbone == 'vit_b_32':
                base_model = models.vit_b_32(pretrained=True)
                feature_dim = 768
            elif backbone == 'vit_l_16':
                base_model = models.vit_l_16(pretrained=True)
                feature_dim = 1024
            elif backbone == 'vit_l_32':
                base_model = models.vit_l_32(pretrained=True)
                feature_dim = 1024
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            # Modify patch embedding to accept 6 channels
            old_conv = base_model.conv_proj
            new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride, padding=old_conv.padding)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight
                new_conv.bias = old_conv.bias
            base_model.conv_proj = new_conv
            
            self.vit_model = base_model
            self.forward_type = 'vit'
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Build adaptive classifier
        if feature_dim >= 2048:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif feature_dim >= 1024:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Task-specific heads
        self.internal_echo_head = nn.Linear(256, 4)
        self.morphology_head = nn.Linear(256, 1)
        self.boundary_head = nn.Linear(256, 1)
        self.solid_head = nn.Linear(256, 1)
        self.separation_head = nn.Linear(256, 1)
        self.nipple_head = nn.Linear(256, 1)
        self.blood_flow_head = nn.Linear(256, 1)
        self.malignant_head = nn.Linear(256, 1)
    
    def forward(self, x):
        if self.forward_type == 'efficientnet':
            x = self.features.extract_features(x)
            x = self.features._avg_pooling(x)
            x = x.flatten(start_dim=1)
            
        elif self.forward_type == 'mobilenet':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            
        elif self.forward_type == 'vgg':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.vgg_classifier(x)
            
        elif self.forward_type == 'resnet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            
        elif self.forward_type == 'densenet':
            x = self.features(x)
            x = F.relu(x, inplace=True)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            
        elif self.forward_type == 'vit':
            x = self.vit_model._process_input(x)
            n = x.shape[0]
            batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.vit_model.encoder(x)
            x = x[:, 0]  # Take class token
        
        # Shared classifier
        x = self.classifier(x)
        
        # Task-specific predictions
        outputs = {
            'Internal_Echo': self.internal_echo_head(x),
            'Morphology': self.morphology_head(x),
            'Boundary': self.boundary_head(x),
            'Solid': self.solid_head(x),
            'Separation': self.separation_head(x),
            'Nipple': self.nipple_head(x),
            'Blood_Flow': self.blood_flow_head(x),
            'Malignant': self.malignant_head(x)
        }
        
        return outputs

# ===========================
# 3. Training Functions
# ===========================
class MultiLabelLoss(nn.Module):
    """Multi-label loss combining cross-entropy for Internal_Echo and BCE for binary tasks"""
    def __init__(self, internal_echo_weight=1.0, binary_weight=1.0, use_focal_loss=False):
        super(MultiLabelLoss, self).__init__()
        self.internal_echo_loss = nn.CrossEntropyLoss()
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.internal_echo_weight = internal_echo_weight
        self.binary_weight = binary_weight
        self.use_focal_loss = use_focal_loss
        self.echo_label_to_idx = {0: 0, 1: 1, 3: 2, 4: 3}
        
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
        
    def forward(self, predictions, targets):
        # Internal Echo loss (multiclass classification)
        echo_targets = torch.tensor([self.echo_label_to_idx[int(t)] 
                                     for t in targets['Internal_Echo'].cpu().numpy()],
                                    device=predictions['Internal_Echo'].device)
        echo_loss = self.internal_echo_loss(predictions['Internal_Echo'], echo_targets)
        
        # Binary tasks loss
        binary_tasks = ['Morphology', 'Boundary', 'Solid', 'Separation', 
                       'Nipple', 'Blood_Flow', 'Malignant']
        
        if self.use_focal_loss:
            # Use focal loss for binary tasks (better for imbalanced data)
            binary_loss = 0
            for task in binary_tasks:
                binary_loss += self.focal_loss(predictions[task].squeeze(), targets[task])
            binary_loss = binary_loss / len(binary_tasks)
        else:
            # Standard BCE loss
            binary_loss = 0
            for task in binary_tasks:
                binary_loss += self.binary_loss(predictions[task].squeeze(), targets[task])
            binary_loss = binary_loss / len(binary_tasks)
        
        # Combined loss
        total_loss = self.internal_echo_weight * echo_loss + self.binary_weight * binary_loss
        
        return total_loss, echo_loss, binary_loss

def compute_hamming_loss(predictions, targets):
    """
    Compute Hamming loss for multi-label classification
    Hamming loss = fraction of labels that are incorrectly predicted
    """
    # Internal Echo (convert to one-hot for consistency)
    echo_label_to_idx = {0: 0, 1: 1, 3: 2, 4: 3}
    idx_to_echo_label = {0: 0, 1: 1, 2: 3, 3: 4}
    
    echo_preds = torch.argmax(predictions['Internal_Echo'], dim=1)
    echo_targets = torch.tensor([echo_label_to_idx[int(t)] 
                                 for t in targets['Internal_Echo'].cpu().numpy()],
                                device=echo_preds.device)
    echo_correct = (echo_preds == echo_targets).float()
    
    # Binary tasks
    binary_tasks = ['Morphology', 'Boundary', 'Solid', 'Separation', 
                   'Nipple', 'Blood_Flow', 'Malignant']
    
    binary_correct = []
    for task in binary_tasks:
        preds = (torch.sigmoid(predictions[task].squeeze()) > 0.5).float()
        correct = (preds == targets[task]).float()
        binary_correct.append(correct)
    
    # Stack all binary predictions
    binary_correct = torch.stack(binary_correct, dim=1)  # [batch, num_binary_tasks]
    
    # Combine Internal_Echo and binary tasks
    all_correct = torch.cat([echo_correct.unsqueeze(1), binary_correct], dim=1)  # [batch, 1+7=8]
    
    # Hamming loss = 1 - average correctness
    hamming_loss_value = 1.0 - all_correct.mean()
    
    return hamming_loss_value.item()

def compute_subset_accuracy(predictions, targets):
    """
    Compute subset accuracy (exact match ratio)
    Returns 1.0 if ALL labels are correct, 0.0 otherwise
    """
    echo_label_to_idx = {0: 0, 1: 1, 3: 2, 4: 3}
    
    echo_preds = torch.argmax(predictions['Internal_Echo'], dim=1)
    echo_targets = torch.tensor([echo_label_to_idx[int(t)] 
                                 for t in targets['Internal_Echo'].cpu().numpy()],
                                device=echo_preds.device)
    echo_correct = (echo_preds == echo_targets)
    
    binary_tasks = ['Morphology', 'Boundary', 'Solid', 'Separation', 
                   'Nipple', 'Blood_Flow', 'Malignant']
    
    all_correct = echo_correct
    for task in binary_tasks:
        preds = (torch.sigmoid(predictions[task].squeeze()) > 0.5).float()
        correct = (preds == targets[task])
        all_correct = all_correct & correct.bool()
    
    # Subset accuracy = fraction of samples where ALL labels are correct
    return all_correct.float().mean().item()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_echo_loss = 0.0
    running_binary_loss = 0.0
    running_hamming_loss = 0.0
    running_subset_acc = 0.0
    
    # Track predictions for accuracy
    all_echo_preds = []
    all_echo_targets = []
    all_malignant_preds = []
    all_malignant_targets = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss, echo_loss, binary_loss = criterion(outputs, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\n⚠️  Warning: NaN loss detected at batch {batch_idx}")
            print(f"Echo loss: {echo_loss.item()}, Binary loss: {binary_loss.item()}")
            continue
        
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        running_echo_loss += echo_loss.item()
        running_binary_loss += binary_loss.item()
        
        # Compute Hamming loss and subset accuracy
        with torch.no_grad():
            hamming = compute_hamming_loss(outputs, labels)
            subset_acc = compute_subset_accuracy(outputs, labels)
            running_hamming_loss += hamming
            running_subset_acc += subset_acc
        
        # Track predictions for training accuracy
        echo_preds = torch.argmax(outputs['Internal_Echo'], dim=1)
        echo_label_to_idx = {0: 0, 1: 1, 3: 2, 4: 3}
        echo_targets = torch.tensor([echo_label_to_idx[int(t)] 
                                     for t in labels['Internal_Echo'].cpu().numpy()],
                                    device=device)
        all_echo_preds.extend(echo_preds.cpu().numpy())
        all_echo_targets.extend(echo_targets.cpu().numpy())
        
        malignant_preds = (torch.sigmoid(outputs['Malignant'].squeeze()) > 0.5).float()
        all_malignant_preds.extend(malignant_preds.cpu().numpy())
        all_malignant_targets.extend(labels['Malignant'].cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'hamming': f'{hamming:.4f}'
        })
    
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    epoch_echo_loss = running_echo_loss / num_batches
    epoch_binary_loss = running_binary_loss / num_batches
    epoch_hamming_loss = running_hamming_loss / num_batches
    epoch_subset_acc = running_subset_acc / num_batches
    
    # Calculate training accuracy
    train_echo_acc = accuracy_score(all_echo_targets, all_echo_preds)
    train_malignant_acc = accuracy_score(all_malignant_targets, all_malignant_preds)
    
    return (epoch_loss, epoch_echo_loss, epoch_binary_loss, 
            train_echo_acc, train_malignant_acc, epoch_hamming_loss, epoch_subset_acc)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_hamming_loss = 0.0
    running_subset_acc = 0.0
    
    idx_to_echo_label = {0: 0, 1: 1, 2: 3, 3: 4}
    
    all_predictions = {k: [] for k in ['Internal_Echo', 'Morphology', 'Boundary', 
                                        'Solid', 'Separation', 'Nipple', 'Blood_Flow', 'Malignant']}
    all_targets = {k: [] for k in ['Internal_Echo', 'Morphology', 'Boundary', 
                                    'Solid', 'Separation', 'Nipple', 'Blood_Flow', 'Malignant']}
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(images)
            loss, _, _ = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Compute Hamming loss and subset accuracy
            hamming = compute_hamming_loss(outputs, labels)
            subset_acc = compute_subset_accuracy(outputs, labels)
            running_hamming_loss += hamming
            running_subset_acc += subset_acc
            
            # Internal Echo
            echo_preds = torch.argmax(outputs['Internal_Echo'], dim=1)
            all_predictions['Internal_Echo'].extend([idx_to_echo_label[p.item()] for p in echo_preds])
            all_targets['Internal_Echo'].extend(labels['Internal_Echo'].cpu().numpy())
            
            # Binary tasks
            binary_tasks = ['Morphology', 'Boundary', 'Solid', 'Separation', 
                          'Nipple', 'Blood_Flow', 'Malignant']
            for task in binary_tasks:
                preds = (torch.sigmoid(outputs[task].squeeze()) > 0.5).float()
                all_predictions[task].extend(preds.cpu().numpy())
                all_targets[task].extend(labels[task].cpu().numpy())
    
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    epoch_hamming_loss = running_hamming_loss / num_batches
    epoch_subset_acc = running_subset_acc / num_batches
    
    # Calculate metrics
    metrics = {}
    metrics['hamming_loss'] = epoch_hamming_loss
    metrics['subset_accuracy'] = epoch_subset_acc
    metrics['Internal_Echo_accuracy'] = accuracy_score(all_targets['Internal_Echo'], 
                                                        all_predictions['Internal_Echo'])
    
    binary_tasks = ['Morphology', 'Boundary', 'Solid', 'Separation', 
                   'Nipple', 'Blood_Flow', 'Malignant']
    for task in binary_tasks:
        metrics[f'{task}_accuracy'] = accuracy_score(all_targets[task], all_predictions[task])
    
    # Overall binary hamming loss
    all_binary_targets = np.column_stack([all_targets[task] for task in binary_tasks])
    all_binary_preds = np.column_stack([all_predictions[task] for task in binary_tasks])
    from sklearn.metrics import hamming_loss as sklearn_hamming_loss
    metrics['sklearn_hamming_loss'] = sklearn_hamming_loss(all_binary_targets, all_binary_preds)
    
    return epoch_loss, metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_hamming_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 20
    
    history = {
        'train_loss': [], 'train_echo_acc': [], 'train_malignant_acc': [],
        'train_hamming': [], 'train_subset_acc': [],
        'val_loss': [], 'val_hamming': [], 'val_subset_acc': [], 'val_metrics': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 70)
        
        # Training
        (train_loss, train_echo_loss, train_binary_loss, train_echo_acc, 
         train_malignant_acc, train_hamming, train_subset_acc) = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_echo_acc'].append(train_echo_acc)
        history['train_malignant_acc'].append(train_malignant_acc)
        history['train_hamming'].append(train_hamming)
        history['train_subset_acc'].append(train_subset_acc)
        history['val_loss'].append(val_loss)
        history['val_hamming'].append(val_metrics['hamming_loss'])
        history['val_subset_acc'].append(val_metrics['subset_accuracy'])
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f'\n📊 Training Metrics:')
        print(f'   Loss: {train_loss:.4f} (Echo: {train_echo_loss:.4f}, Binary: {train_binary_loss:.4f})')
        print(f'   Hamming Loss: {train_hamming:.4f} (lower is better)')
        print(f'   Subset Accuracy: {train_subset_acc:.4f} (exact match for all labels)')
        print(f'   Echo Accuracy: {train_echo_acc:.4f}')
        print(f'   Malignant Accuracy: {train_malignant_acc:.4f}')
        
        print(f'\n📈 Validation Metrics:')
        print(f'   Loss: {val_loss:.4f}')
        print(f'   Hamming Loss: {val_metrics["hamming_loss"]:.4f}')
        print(f'   Subset Accuracy: {val_metrics["subset_accuracy"]:.4f}')
        print(f'   Individual Task Accuracies:')
        print(f'      Internal_Echo: {val_metrics["Internal_Echo_accuracy"]:.4f}')
        for task in ['Morphology', 'Boundary', 'Solid', 'Separation', 
                     'Nipple', 'Blood_Flow', 'Malignant']:
            print(f'      {task}: {val_metrics[f"{task}_accuracy"]:.4f}')
        
        # Check for issues
        if train_loss > 2.0 and epoch > 5:
            print("\n⚠️  WARNING: Training loss is very high! Check data loading and labels.")
        
        if train_hamming > 0.5 and epoch > 10:
            print("\n⚠️  WARNING: High Hamming loss indicates poor multi-label performance.")
        
        if val_metrics['Malignant_accuracy'] < 0.6 and epoch > 10:
            print("\n⚠️  WARNING: Low validation accuracy. Model may not be learning.")
        
        # Check for divergence (loss and accuracy both increasing)
        if epoch > 5:
            if (train_loss > history['train_loss'][epoch-5] and 
                train_malignant_acc > history['train_malignant_acc'][epoch-5]):
                print("\n⚠️  WARNING: Loss and accuracy both increasing - possible issue!")
        
        # Save best model based on Hamming loss
        if val_metrics['hamming_loss'] < best_hamming_loss:
            best_hamming_loss = val_metrics['hamming_loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_hamming_loss': val_metrics['hamming_loss'],
                'val_metrics': val_metrics,
                'train_loss': train_loss,
                'train_hamming': train_hamming
            }, os.path.join(save_dir, 'best_hamming_model.pth'))
            print(f'\n✓ Best Hamming model saved! (Hamming Loss: {val_metrics["hamming_loss"]:.4f})')
        
        # Also save best validation loss model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'✓ Best loss model saved! (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
        
        # Also save best accuracy model
        if val_metrics['Malignant_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['Malignant_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics
            }, os.path.join(save_dir, 'best_acc_model.pth'))
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping triggered! No improvement for {early_stop_patience} epochs.")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best Hamming loss: {best_hamming_loss:.4f}")
    print(f"Best Malignant accuracy: {best_val_acc:.4f}")
    print(f"{'='*70}")
    
    return history

# ===========================
# 4. Main Training
# ===========================
def main_train(augmented_data_dir, use_focal_loss=False, backbone='efficientnet-b7'):
    """
    Main training function
    
    Args:
        augmented_data_dir: Directory with augmented data
        use_focal_loss: Whether to use focal loss for binary tasks
        backbone: Model backbone - options:
                  'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                  'resnet18', 'resnet34', 'resnet50'
    """
    CONFIG = {
        'img_size': 224,
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.0001,  # Lower LR for fine-tuning
        'use_focal_loss': use_focal_loss,
        'backbone': backbone,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"Backbone: {CONFIG['backbone']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Using focal loss: {CONFIG['use_focal_loss']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print("="*70)
    
    augmented_path = Path(augmented_data_dir)
    train_df = pd.read_csv(augmented_path / 'train_clinical_data.csv')
    val_df = pd.read_csv(augmented_path / 'val_clinical_data.csv')
    
    print(f"\nDataset Statistics:")
    print(f"Training: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    
    # Print label distribution
    print(f"\nLabel Distribution (Training):")
    print(f"Internal_Echo: {dict(train_df['Internal_Echo'].value_counts())}")
    print(f"Malignant: {dict(train_df['Malignant'].value_counts())}")
    
    # Simple transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SimpleUltrasoundDataset(
        train_df, 
        augmented_path / 'train' / 'B_mode',
        augmented_path / 'train' / 'CEUS',
        transform
    )
    
    val_dataset = SimpleUltrasoundDataset(
        val_df,
        augmented_path / 'val' / 'B_mode',
        augmented_path / 'val' / 'CEUS',
        transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    model = SimplePretrainedModel(dropout=0.5, backbone=CONFIG['backbone']).to(CONFIG['device'])
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = MultiLabelLoss(
        internal_echo_weight=1.0, 
        binary_weight=1.0,
        use_focal_loss=CONFIG['use_focal_loss']
    )
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    save_dir = 'checkpoints_'+CONFIG['backbone']+'_'+augmented_data_dir[15:]
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         scheduler, CONFIG['num_epochs'], CONFIG['device'], save_dir)
    
    print("\n✓ Training completed!")
    return model, history

# ===========================
# 5. Inference Functions
# ===========================
def predict_single_image(model, b_mode_path, ceus_path, device, img_size=224):
    """Predict labels for a single image pair"""
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    b_mode_img = Image.open(b_mode_path).convert('RGB')
    ceus_img = Image.open(ceus_path).convert('RGB')
    
    b_mode_tensor = transform(b_mode_img)
    ceus_tensor = transform(ceus_img)
    
    combined = torch.cat([b_mode_tensor, ceus_tensor], dim=0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(combined)
    
    # Process predictions
    idx_to_echo_label = {0: 0, 1: 1, 2: 3, 3: 4}
    echo_pred = idx_to_echo_label[torch.argmax(outputs['Internal_Echo'], dim=1).item()]
    
    predictions = {'Internal_Echo': echo_pred}
    
    binary_tasks = ['Morphology', 'Boundary', 'Solid', 'Separation', 
                   'Nipple', 'Blood_Flow', 'Malignant']
    for task in binary_tasks:
        prob = torch.sigmoid(outputs[task].squeeze()).item()
        predictions[task] = int(prob > 0.5)
        predictions[f'{task}_prob'] = prob
    
    return predictions

def batch_inference(model_path, csv_path, b_mode_dir, ceus_dir, 
                   output_path='predictions.csv', device='cuda', img_size=224):
    """Run inference on entire dataset"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SimplePretrainedModel(dropout=0.5).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Best validation loss: {checkpoint.get('val_loss', 'N/A')}")
    if 'val_metrics' in checkpoint:
        print("Validation metrics:")
        for k, v in checkpoint['val_metrics'].items():
            print(f"  {k}: {v:.4f}")
    
    # Load data
    df = pd.read_csv(csv_path)
    results = []
    
    print(f"\nRunning inference on {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        name = row['English_Name']
        
        b_mode_path = find_image(b_mode_dir, name)
        ceus_path = find_image(ceus_dir, name)
        
        if b_mode_path is None or ceus_path is None:
            print(f"Warning: Images not found for {name}")
            continue
        
        try:
            predictions = predict_single_image(model, b_mode_path, ceus_path, 
                                             device, img_size)
            predictions['English_Name'] = name
            
            # Add ground truth if available
            if 'Internal_Echo' in row:
                predictions['True_Internal_Echo'] = row['Internal_Echo']
            if 'Malignant' in row:
                predictions['True_Malignant'] = row['Malignant']
            if 'Morphology' in row:
                predictions['True_Morphology'] = row['Morphology']
            if 'Boundary' in row:
                predictions['True_Boundary'] = row['Boundary']
            if 'Solid' in row:
                predictions['True_Solid'] = row['Solid']
            if 'Separation' in row:
                predictions['True_Separation'] = row['Separation']
            if 'Nipple' in row:
                predictions['True_Nipple'] = row['Nipple']
            if 'Blood_Flow' in row:
                predictions['True_Blood_Flow'] = row['Blood_Flow']
            
            results.append(predictions)
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Calculate accuracy metrics if ground truth available
    if 'True_Malignant' in results_df.columns:
        print("\n" + "="*60)
        print("Performance Metrics")
        print("="*60)
        
        malignant_acc = accuracy_score(results_df['True_Malignant'], 
                                       results_df['Malignant'])
        print(f"Malignant classification accuracy: {malignant_acc:.4f}")
        
        if 'True_Internal_Echo' in results_df.columns:
            echo_acc = accuracy_score(results_df['True_Internal_Echo'], 
                                     results_df['Internal_Echo'])
            print(f"Internal Echo classification accuracy: {echo_acc:.4f}")
        
        # Binary task accuracies
        binary_tasks = ['Morphology', 'Boundary', 'Solid', 'Separation', 
                       'Nipple', 'Blood_Flow']
        print("\nBinary Task Accuracies:")
        for task in binary_tasks:
            if f'True_{task}' in results_df.columns:
                acc = accuracy_score(results_df[f'True_{task}'], results_df[task])
                print(f"  {task}: {acc:.4f}")
        
        # Overall hamming loss
        binary_true_cols = [f'True_{task}' for task in binary_tasks + ['Malignant'] 
                           if f'True_{task}' in results_df.columns]
        binary_pred_cols = [task for task in binary_tasks + ['Malignant'] 
                           if f'True_{task}' in results_df.columns]
        
        if binary_true_cols:
            true_binary = results_df[binary_true_cols].values
            pred_binary = results_df[binary_pred_cols].values
            hloss = hamming_loss(true_binary, pred_binary)
            print(f"\nOverall Hamming Loss: {hloss:.4f}")
    
    return results_df

# ===========================
# 6. Complete Pipeline
# ===========================
def run_complete_pipeline(csv_path, b_mode_dir, ceus_dir, output_dir,
                         train_split=0.7, random_seed=2025, run_inference=True,
                         backbone='efficientnet-b7', use_focal_loss=False):
    """
    Complete pipeline with centralized configuration
    
    All paths and backbone are configured in main function
    """
    print("="*60)
    print("SIMPLE BASELINE PIPELINE")
    print("="*60)
    
    # Step 1: Augmentation
    print("\n### STEP 1: DATA AUGMENTATION ###")
    train_df, val_df = augment_dataset(csv_path, b_mode_dir, ceus_dir, 
                                      output_dir, train_split, random_seed)
    
    # Step 2: Training
    print("\n### STEP 2: MODEL TRAINING ###")
    model, history = main_train(output_dir, use_focal_loss=use_focal_loss, backbone=backbone)
    
    # Step 3: Inference on validation set
    if run_inference:
        print("\n### STEP 3: INFERENCE ON VALIDATION SET ###")
        augmented_path = Path(output_dir)
        val_csv = augmented_path / 'val_clinical_data.csv'
        val_b_mode_dir = augmented_path / 'val' / 'B_mode'
        val_ceus_dir = augmented_path / 'val' / 'CEUS'
        
        results = batch_inference(
            model_path='checkpoints_'+backbone+'_'+output_dir[15:]+'/best_hamming_model.pth',
            csv_path=str(val_csv),
            b_mode_dir=str(val_b_mode_dir),
            ceus_dir=str(val_ceus_dir),
            output_path='validation_predictions_hamming_'+output_dir[15:]+'.csv',
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            img_size=224
        )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED!")
    print("="*60)
    print("\nOutputs:")
    print("  - Best model: checkpoints_"+backbone+"_"+output_dir[15:]+"/best_model.pth")
    print("  - Best Hamming model: checkpoints_"+backbone+"_"+output_dir[15:]+"/best_hamming_model.pth")
    print("  - Training history: checkpoints_"+backbone+"_"+output_dir[15:]+"/history.json")
    if run_inference:
        print("  - Predictions: validation_hamming_"+output_dir[15:]+".csv")

# ===========================
# 7. Example Usage
# ===========================
if __name__ == '__main__':
    # Option 1: Run complete pipeline (augmentation + training + inference)
    '''
    run_complete_pipeline(
        csv_path='clinical_data.csv',
        b_mode_dir='data/bmode',
        ceus_dir='data/Pix2Pix_enhanced_ceus',
        output_dir='augmented_data_Pix2Pix_enhanced_ceus',
        train_split=0.7,
        random_seed=2025,
        run_inference=True, 
        backbone='efficientnet-b7', 
        use_focal_loss=False
    )
    '''
    # Option 2: Run only training (if augmentation already done)
    # model, history = main_train(augmented_data_dir='augmented_data')
    
    # Option 3: Run only inference (if model already trained)
    model_name = 'vit_l_32'
    CEUS_path = 'wo_ssim'
    results = batch_inference(
        model_path=model_name+'/checkpoints_'+model_name+'_'+CEUS_path+'/best_hamming_model.pth',
        csv_path='TestData/test_clinical_data.csv',
        b_mode_dir='TestData/bmode',
        ceus_dir='TestData/'+CEUS_path,
        output_path=model_name+'/test_predictions_hamming_'+CEUS_path+'.csv'
    )
    
    # Option 4: Predict single image
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = SimplePretrainedModel().to(device)
    # checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # 
    # prediction = predict_single_image(
    #     model,
    #     'B_mode_images/sample.png',
    #     'CEUS_images/sample.png',
    #     device
    # )
    # print(prediction)