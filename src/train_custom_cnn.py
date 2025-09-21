#!/usr/bin/env python3
"""
Improved Custom CNN Model for Carotid Artery Stenosis Classification
Enhanced version with better architecture and training strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class ResidualBlock(nn.Module):
    """Improved residual block with skip connections."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature refinement."""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for important region focus."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


class MedicalUltrasoundCNN(nn.Module):
    """
    Advanced CNN architecture with:
    - Residual connections for better gradient flow
    - Dual attention mechanisms (channel + spatial)
    - Progressive feature extraction
    - Medical-specific design choices
    """
    
    def __init__(self, num_classes=4, input_channels=3, dropout_rate=0.3):
        super(MedicalUltrasoundCNN, self).__init__()
        
        # Initial convolution with larger receptive field
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks with progressive channel increase
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(512)
        self.spatial_attention = SpatialAttention()
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced classifier with multiple paths
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),  # Combined avg + max pooling
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        
        # Progressive feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention mechanisms
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Global pooling (combine avg and max)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class AdvancedUltrasoundDataset(torch.utils.data.Dataset):
    """Enhanced dataset with better preprocessing and augmentation."""
    
    def __init__(self, csv_file, root_dir, transform=None, medical_preprocessing=True, 
                 advanced_augment=False):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.medical_preprocessing = medical_preprocessing
        self.advanced_augment = advanced_augment
        
        # Class mappings
        self.label_to_idx = {
            'normal': 0,
            'mild_stenosis': 1,
            'moderate_stenosis': 2,
            'severe_stenosis': 3
        }
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_name = self.labels_df.iloc[idx]['filename']
        img_path = self.root_dir / img_name
        label = self.labels_df.iloc[idx]['stenosis_class']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (300, 300), (0, 0, 0))
        
        # Apply medical preprocessing
        if self.medical_preprocessing:
            image = self._advanced_medical_preprocess(image)
        
        # Apply advanced augmentation
        if self.advanced_augment:
            image = self._advanced_augment(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        return image, label_idx
    
    def _advanced_medical_preprocess(self, image):
        """Enhanced medical preprocessing for ultrasound images."""
        img_np = np.array(image)
        
        # 1. Adaptive histogram equalization
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 2. Advanced denoising
        img_denoised = cv2.bilateralFilter(img_enhanced, 9, 75, 75)
        
        # 3. Sharpening for better edge definition
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_sharpened = cv2.filter2D(img_denoised, -1, kernel)
        
        # 4. Contrast enhancement
        img_contrast = cv2.convertScaleAbs(img_sharpened, alpha=1.2, beta=10)
        
        return Image.fromarray(img_contrast)
    
    def _advanced_augment(self, image):
        """Advanced augmentation techniques."""
        img_np = np.array(image)
        
        # Random gamma correction
        if np.random.random() > 0.7:
            gamma = np.random.uniform(0.8, 1.2)
            img_np = np.power(img_np / 255.0, gamma) * 255.0
            img_np = img_np.astype(np.uint8)
        
        # Random noise addition
        if np.random.random() > 0.8:
            noise = np.random.normal(0, 5, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np)


class ImprovedTrainer:
    """Enhanced trainer with better optimization strategies."""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Advanced optimizer with better hyperparameters
        self.optimizer = optim.AdamW(
            [
                {'params': self.model.initial_conv.parameters(), 'lr': learning_rate * 0.1},
                {'params': self.model.layer1.parameters(), 'lr': learning_rate * 0.1},
                {'params': self.model.layer2.parameters(), 'lr': learning_rate * 0.2},
                {'params': self.model.layer3.parameters(), 'lr': learning_rate * 0.5},
                {'params': self.model.layer4.parameters(), 'lr': learning_rate * 0.8},
                {'params': self.model.channel_attention.parameters(), 'lr': learning_rate},
                {'params': self.model.spatial_attention.parameters(), 'lr': learning_rate},
                {'params': self.model.classifier.parameters(), 'lr': learning_rate}
            ],
            weight_decay=1e-4
        )
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.patience = 30
    
    def train_epoch(self, train_loader, criterion, epoch, total_epochs):
        """Enhanced training epoch with better techniques."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Apply label smoothing for loss calculation
            target_smooth = self._label_smoothing(target, 0.1)
            loss = F.cross_entropy(output, target_smooth)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics (use hard labels for accuracy)
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def _label_smoothing(self, target, smoothing=0.1):
        """Apply label smoothing for better generalization."""
        num_classes = 4
        confidence = 1.0 - smoothing
        smooth_target = torch.zeros(target.size(0), num_classes, device=target.device)
        smooth_target.fill_(smoothing / (num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        return smooth_target
    
    def validate(self, val_loader, criterion):
        """Enhanced validation with better metrics."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return val_loss, val_acc, val_f1, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs=40, class_weights=None):
        """Enhanced training loop with better strategies."""
        print("Starting Improved Custom CNN Training!")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.learning_rate}")
        
        # Enhanced loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        print(f"Class weights: {class_weights}")
        print("\nStarting enhanced training...")
        print("=" * 70)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, epoch, epochs)
            
            # Validation
            val_loss, val_acc, val_f1, val_preds, val_targets = self.validate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            # Learning rate scheduling
            scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"[NEW BEST] New best validation F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
            
            # Print progress
            print(f"Epoch {epoch+1:2d}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 40)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print("\nImproved Custom CNN Training Completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        return self.best_val_f1
    
    def save_model(self, filepath):
        """Save the best model."""
        if self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'best_val_f1': self.best_val_f1,
                'model_name': 'custom_cnn',
                'architecture': 'MedicalUltrasoundCNN',
                'training_history': {
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies,
                    'val_f1_scores': self.val_f1_scores
                }
            }, filepath)
            print(f"Model saved to: {filepath}")


def get_improved_data_loaders(batch_size=32):
    """Create enhanced data loaders with better augmentation."""
    # Enhanced transforms with more sophisticated augmentation
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Slightly larger for better resolution
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read and create splits
    main_df = pd.read_csv('data/processed/image_labels_with_splits.csv')
    train_df = main_df[main_df['split'] == 'train'][['filename', 'label']].copy()
    val_df = main_df[main_df['split'] == 'val'][['filename', 'label']].copy()
    test_df = main_df[main_df['split'] == 'test'][['filename', 'label']].copy()
    
    train_df = train_df.rename(columns={'label': 'stenosis_class'})
    val_df = val_df.rename(columns={'label': 'stenosis_class'})
    test_df = test_df.rename(columns={'label': 'stenosis_class'})
    
    Path('data/processed').mkdir(exist_ok=True)
    train_df.to_csv('data/processed/train_labels.csv', index=False)
    val_df.to_csv('data/processed/val_labels.csv', index=False)
    test_df.to_csv('data/processed/test_labels.csv', index=False)
    
    # Create enhanced datasets
    train_dataset = AdvancedUltrasoundDataset(
        'data/processed/train_labels.csv',
        'data/raw/Common Carotid Artery Ultrasound Images/Common Carotid Artery Ultrasound Images/US images',
        transform=train_transform,
        advanced_augment=True
    )
    
    val_dataset = AdvancedUltrasoundDataset(
        'data/processed/val_labels.csv',
        'data/raw/Common Carotid Artery Ultrasound Images/Common Carotid Artery Ultrasound Images/US images',
        transform=val_transform
    )
    
    # Enhanced class weights
    class_counts = train_df['stenosis_class'].value_counts()
    total_samples = len(train_df)
    class_weights = []
    for cls in ['normal', 'mild_stenosis', 'moderate_stenosis', 'severe_stenosis']:
        weight = total_samples / (len(class_counts) * class_counts[cls])
        # Apply sqrt to reduce extreme weights
        weight = np.sqrt(weight)
        class_weights.append(weight)
    
    # Create balanced sampler
    labels = [train_dataset.label_to_idx[label] for label in train_df['stenosis_class']]
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, class_weights


def main():
    """Main training function for improved model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create enhanced data loaders
    train_loader, val_loader, class_weights = get_improved_data_loaders(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Enhanced class weights: {class_weights}")
    
    # Create enhanced model
    model = MedicalUltrasoundCNN(num_classes=4, dropout_rate=0.3)
    print(f"Improved model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create enhanced trainer
    trainer = ImprovedTrainer(model, device=device, learning_rate=1e-3)
    
    # Train model with enhanced strategies
    best_f1 = trainer.train(train_loader, val_loader, epochs=40, class_weights=class_weights)
    
    # Save improved model
    model_path = 'models/custom_cnn_carotid_classifier.pth'
    Path('models').mkdir(exist_ok=True)
    trainer.save_model(model_path)
    
    print(f"\nImproved Model Results:")
    print(f"Best Validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()