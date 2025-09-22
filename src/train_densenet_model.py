#!/usr/bin/env python3
"""
Enhanced DenseNet Model for Carotid Artery Stenosis Classification
Advanced implementation with medical imaging optimizations
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tqdm import tqdm
import json
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')


class MedicalAttentionModule(nn.Module):
    """Medical-specific attention module for ultrasound imaging"""
    
    def __init__(self, channels, reduction=16):
        super(MedicalAttentionModule, self).__init__()
        self.channels = channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x


class EnhancedDenseBlock(nn.Module):
    """Enhanced Dense Block with medical imaging optimizations"""
    
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0):
        super(EnhancedDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = self._make_layer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)
            
        # Add attention after dense connections
        total_features = num_input_features + num_layers * growth_rate
        self.attention = MedicalAttentionModule(total_features)
        
    def _make_layer(self, num_input_features, growth_rate, bn_size, drop_rate):
        layers = []
        layers.append(nn.BatchNorm2d(num_input_features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_input_features, bn_size * growth_rate, 
                              kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(bn_size * growth_rate))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False))
        
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
            
        output = torch.cat(features, 1)
        output = self.attention(output)
        
        return output


class MedicalDenseNet(nn.Module):
    """
    Enhanced DenseNet for medical ultrasound imaging with:
    - Medical-specific attention mechanisms
    - Enhanced preprocessing for ultrasound
    - Multi-scale feature extraction
    - Clinical-oriented architecture
    """
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=4):
        super(MedicalDenseNet, self).__init__()
        
        # Enhanced initial convolution for medical imaging
        self.features = nn.Sequential(
            # Multi-scale initial feature extraction
            nn.Conv2d(3, num_init_features//2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_init_features//2, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Build dense blocks with attention
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = EnhancedDenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            
            # Transition layer (except for the last block)
            if i != len(block_config) - 1:
                transition = nn.Sequential(
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
                self.transitions.append(transition)
                num_features = num_features // 2
        
        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)
        
        # Global pooling with multiple strategies
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced classifier with medical focus
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(num_features * 2, 512),  # *2 for avg + max pooling
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.features(x)
        
        # Dense blocks with transitions
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        # Final normalization
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).flatten(1)
        max_pool = self.global_max_pool(x).flatten(1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class MedicalUltrasoundDataset(Dataset):
    """Enhanced Dataset with medical ultrasound preprocessing"""
    
    def __init__(self, csv_file, transform=None, split='train', enhance_contrast=True):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[self.data_frame['split'] == split].reset_index(drop=True)
        self.transform = transform
        self.split = split
        self.enhance_contrast = enhance_contrast
        
        # Label mapping
        self.label_to_idx = {
            'normal': 0,
            'mild_stenosis': 1,
            'moderate_stenosis': 2,
            'severe_stenosis': 3
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        # Print dataset info
        print(f"\n{split.capitalize()} dataset: {len(self.data_frame)} samples")
        label_counts = self.data_frame['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  - {label}: {count} ({count/len(self.data_frame)*100:.1f}%)")
        
    def _enhance_ultrasound_image(self, image):
        """Apply medical-specific image enhancements"""
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(img_array.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_array)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Apply Gaussian filtering for speckle noise reduction
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        # Enhance edges (important for vessel boundaries)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Combine original enhanced image with edge information
        enhanced = cv2.addWeighted(enhanced, 0.8, edges_colored, 0.2, 0)
        
        return Image.fromarray(enhanced)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['filepath']
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply medical enhancements
            if self.enhance_contrast:
                image = self._enhance_ultrasound_image(image)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Create a dummy image
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
        label = self.data_frame.iloc[idx]['label']
        label_idx = self.label_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx, img_path
    
    def get_class_weights(self):
        """Calculate enhanced class weights for medical imaging"""
        labels = [self.label_to_idx[label] for label in self.data_frame['label']]
        
        # Standard balanced weights
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        
        # Medical adjustment: penalize severe stenosis misclassification more
        # Severe stenosis is most critical to detect correctly
        medical_multipliers = [1.0, 1.1, 1.3, 1.5]  # normal, mild, moderate, severe
        adjusted_weights = class_weights * medical_multipliers
        
        return torch.FloatTensor(adjusted_weights)


def get_enhanced_transforms():
    """Get medical-specific data transforms"""
    
    # Training transforms with medical-appropriate augmentations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced for medical images
        transforms.RandomRotation(5),  # Small rotation for medical images
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_enhanced_densenet(num_classes=4, pretrained=True):
    """Create enhanced DenseNet model"""
    if pretrained:
        # Load pretrained DenseNet and modify for medical imaging
        model = models.densenet121(pretrained=True)
        
        # Replace classifier with enhanced medical classifier
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Add attention to features
        # Insert attention module before final classification
        original_features = model.features
        
        class EnhancedDenseNetPretrained(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.features = original_model.features
                self.attention = MedicalAttentionModule(1024)  # DenseNet121 has 1024 features
                self.classifier = original_model.classifier
                
            def forward(self, x):
                features = self.features(x)
                features = self.attention(features)
                features = F.relu(features, inplace=True)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                out = self.classifier(features)
                return out
        
        enhanced_model = EnhancedDenseNetPretrained(model)
        return enhanced_model
    else:
        # Use custom medical DenseNet
        return MedicalDenseNet(num_classes=num_classes)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Training epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/len(dataloader):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validation epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, f1, all_preds, all_labels


def train_enhanced_densenet():
    """Main training function for enhanced DenseNet"""
    
    print("Enhanced DenseNet Training for Carotid Artery Stenosis Classification")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    csv_file = 'data/processed/image_labels_with_splits.csv'
    model_save_path = 'models/enhanced_densenet_carotid_classifier.pth'
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Get transforms
    train_transform, val_transform = get_enhanced_transforms()
    
    # Create datasets
    train_dataset = MedicalUltrasoundDataset(csv_file, train_transform, 'train')
    val_dataset = MedicalUltrasoundDataset(csv_file, val_transform, 'val')
    test_dataset = MedicalUltrasoundDataset(csv_file, val_transform, 'test')
    
    # Get class weights
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"\nEnhanced class weights: {class_weights.tolist()}")
    
    # Create weighted sampler
    sample_weights = [class_weights[train_dataset.label_to_idx[label]] 
                     for label in train_dataset.data_frame['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create enhanced model
    model = create_enhanced_densenet(num_classes=4, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Enhanced DenseNet created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training parameters
    epochs = 50
    best_f1 = 0.0
    early_stopping_patience = 10
    no_improvement_count = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print(f"\nStarting Enhanced DenseNet Training!")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model parameters: {total_params:,}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: 0.001")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("\nStarting enhanced training...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validation
        val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Check for improvement
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'class_weights': class_weights,
                'history': history
            }, model_save_path)
            print(f"[NEW BEST] New best validation F1: {val_f1:.4f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, Val F1: {val_f1:.4f}")
        print(f"LR: {current_lr:.6f}")
        print("-" * 40)
        
        # Early stopping
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nEnhanced DenseNet Training Completed in {training_time:.2f} seconds!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Model saved to: {model_save_path}")
    
    # Load best model for testing
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    # Detailed test results
    label_names = ['Normal', 'Mild Stenosis', 'Moderate Stenosis', 'Severe Stenosis']
    
    # Classification report
    test_report = classification_report(test_labels, test_preds, 
                                      target_names=label_names, 
                                      output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\nEnhanced DenseNet Test Results:")
    print("=" * 50)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    print(f"Test F1-Score: {test_f1:.4f}")
    print("\nPer-class Performance:")
    for i, label in enumerate(label_names):
        precision = test_report[label]['precision']
        recall = test_report[label]['recall']
        f1 = test_report[label]['f1-score']
        support = test_report[label]['support']
        print(f"  {label}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}")
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
        'classification_report': test_report,
        'confusion_matrix': cm.tolist(),
        'training_history': history,
        'training_time': training_time,
        'model_parameters': total_params,
        'best_validation_f1': best_f1
    }
    
    with open(results_dir / 'enhanced_densenet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Enhanced DenseNet - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Enhanced DenseNet - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='Val F1', color='green')
    plt.title('Enhanced DenseNet - F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'enhanced_densenet_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Enhanced DenseNet - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(results_dir / 'enhanced_densenet_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {results_dir}")
    print(f"Enhanced DenseNet Model Results:")
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    train_enhanced_densenet()