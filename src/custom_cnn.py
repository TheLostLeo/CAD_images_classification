#!/usr/bin/env python3
"""
Custom CNN Model for Binary Carotid Artery Stenosis Classification
Designed specifically for binary classification: Normal (0) vs Disease (1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time


class BinaryCADDataset(Dataset):
    """Dataset class for loading binary classification data"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load normal images (class 0)
        normal_dir = os.path.join(data_dir, "0")
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)
        
        # Load disease images (class 1)
        disease_dir = os.path.join(data_dir, "1")
        if os.path.exists(disease_dir):
            for img_name in os.listdir(disease_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(disease_dir, img_name))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class MedicalAttention(nn.Module):
    """Attention mechanism for medical images"""
    
    def __init__(self, in_channels):
        super(MedicalAttention, self).__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention_map = self.attention_conv(x)
        attention_weights = self.sigmoid(attention_map)
        return x * attention_weights


class BinaryCustomCNN(nn.Module):
    """
    Custom CNN architecture for binary carotid artery stenosis classification
    Optimized for medical ultrasound images
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(BinaryCustomCNN, self).__init__()
        
        # Feature extraction blocks
        self.conv_block1 = self._make_conv_block(3, 32, kernel_size=7, padding=3)
        self.conv_block2 = self._make_conv_block(32, 64, kernel_size=5, padding=2)
        self.conv_block3 = self._make_conv_block(64, 128, kernel_size=3, padding=1)
        self.conv_block4 = self._make_conv_block(128, 256, kernel_size=3, padding=1)
        self.conv_block5 = self._make_conv_block(256, 512, kernel_size=3, padding=1)
        
        # Attention mechanism
        self.attention = MedicalAttention(512)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2, 256),  # *2 for avg + max pooling
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        """Create a convolutional block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _initialize_weights(self):
        """Initialize network weights"""
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
        # Feature extraction
        x = self.conv_block1(x)  # 224x224 -> 112x112
        x = self.conv_block2(x)  # 112x112 -> 56x56
        x = self.conv_block3(x)  # 56x56 -> 28x28
        x = self.conv_block4(x)  # 28x28 -> 14x14
        x = self.conv_block5(x)  # 14x14 -> 7x7
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        x = self.classifier(x)
        return x


def get_transforms(image_size=224):
    """Get data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir="data/preprocessed", batch_size=32, image_size=224):
    """Create data loaders for train/val/test"""
    
    train_transform, val_transform = get_transforms(image_size)
    
    # Create datasets
    train_dataset = BinaryCADDataset(
        os.path.join(data_dir, "train"), 
        transform=train_transform
    )
    val_dataset = BinaryCADDataset(
        os.path.join(data_dir, "val"), 
        transform=val_transform
    )
    test_dataset = BinaryCADDataset(
        os.path.join(data_dir, "test"), 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Data loaded:")
    print(f"  Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} images ({len(val_loader)} batches)")
    print(f"  Test: {len(test_dataset)} images ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """Train the binary CNN model"""
    
    # Loss function with class weights for imbalanced data
    class_weights = torch.tensor([1.0, 1.5]).to(device)  # Weight disease class more
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\nStarting training on {device}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * train_correct / train_total:.2f}%'
            })
        
        train_accuracy = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * val_correct / val_total:.2f}%'
                })
        
        val_accuracy = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Time: {epoch_time:.2f}s, Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 60)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    class_names = ['Normal', 'Disease']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Binary CAD Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/binary_cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1


def plot_training_history(history):
    """Plot training history"""
    
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Val Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/binary_cnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir="data/preprocessed",
        batch_size=32,
        image_size=224
    )
    
    # Create model
    model = BinaryCustomCNN(num_classes=2, dropout_rate=0.5)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), 'models/custom_cnn.pth')
    print(f"\nModel saved to: models/custom_cnn.pth")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
        model, test_loader, device
    )
    
    # Save results
    results = {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    import json
    with open('results/binary_cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/binary_cnn_results.json")
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()