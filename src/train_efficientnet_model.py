#!/usr/bin/env python3
"""
Enhanced Carotid Artery Stenosis Classification Model

Improved version with better architecture, training strategies, and data handling
for higher accuracy on medical ultrasound images.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


class AdvancedCarotidDataset(Dataset):
    """Enhanced Dataset with better preprocessing for medical images."""
    
    def __init__(self, csv_file, transform=None, split='train'):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[self.data_frame['split'] == split].reset_index(drop=True)
        self.transform = transform
        self.split = split
        
        # Label mapping
        self.label_to_idx = {
            'normal': 0,
            'mild_stenosis': 1,
            'moderate_stenosis': 2,
            'severe_stenosis': 3
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        # Print dataset info
        print(f"{split.capitalize()} dataset: {len(self.data_frame)} samples")
        label_counts = self.data_frame['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  - {label}: {count} ({count/len(self.data_frame)*100:.1f}%)")
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['filepath']
        
        # Enhanced image loading with error handling
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Basic quality check - skip very dark or very bright images
            np_img = np.array(image)
            if np_img.mean() < 10 or np_img.mean() > 245:
                # Use histogram equalization for poor contrast images
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                enhanced = cv2.equalizeHist(gray)
                image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
                
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
        """Calculate class weights for imbalanced dataset."""
        labels = [self.label_to_idx[label] for label in self.data_frame['label']]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        return torch.FloatTensor(class_weights)


class ImprovedCarotidClassifier(nn.Module):
    """Enhanced CNN with better architecture for medical imaging."""
    
    def __init__(self, num_classes=4, pretrained=True, model_name='efficientnet_b3', dropout_rate=0.5):
        super(ImprovedCarotidClassifier, self).__init__()
        
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # Choose backbone
        if model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Enhanced classifier with better regularization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class EnhancedTrainer:
    """Improved trainer with better optimization strategies."""
    
    def __init__(self, model, device, train_loader, val_loader, test_loader, 
                 results_dir='results', models_dir='models'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
    def train_epoch(self, optimizer, criterion, scaler=None):
        """Enhanced training with mixed precision."""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_bar = tqdm(self.train_loader, desc='Training')
        for images, labels, _ in train_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, criterion):
        """Enhanced validation with detailed metrics."""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct_predictions / total_samples
        
        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return epoch_loss, epoch_acc, f1
    
    def train(self, num_epochs=30, learning_rate=0.0001, weight_decay=1e-4, 
              use_class_weights=True, use_mixed_precision=True):
        """Enhanced training with better optimization."""
        print(f"Enhanced Training Configuration:")
        print(f"- Device: {self.device}")
        print(f"- Model: {self.model.model_name}")
        print(f"- Epochs: {num_epochs}")
        print(f"- Learning Rate: {learning_rate}")
        print(f"- Mixed Precision: {use_mixed_precision}")
        print(f"- Class Weights: {use_class_weights}")
        
        # Get class weights
        if use_class_weights:
            class_weights = self.train_loader.dataset.get_class_weights().to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"- Class Weights: {class_weights.cpu().numpy()}")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Enhanced optimizer with different learning rates for backbone and classifier
        backbone_params = list(self.model.backbone.parameters())
        classifier_params = list(self.model.classifier.parameters())
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained backbone
            {'params': classifier_params, 'lr': learning_rate}       # Higher LR for new classifier
        ], weight_decay=weight_decay)
        
        # Enhanced learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[learning_rate * 0.1, learning_rate],
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.device.type == 'cuda' else None
        
        best_val_f1 = 0.0
        best_model_state = None
        patience = 8
        patience_counter = 0
        
        print(f"\nStarting enhanced training...")
        print("="*70)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(optimizer, criterion, scaler)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self.validate_epoch(criterion)
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"LR Backbone: {optimizer.param_groups[0]['lr']:.6f}, LR Classifier: {optimizer.param_groups[1]['lr']:.6f}")
            
            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"[NEW BEST] New best validation F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
                
            # Update learning rate
            scheduler.step()
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        # Save final model
        model_path = self.models_dir / 'enhanced_carotid_classifier_best.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'model_name': self.model.model_name,
            'best_val_f1': best_val_f1,
            'history': self.history,
            'config': {
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'use_class_weights': use_class_weights
            }
        }, model_path)
        
        print(f"\nEnhanced training completed!")
        print(f"Best validation F1: {best_val_f1:.4f}")
        print(f"Model saved to: {model_path}")
        
        return self.history
    
    def evaluate(self):
        """Enhanced evaluation with more detailed metrics."""
        print(f"\nEnhanced evaluation on test set...")
        print("="*50)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        class_correct = [0] * 4
        class_total = [0] * 4
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"Overall Test Accuracy: {accuracy:.4f}")
        print(f"Overall Test F1-Score: {f1:.4f}")
        
        # Per-class accuracy
        class_names = ['Normal', 'Mild Stenosis', 'Moderate Stenosis', 'Severe Stenosis']
        print(f"\nPer-class Accuracy:")
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"- {class_name}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        # Enhanced confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Enhanced Confusion Matrix - Carotid Stenosis Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        eval_results = {
            'test_accuracy': float(accuracy),
            'test_f1_score': float(f1),
            'per_class_accuracy': {class_names[i]: class_correct[i]/class_total[i] if class_total[i] > 0 else 0 
                                 for i in range(len(class_names))},
            'classification_report': classification_report(all_labels, all_predictions, 
                                                         target_names=class_names, output_dict=True),
            'confusion_matrix': cm.tolist()
        }
        
        with open(self.results_dir / 'enhanced_evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def plot_enhanced_history(self):
        """Plot enhanced training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Training Loss', color='blue')
        ax1.plot(self.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Training Accuracy', color='blue')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(self.history['learning_rates'], label='Learning Rate', color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        ax4.plot(loss_diff, label='Val Loss - Train Loss', color='orange')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Indicator')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def get_enhanced_transforms():
    """Get enhanced data transforms with medical image considerations."""
    
    # Enhanced training transforms
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
    # Test transforms
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def main():
    """Enhanced main training function."""
    print("Enhanced Carotid Artery Stenosis Classification Training!")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Data paths
    csv_file = 'data/processed/image_labels_with_splits.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run data_exploration.py first.")
        return
    
    # Get enhanced transforms
    train_transforms, val_transforms = get_enhanced_transforms()
    
    # Create enhanced datasets
    train_dataset = AdvancedCarotidDataset(csv_file, transform=train_transforms, split='train')
    val_dataset = AdvancedCarotidDataset(csv_file, transform=val_transforms, split='val')
    test_dataset = AdvancedCarotidDataset(csv_file, transform=val_transforms, split='test')
    
    # Enhanced data loaders with better sampling
    batch_size = 32  # Increased batch size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    # Create enhanced model
    model = ImprovedCarotidClassifier(
        num_classes=4, 
        pretrained=True, 
        model_name='efficientnet_b3',  # Better architecture
        dropout_rate=0.4
    )
    
    # Create enhanced trainer
    trainer = EnhancedTrainer(model, device, train_loader, val_loader, test_loader)
    
    # Enhanced training with better hyperparameters
    history = trainer.train(
        num_epochs=35,
        learning_rate=0.001,
        weight_decay=1e-4,
        use_class_weights=True,
        use_mixed_precision=True
    )
    
    # Plot enhanced training history
    trainer.plot_enhanced_history()
    
    # Enhanced evaluation
    eval_results = trainer.evaluate()
    
    print(f"\n{'ENHANCED TRAINING COMPLETED!':^70}")
    print("="*70)
    print(f"Final Test Accuracy: {eval_results['test_accuracy']:.4f}")
    print(f"Final Test F1-Score: {eval_results['test_f1_score']:.4f}")
    print("\nKey Improvements:")
    print("- EfficientNet-B3 architecture (better for medical images)")
    print("- Enhanced data augmentation")
    print("- Class-balanced training")
    print("- Mixed precision training") 
    print("- OneCycle learning rate scheduling")
    print("- Early stopping with F1-score monitoring")
    print("- Gradient clipping for stability")
    print("\nCheck 'results' directory for detailed analysis!")


if __name__ == "__main__":
    main()