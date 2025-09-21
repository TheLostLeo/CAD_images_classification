#!/usr/bin/env python3
"""
Complete Model Comparison: EfficientNet vs Original CNN vs Improved CNN
Comprehensive analysis of all three models for carotid artery stenosis classification
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Import model architectures
from train_efficientnet_model import ImprovedCarotidClassifier
from train_custom_cnn import MedicalUltrasoundCNN
from train_improved_cnn import ImprovedMedicalCNN


class ComprehensiveModelComparison:
    """Complete comparison framework for all three models."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_names = ['Normal', 'Mild Stenosis', 'Moderate Stenosis', 'Severe Stenosis']
        
        # Load all models
        self._load_models()
        
        # Setup data transform
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_models(self):
        """Load all three trained models."""
        model_configs = [
            {
                'name': 'EfficientNet-B3',
                'path': 'models/enhanced_carotid_classifier_best.pth',
                'class': ImprovedCarotidClassifier,
                'color': '#2E86AB'
            },
            {
                'name': 'Original CNN',
                'path': 'models/custom_cnn_carotid_classifier.pth',
                'class': MedicalUltrasoundCNN,
                'color': '#A23B72'
            },
            {
                'name': 'Improved CNN',
                'path': 'models/improved_custom_cnn_carotid_classifier.pth',
                'class': ImprovedMedicalCNN,
                'color': '#F18F01'
            }
        ]
        
        for config in model_configs:
            try:
                model = config['class'](num_classes=4)
                checkpoint = torch.load(config['path'], map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models[config['name']] = {
                    'model': model,
                    'color': config['color'],
                    'best_f1': checkpoint.get('best_val_f1', 0.0),
                    'path': config['path']
                }
                print(f"Loaded {config['name']} (F1: {checkpoint.get('best_val_f1', 0.0):.4f})")
            except Exception as e:
                print(f"Failed to load {config['name']}: {e}")
    
    def predict_image(self, image_path):
        """Get predictions from all models for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            predictions = {}
            with torch.no_grad():
                for name, model_info in self.models.items():
                    outputs = model_info['model'](image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                    
                    predictions[name] = {
                        'class': predicted_class,
                        'class_name': self.class_names[predicted_class],
                        'confidence': confidence,
                        'probabilities': probabilities[0].cpu().numpy()
                    }
            
            return predictions
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def evaluate_on_test_set(self):
        """Comprehensive evaluation on test set."""
        # Load test data
        test_df = pd.read_csv('data/processed/test_labels.csv')
        
        results = {}
        all_predictions = {}
        all_true_labels = []
        
        # Initialize prediction storage
        for name in self.models.keys():
            all_predictions[name] = []
        
        print("Evaluating all models on test set...")
        
        for idx, row in test_df.iterrows():
            image_path = Path('data/raw/Common Carotid Artery Ultrasound Images/Common Carotid Artery Ultrasound Images/US images') / row['filename']
            true_label = self._label_to_idx(row['stenosis_class'])
            all_true_labels.append(true_label)
            
            predictions = self.predict_image(image_path)
            if predictions:
                for name in self.models.keys():
                    all_predictions[name].append(predictions[name]['class'])
            else:
                # Handle failed predictions
                for name in self.models.keys():
                    all_predictions[name].append(0)  # Default to normal
        
        # Calculate metrics for each model
        for name in self.models.keys():
            y_true = all_true_labels
            y_pred = all_predictions[name]
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred,
                'classification_report': classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }
        
        results['true_labels'] = all_true_labels
        return results
    
    def _label_to_idx(self, label):
        """Convert label string to index."""
        label_map = {
            'normal': 0,
            'mild_stenosis': 1,
            'moderate_stenosis': 2,
            'severe_stenosis': 3
        }
        return label_map[label]
    
    def create_comparison_visualization(self, results):
        """Create comparison visualization between models."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        model_names = list(self.models.keys())
        
        # 1. Performance Comparison Bar Chart
        ax = axes[0, 0]
        accuracies = [results[name]['accuracy'] for name in model_names]
        f1_scores = [results[name]['f1_score'] for name in model_names]
        colors = [self.models[name]['color'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2-4. Confusion Matrices
        for idx, name in enumerate(model_names):
            if idx < 3:  # Only plot first 3 models
                ax = axes[0, idx] if idx == 0 else (axes[0, idx] if idx < 2 else axes[1, idx-2])
                if idx > 0:  # Skip first position as it's already used
                    ax = axes[0, 1] if idx == 1 else axes[0, 2]
                
                cm = results[name]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
                ax.set_title(f'{name}\\nF1: {results[name]["f1_score"]:.3f}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
        
        # Model Size Comparison
        ax = axes[1, 0]
        param_counts = []
        for name in model_names:
            model = self.models[name]['model']
            param_count = sum(p.numel() for p in model.parameters()) / 1e6  # In millions
            param_counts.append(param_count)
        
        bars = ax.bar(model_names, param_counts, color=colors, alpha=0.8)
        ax.set_ylabel('Parameters (Millions)')
        ax.set_title('Model Size Comparison')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, param_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count:.1f}M', ha='center', va='bottom')
        
        # Class Distribution
        ax = axes[1, 1]
        true_dist = np.bincount(results['true_labels'], minlength=4)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, true_dist, width, label='True Distribution', alpha=0.8, color='gray')
        
        for i, name in enumerate(model_names):
            pred_dist = np.bincount(results[name]['predictions'], minlength=4)
            ax.bar(x + i*width*0.8, pred_dist, width*0.8, label=name, 
                   color=self.models[name]['color'], alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution vs True Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace(' ', '\\n') for name in self.class_names])
        ax.legend()
        
        # Summary Text
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "MODEL COMPARISON SUMMARY\\n\\n"
        best_model = max(model_names, key=lambda x: results[x]['f1_score'])
        summary_text += f"Best Model: {best_model}\\n"
        summary_text += f"F1-Score: {results[best_model]['f1_score']:.4f}\\n\\n"
        
        summary_text += "All Model F1-Scores:\\n"
        for name in model_names:
            f1 = results[name]['f1_score']
            summary_text += f"• {name}: {f1:.4f}\\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig('models/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, results):
        """Generate detailed comparison report."""
        print("\\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        model_names = list(self.models.keys())
        
        # Summary comparison
        print("\\nPERFORMANCE SUMMARY:")
        print("-" * 50)
        for name in model_names:
            accuracy = results[name]['accuracy']
            f1 = results[name]['f1_score']
            params = sum(p.numel() for p in self.models[name]['model'].parameters())
            
            print(f"\\n{name}:")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"F1-Score: {f1:.4f}")
            print(f"Parameters: {params:,}")
        
        # Best model
        best_model = max(model_names, key=lambda x: results[x]['f1_score'])
        print(f"\\nBest Performing Model: {best_model}")
        print(f"   F1-Score: {results[best_model]['f1_score']:.4f}")
        
        # Model agreement
        print("\\nModel Agreement Analysis:")
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names[i+1:], i+1):
                pred1 = np.array(results[name1]['predictions'])
                pred2 = np.array(results[name2]['predictions'])
                agreement = np.mean(pred1 == pred2)
                print(f"  {name1} vs {name2}: {agreement:.4f} ({agreement*100:.1f}% agreement)")
        
        return results
    
    def run_comprehensive_comparison(self):
        """Run complete comparison analysis."""
        print("Starting Comprehensive Model Comparison")
        print("=" * 60)
        
        # Evaluate all models
        results = self.evaluate_on_test_set()
        
        # Create visualizations
        self.create_comparison_visualization(results)
        
        # Generate detailed report
        report = self.generate_detailed_report(results)
        
        print("\\nComprehensive comparison completed!")
        print("Visualization saved to: models/comprehensive_model_comparison.png")
        
        return results, report


def main():
    """Main function to run comprehensive comparison."""
    comparison = ComprehensiveModelComparison()
    
    if not comparison.models:
        print("No models could be loaded. Please ensure all model files exist.")
        return
    
    print(f"\\nLoaded {len(comparison.models)} models for comparison:")
    for name in comparison.models.keys():
        print(f"  - {name}")
    
    # Run comprehensive comparison
    results, report = comparison.run_comprehensive_comparison()
    
    print("\\nFINAL SUMMARY:")
    print("-" * 40)
    for name in comparison.models.keys():
        f1 = results[name]['f1_score']
        acc = results[name]['accuracy']
        print(f"{name}: F1={f1:.4f}, Accuracy={acc:.4f}")


if __name__ == "__main__":
    main()