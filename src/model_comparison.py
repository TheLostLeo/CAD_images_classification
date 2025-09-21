#!/usr/bin/env python3
"""
Two-Model Comparison: EfficientNet-B3 vs Custom CNN
Comprehensive analysis comparing transfer learning vs custom architecture
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


class TwoModelComparison:
    """Comparison framework for EfficientNet-B3 vs Custom CNN."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_names = ['Normal', 'Mild Stenosis', 'Moderate Stenosis', 'Severe Stenosis']
        
        # Load both models
        self._load_models()
        
        # Setup data transform
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_models(self):
        """Load both trained models."""
        model_configs = [
            {
                'name': 'EfficientNet-B3',
                'path': 'models/enhanced_carotid_classifier_best.pth',
                'class': ImprovedCarotidClassifier,
                'color': '#2E86AB',
                'type': 'Transfer Learning'
            },
            {
                'name': 'Custom CNN',
                'path': 'models/custom_cnn_carotid_classifier.pth',
                'class': MedicalUltrasoundCNN,
                'color': '#F18F01',
                'type': 'Custom Architecture'
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
                    'type': config['type'],
                    'best_f1': checkpoint.get('best_val_f1', 0.0),
                    'path': config['path']
                }
                print(f"[SUCCESS] Loaded {config['name']} ({config['type']}) - F1: {checkpoint.get('best_val_f1', 0.0):.4f}")
            except Exception as e:
                print(f"[ERROR] Failed to load {config['name']}: {e}")
    
    def predict_image(self, image_path):
        """Get predictions from both models for a single image."""
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
        
        print("Evaluating both models on test set...")
        
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
        """Create comprehensive comparison visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        model_names = list(self.models.keys())
        colors = [self.models[name]['color'] for name in model_names]
        
        # 1. Performance Comparison
        ax = axes[0, 0]
        metrics = ['Accuracy', 'F1-Score']
        model1_scores = [results[model_names[0]]['accuracy'], results[model_names[0]]['f1_score']]
        model2_scores = [results[model_names[1]]['accuracy'], results[model_names[1]]['f1_score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, model1_scores, width, label=model_names[0], 
                      color=colors[0], alpha=0.8)
        bars2 = ax.bar(x + width/2, model2_scores, width, label=model_names[1], 
                      color=colors[1], alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(max(model1_scores), max(model2_scores)) + 0.1)
        
        # 2-3. Confusion Matrices
        for idx, name in enumerate(model_names):
            ax = axes[0, idx + 1]
            cm = results[name]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=[n.replace(' ', '\\n') for n in self.class_names], 
                       yticklabels=[n.replace(' ', '\\n') for n in self.class_names], 
                       ax=ax, cbar_kws={'shrink': 0.8})
            
            ax.set_title(f'{name}\\n({self.models[name]["type"]})\\nF1: {results[name]["f1_score"]:.3f}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # 4. Model Architecture Comparison
        ax = axes[1, 0]
        param_counts = []
        model_types = []
        
        for name in model_names:
            model = self.models[name]['model']
            param_count = sum(p.numel() for p in model.parameters()) / 1e6  # In millions
            param_counts.append(param_count)
            model_types.append(self.models[name]['type'])
        
        bars = ax.bar(model_names, param_counts, color=colors, alpha=0.8)
        ax.set_ylabel('Parameters (Millions)')
        ax.set_title('Model Architecture Comparison')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Add value labels and type annotations
        for bar, count, model_type in zip(bars, param_counts, model_types):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{count:.1f}M', ha='center', va='bottom', fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    model_type, ha='center', va='center', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 5. Per-Class Performance
        ax = axes[1, 1]
        class_indices = list(range(len(self.class_names)))
        
        model1_f1s = []
        model2_f1s = []
        
        for i in class_indices:
            cls_key = str(i)
            f1_1 = results[model_names[0]]['classification_report'].get(cls_key, {}).get('f1-score', 0.0)
            f1_2 = results[model_names[1]]['classification_report'].get(cls_key, {}).get('f1-score', 0.0)
            model1_f1s.append(f1_1)
            model2_f1s.append(f1_2)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        ax.bar(x - width/2, model1_f1s, width, label=model_names[0], 
               color=colors[0], alpha=0.8)
        ax.bar(x + width/2, model2_f1s, width, label=model_names[1], 
               color=colors[1], alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('F1-Score')
        ax.set_title('Per-Class Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace(' ', '\\n') for name in self.class_names])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 6. Model Comparison Summary
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate improvements
        f1_diff = results[model_names[0]]['f1_score'] - results[model_names[1]]['f1_score']
        acc_diff = results[model_names[0]]['accuracy'] - results[model_names[1]]['accuracy']
        
        # Model agreement
        pred1 = np.array(results[model_names[0]]['predictions'])
        pred2 = np.array(results[model_names[1]]['predictions'])
        agreement = np.mean(pred1 == pred2)
        
        summary_text = "COMPARISON SUMMARY\\n\\n"
        
        if f1_diff > 0:
            summary_text += f"Winner: {model_names[0]}\\n"
            summary_text += f"   F1 Advantage: +{f1_diff:.4f}\\n"
        else:
            summary_text += f"Winner: {model_names[1]}\\n"
            summary_text += f"   F1 Advantage: +{abs(f1_diff):.4f}\\n"
        
        summary_text += f"\\nPerformance Metrics:\\n"
        for name in model_names:
            summary_text += f"• {name}:\\n"
            summary_text += f"  F1: {results[name]['f1_score']:.4f}\\n"
            summary_text += f"  Acc: {results[name]['accuracy']:.4f}\\n"
        
        summary_text += f"\\nModel Agreement: {agreement:.1%}\\n"
        summary_text += f"\\nArchitecture Types:\\n"
        for name in model_names:
            param_count = sum(p.numel() for p in self.models[name]['model'].parameters()) / 1e6
            summary_text += f"• {self.models[name]['type']}: {param_count:.1f}M params\\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('models/two_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, results):
        """Generate detailed comparison report."""
        print("\\n" + "="*80)
        print("TWO-MODEL COMPARISON REPORT")
        print("Transfer Learning vs Custom Architecture")
        print("="*80)
        
        model_names = list(self.models.keys())
        
        # Performance Summary
        print("\\nPERFORMANCE SUMMARY:")
        print("-" * 50)
        for name in model_names:
            accuracy = results[name]['accuracy']
            f1 = results[name]['f1_score']
            params = sum(p.numel() for p in self.models[name]['model'].parameters())
            model_type = self.models[name]['type']
            
            print(f"\\n{name} ({model_type}):")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Parameters: {params:,}")
        
        # Winner determination
        best_model = max(model_names, key=lambda x: results[x]['f1_score'])
        worst_model = min(model_names, key=lambda x: results[x]['f1_score'])
        
        f1_improvement = results[best_model]['f1_score'] - results[worst_model]['f1_score']
        acc_improvement = results[best_model]['accuracy'] - results[worst_model]['accuracy']
        
        print(f"\\nBest Performing Model: {best_model}")
        print(f"   F1-Score: {results[best_model]['f1_score']:.4f}")
        print(f"   Improvement over {worst_model}: +{f1_improvement:.4f} F1 (+{f1_improvement/results[worst_model]['f1_score']*100:.1f}%)")
        
        # Model agreement
        pred1 = np.array(results[model_names[0]]['predictions'])
        pred2 = np.array(results[model_names[1]]['predictions'])
        agreement = np.mean(pred1 == pred2)
        
        print(f"\\nModel Agreement: {agreement:.4f} ({agreement*100:.1f}%)")
        
        # Architecture Analysis
        print(f"\\n🏗️ ARCHITECTURE ANALYSIS:")
        print("-" * 30)
        for name in model_names:
            model_type = self.models[name]['type']
            param_count = sum(p.numel() for p in self.models[name]['model'].parameters()) / 1e6
            
            if model_type == 'Transfer Learning':
                print(f"\\n{name}:")
                print(f"  • Pre-trained on ImageNet")
                print(f"  • EfficientNet-B3 backbone")
                print(f"  • Fine-tuned for medical imaging")
                print(f"  • Parameters: {param_count:.1f}M")
            else:
                print(f"\\n{name}:")
                print(f"  • Built from scratch")
                print(f"  • Residual connections")
                print(f"  • Attention mechanisms")
                print(f"  • Medical-specific design")
                print(f"  • Parameters: {param_count:.1f}M")
        
        # Key Insights
        print(f"\\nKEY INSIGHTS:")
        print("-" * 20)
        
        if 'EfficientNet-B3' in model_names and results['EfficientNet-B3']['f1_score'] > results['Custom CNN']['f1_score']:
            print("• Transfer learning outperforms custom architecture")
            print("• Pre-trained features provide strong foundation")
            print("• Less training data needed for good performance")
        else:
            print("• Custom architecture shows competitive performance")
            print("• Medical-specific design provides benefits")
            print("• From-scratch training can be effective")
        
        print(f"• Model agreement of {agreement:.1%} suggests different learned representations")
        print("• Both models could potentially be ensembled for better performance")
        
        return results
    
    def run_comparison(self):
        """Run complete two-model comparison."""
        print("Starting Two-Model Comparison")
        print("Transfer Learning vs Custom Architecture")
        print("=" * 60)
        
        # Evaluate both models
        results = self.evaluate_on_test_set()
        
        # Create visualizations
        self.create_comparison_visualization(results)
        
        # Generate detailed report
        report = self.generate_detailed_report(results)
        
        print("\\nTwo-model comparison completed!")
        print("Visualization saved to: models/two_model_comparison.png")
        
        return results, report


def main():
    """Main function to run two-model comparison."""
    comparison = TwoModelComparison()
    
    if len(comparison.models) < 2:
        print("[ERROR] Need at least 2 models for comparison. Please ensure model files exist.")
        return
    
    print(f"\\nLoaded {len(comparison.models)} models for comparison:")
    for name, info in comparison.models.items():
        print(f"  - {name} ({info['type']})")
    
    # Run comparison
    results, report = comparison.run_comparison()
    
    print("\\nFINAL SUMMARY:")
    print("-" * 40)
    for name in comparison.models.keys():
        f1 = results[name]['f1_score']
        acc = results[name]['accuracy']
        model_type = comparison.models[name]['type']
        print(f"{name} ({model_type}): F1={f1:.4f}, Accuracy={acc:.4f}")


if __name__ == "__main__":
    main()