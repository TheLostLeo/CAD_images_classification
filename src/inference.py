#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import argparse
import sys
from pathlib import Path
import cv2
from datetime import datetime

# Import our model class
from train_model import ImprovedCarotidClassifier


class CarotidInference:
    """Inference class for Carotid Stenosis Classification."""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to the trained model file
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # Class mappings
        self.idx_to_label = {
            0: 'normal',
            1: 'mild_stenosis', 
            2: 'moderate_stenosis',
            3: 'severe_stenosis'
        }
        
        self.label_descriptions = {
            'normal': 'Normal (0-29% stenosis)',
            'mild_stenosis': 'Mild Stenosis (30-49%)',
            'moderate_stenosis': 'Moderate Stenosis (50-69%)',
            'severe_stenosis': 'Severe Stenosis (70-99%)'
        }
        
        # Load model
        self.model = self._load_model()
        
        # Define transforms (matching enhanced model)
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create enhanced model
        model = ImprovedCarotidClassifier(num_classes=4)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Enhanced model loaded successfully!")
        print(f"Best validation F1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
        
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device), image
    
    def predict_single(self, image_path, return_probs=True):
        """
        Predict stenosis class for a single image.
        
        Args:
            image_path (str): Path to the image file
            return_probs (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        image_tensor, original_image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_label = self.idx_to_label[predicted_class]
        
        result = {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'description': self.label_descriptions[predicted_label],
            'confidence': confidence
        }
        
        if return_probs:
            result['class_probabilities'] = {
                self.label_descriptions[self.idx_to_label[i]]: probabilities[0][i].item()
                for i in range(len(self.idx_to_label))
            }
        
        return result, original_image
    
    def predict_batch(self, image_paths, return_probs=True):
        """
        Predict stenosis classes for multiple images.
        
        Args:
            image_paths (list): List of image file paths
            return_probs (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result, _ = self.predict_single(image_path, return_probs)
                results.append(result)
                print(f"✓ Processed: {Path(image_path).name}")
            except Exception as e:
                print(f"✗ Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction results with the original image.
        
        Args:
            image_path (str): Path to the image file
            save_path (str): Optional path to save the visualization
        """
        result, original_image = self.predict_single(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image
        ax1.imshow(original_image)
        ax1.set_title(f"Original Image\n{Path(image_path).name}", fontsize=12)
        ax1.axis('off')
        
        # Display prediction results
        if 'class_probabilities' in result:
            classes = list(result['class_probabilities'].keys())
            probs = list(result['class_probabilities'].values())
            
            colors = ['green' if i == result['predicted_class'] else 'lightblue' 
                     for i in range(len(classes))]
            
            bars = ax2.barh(classes, probs, color=colors)
            ax2.set_xlabel('Probability')
            ax2.set_title(f'Stenosis Classification Results\n'
                         f'Predicted: {result["description"]}\n'
                         f'Confidence: {result["confidence"]:.3f}', fontsize=12)
            ax2.set_xlim(0, 1)
            
            # Add probability values on bars
            for bar, prob in zip(bars, probs):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return result
    
    def clinical_report(self, image_path):
        """
        Generate a clinical-style report for the prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Clinical report
        """
        result, _ = self.predict_single(image_path)
        
        # Risk level mapping
        risk_levels = {
            'normal': 'Low',
            'mild_stenosis': 'Low-Moderate',
            'moderate_stenosis': 'Moderate-High',
            'severe_stenosis': 'High'
        }
        
        # Clinical recommendations
        recommendations = {
            'normal': 'Continue routine monitoring. Follow-up in 1-2 years.',
            'mild_stenosis': 'Monitor closely. Consider lifestyle modifications. Follow-up in 6-12 months.',
            'moderate_stenosis': 'Requires medical evaluation. Consider intervention. Follow-up in 3-6 months.',
            'severe_stenosis': 'Urgent medical evaluation required. High risk for stroke. Consider immediate intervention.'
        }
        
        report = f"""
CAROTID ARTERY STENOSIS ASSESSMENT REPORT
{'='*50}

Patient Image: {Path(image_path).name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINDINGS:
- Predicted Classification: {result['description']}
- Confidence Level: {result['confidence']:.1%}
- Stroke Risk Level: {risk_levels[result['predicted_label']]}

DETAILED PROBABILITIES:
"""
        
        if 'class_probabilities' in result:
            for class_name, prob in result['class_probabilities'].items():
                report += f"- {class_name}: {prob:.1%}\n"
        
        report += f"""
CLINICAL RECOMMENDATION:
{recommendations[result['predicted_label']]}

IMPORTANT NOTES:
- This is a computer-aided diagnosis tool for reference only
- Results should be interpreted by a qualified medical professional
- Clinical correlation with patient history and other imaging is essential
- This assessment is based on synthetic training data for demonstration purposes

DISCLAIMER:
This automated analysis is for educational and research purposes only.
Always consult with a qualified healthcare provider for medical decisions.
"""
        
        return report


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Carotid Stenosis Inference')
    parser.add_argument('--model', type=str, default='models/carotid_classifier_best.pth',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--batch', type=str, help='Path to directory with multiple images')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--report', action='store_true', help='Generate clinical report')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first by running: python src/train_model.py")
        return
    
    # Initialize inference engine
    try:
        inference = CarotidInference(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = Path('results/inference')
        output_dir.mkdir(exist_ok=True)
    
    # Single image inference
    if args.image:
        print(f"Processing image: {args.image}")
        
        try:
            if args.visualize:
                result = inference.visualize_prediction(
                    args.image, 
                    save_path=output_dir / f"{Path(args.image).stem}_prediction.png"
                )
            else:
                result, _ = inference.predict_single(args.image)
            
            # Print results
            print(f"\nPrediction Results:")
            print(f"- Classification: {result['description']}")
            print(f"- Confidence: {result['confidence']:.3f}")
            
            if 'class_probabilities' in result:
                print(f"\nClass Probabilities:")
                for class_name, prob in result['class_probabilities'].items():
                    print(f"- {class_name}: {prob:.3f}")
            
            # Generate clinical report
            if args.report:
                report = inference.clinical_report(args.image)
                print(report)
                
                # Save report
                report_file = output_dir / f"{Path(args.image).stem}_report.txt"
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"\nClinical report saved to: {report_file}")
            
            # Save JSON results
            json_file = output_dir / f"{Path(args.image).stem}_results.json"
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {json_file}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    # Batch processing
    if args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Error: Batch directory not found: {batch_dir}")
            return
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(batch_dir.glob(f"*{ext}"))
            image_files.extend(batch_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in: {batch_dir}")
            return
        
        print(f"Processing {len(image_files)} images from: {batch_dir}")
        
        # Process batch
        results = inference.predict_batch(image_files)
        
        # Save batch results
        batch_results_file = output_dir / 'batch_results.json'
        with open(batch_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch processing completed!")
        print(f"Results saved to: {batch_results_file}")


if __name__ == "__main__":
    # For testing without command line args
    if len(sys.argv) == 1:
        print("Carotid Stenosis Inference Tool")
        print("Usage examples:")
        print("  python src/inference.py --image path/to/image.png --visualize --report")
        print("  python src/inference.py --batch path/to/images/ --output results/")
        print("  python src/inference.py --model models/my_model.pth --image test.png")
    else:
        main()