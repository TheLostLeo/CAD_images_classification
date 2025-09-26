#!/usr/bin/env python3
"""
Inference Script for Carotid Artery Stenosis Classification
CLI tool to run inference on ultrasound images using trained models
"""

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import model architectures
from custom_cnn import BinaryCustomCNN, MedicalAttention


class BinaryDenseNet(nn.Module):
    """DenseNet architecture for binary classification (placeholder)"""
    def __init__(self, num_classes=2):
        super(BinaryDenseNet, self).__init__()
        # This is a placeholder - you can replace with actual DenseNet implementation
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CADInference:
    """Inference class for carotid artery stenosis classification"""
    
    def __init__(self, model_path, model_type='custom_cnn', device='auto'):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to the trained model file (.pth)
            model_type: Type of model ('custom_cnn', 'densenet')
            device: Device to run inference on ('cuda', 'cpu', 'auto')
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ['Normal', 'Disease']
        
        print(f"Model loaded: {self.model_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self):
        """Load the specified model architecture and weights"""
        
        if self.model_type == 'custom_cnn':
            model = BinaryCustomCNN(num_classes=2, dropout_rate=0.5)
        elif self.model_type == 'densenet':
            model = BinaryDenseNet(num_classes=2)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")
        
        return model.to(self.device)
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for inference"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor.to(self.device), image
        
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {e}")
    
    def predict_single(self, image_path, return_probabilities=True):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to the image file
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Prepare results
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'image_size': original_image.size,
            'model_type': self.model_type
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.class_names[i]: probabilities[0][i].item() 
                for i in range(len(self.class_names))
            }
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=True):
        """Run inference on multiple images"""
        
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.predict_single(image_path, return_probabilities)
                results.append(result)
                
                print(f"[{i}/{len(image_paths)}] {Path(image_path).name}: {result['predicted_label']} ({result['confidence']:.3f})")
                
            except Exception as e:
                print(f"[{i}/{len(image_paths)}] Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory_path, return_probabilities=True):
        """Run inference on all images in a directory"""
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(directory_path).glob(f"*{ext}"))
            image_paths.extend(Path(directory_path).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            raise ValueError(f"No images found in directory: {directory_path}")
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        return self.predict_batch(image_paths, return_probabilities)


def print_results(results, detailed=False):
    """Print inference results in a formatted way"""
    
    if isinstance(results, dict):  # Single result
        results = [results]
    
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"\n{i}. ERROR: {result['image_path']}")
            print(f"   {result['error']}")
            continue
        
        image_name = Path(result['image_path']).name
        print(f"\n{i}. {image_name}")
        print(f"   Prediction: {result['predicted_label']}")
        print(f"   Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        
        if detailed and 'probabilities' in result:
            print(f"   Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"     {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        
        if detailed:
            print(f"   Model: {result['model_type']}")
            print(f"   Image size: {result['image_size']}")
    
    # Summary
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        normal_count = sum(1 for r in successful_results if r['predicted_class'] == 0)
        disease_count = sum(1 for r in successful_results if r['predicted_class'] == 1)
        
        print(f"\n" + "-" * 40)
        print(f"SUMMARY:")
        print(f"  Total images: {len(results)}")
        print(f"  Successful predictions: {len(successful_results)}")
        print(f"  Normal: {normal_count}")
        print(f"  Disease: {disease_count}")
        
        if len(successful_results) > 0:
            avg_confidence = np.mean([r['confidence'] for r in successful_results])
            print(f"  Average confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
    
    print("=" * 80)


def save_results(results, output_path):
    """Save results to JSON file"""
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Carotid Artery Stenosis Classification - Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with custom CNN
  python inference_cli.py --image path/to/image.png --model custom_cnn --model-path models/custom_cnn.pth
  
  # Directory with DenseNet
  python inference_cli.py --directory data/test_images --model densenet --model-path models/densenet.pth
  
  # Batch processing with detailed output
  python inference_cli.py --images img1.png img2.png img3.png --model custom_cnn --model-path models/custom_cnn.pth --detailed
  
  # Save results to JSON
  python inference_cli.py --directory test_data --model custom_cnn --model-path models/custom_cnn.pth --output results.json
        """
    )
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to single image file')
    input_group.add_argument('--images', nargs='+', help='Paths to multiple image files')
    input_group.add_argument('--directory', type=str, help='Path to directory containing images')
    
    # Model arguments
    parser.add_argument('--model', type=str, choices=['custom_cnn', 'densenet'], 
                       default='custom_cnn', help='Model type to use (default: custom_cnn)')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to trained model file (.pth)')
    
    # Device argument
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], 
                       default='auto', help='Device to run inference on (default: auto)')
    
    # Output arguments
    parser.add_argument('--output', type=str, help='Path to save results as JSON file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    parser.add_argument('--no-probabilities', action='store_true', help='Skip probability calculation')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference pipeline
        inference = CADInference(
            model_path=args.model_path,
            model_type=args.model,
            device=args.device
        )
        
        # Run inference based on input type
        return_probs = not args.no_probabilities
        
        if args.image:
            results = inference.predict_single(args.image, return_probs)
        elif args.images:
            results = inference.predict_batch(args.images, return_probs)
        elif args.directory:
            results = inference.predict_directory(args.directory, return_probs)
        
        # Print results
        print_results(results, detailed=args.detailed)
        
        # Save results if requested
        if args.output:
            save_results(results, args.output)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()