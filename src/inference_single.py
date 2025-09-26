#!/usr/bin/env python3
"""
Simple Single Image Inference Script for Carotid Artery Stenosis Classification
CLI tool to run inference on a single ultrasound image using trained models
"""

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import json

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import model architectures
try:
    from custom_cnn import BinaryCustomCNN, MedicalAttention
except ImportError:
    print("Warning: Could not import custom_cnn. Make sure the module is available.")


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
    """Custom CNN architecture for binary classification"""
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
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        x = self.classifier(x)
        return x


def load_model(model_name, model_path, device):
    """Load the specified model"""
    
    print(f"Loading {model_name} model...")
    
    if model_name.lower() == 'custom_cnn':
        model = BinaryCustomCNN(num_classes=2, dropout_rate=0.5)
    else:
        print(f"Error: Unsupported model type: {model_name}")
        return None
    
    # Load weights
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(image_path, model_name='custom_cnn', model_path=None):
    """
    Run inference on a single image
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Default model path if not provided
    if model_path is None:
        if model_name.lower() == 'custom_cnn':
            model_path = 'models/custom_cnn.pth'
        else:
            print(f"Error: Please provide model path for {model_name}")
            return None
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Load model
    model = load_model(model_name, model_path, device)
    if model is None:
        return None
    
    model.eval()
    
    # Load and preprocess image
    try:
        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        print(f"Original image size: {image.size}")
        
        # Apply transforms
        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Preprocessed tensor shape: {image_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            probs = probabilities.cpu().numpy()[0]
        
        # Class names
        class_names = ['Normal', 'Disease']
        
        # Display results
        print(f"\nINFERENCE RESULTS")
        print("=" * 40)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Model: {model_name}")
        print(f"Prediction: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"\nClass Probabilities:")
        
        for i, class_name in enumerate(class_names):
            bar = "█" * int(probs[i] * 30)  # Progress bar
            print(f"  {class_name:8}: {probs[i]:.4f} ({probs[i]*100:.2f}%) {bar}")
        
        # Medical interpretation
        print(f"\nMedical Interpretation:")
        if predicted_class == 0:
            print("  No significant stenosis detected")
            if confidence > 0.8:
                print("  High confidence - Routine follow-up recommended")
            else:
                print("  Moderate confidence - Consider additional imaging")
        else:
            print("  Stenosis detected")
            if confidence > 0.8:
                print("  High confidence - Clinical evaluation advised")
            else:
                print("  Moderate confidence - Further assessment recommended")
        
        if confidence < 0.6:
            print(f"\nNote: Low confidence prediction ({confidence*100:.1f}%)")
            print("      Consider additional imaging or clinical correlation.")
        
        # Save results to JSON
        result = {
            'image_path': image_path,
            'model_used': model_name,
            'prediction': class_names[predicted_class],
            'prediction_index': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                'Normal': float(probs[0]),
                'Disease': float(probs[1])
            }
        }
        
        output_file = f"results/inference_{os.path.basename(image_path).split('.')[0]}.json"
        os.makedirs('results', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def main():
    """Main function for CLI"""
    
    parser = argparse.ArgumentParser(
        description="Single Image Inference for Carotid Artery Stenosis Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_single.py path/to/image.png
  python inference_single.py path/to/image.png --model custom_cnn
  python inference_single.py path/to/image.png --model custom_cnn --model_path models/my_model.pth
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the ultrasound image file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='custom_cnn',
        choices=['custom_cnn'],
        help='Model to use for inference (default: custom_cnn)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model weights file (default: models/{model}.pth)'
    )
    
    args = parser.parse_args()
    
    print("🩺 Carotid Artery Stenosis Classification - Single Image Inference")
    print("=" * 70)
    
    # Run inference
    result = predict_single_image(
        image_path=args.image_path,
        model_name=args.model,
        model_path=args.model_path
    )
    
    if result:
        print(f"\nInference completed successfully!")
    else:
        print(f"\nInference failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()