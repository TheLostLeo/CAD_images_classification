# Carotid Artery Stenosis Detection

A deep learning project for classifying carotid artery stenosis severity from ultrasound images using CNN and transfer learning.

## 🎯 Project Overview

This project aims to develop a CNN-based classifier to detect and classify carotid artery stenosis into four categories:
- **Normal**: No significant stenosis (0-29%)
- **Mild**: Mild stenosis (30-49%)
- **Moderate**: Moderate stenosis (50-69%)
- **Severe**: Severe stenosis (70-99%)

## 📊 Dataset

- **Source**: Carotid Ultrasound Images from Kaggle
- **Size**: 1,100 ultrasound images from 35 patients/studies
- **Format**: 709×749 pixel RGB images
- **Labels**: Synthetic labels created for demonstration (4 classes)

## 🗂️ Project Structure

```
CAD_images_classification/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Preprocessed images and labels
├── models/                  # Trained models
├── src/                     # Source code
│   ├── data_exploration.py  # Dataset analysis and preprocessing
│   ├── train_model.py      # CNN training pipeline
│   └── inference.py        # Prediction script
├── results/                # Training results and visualizations
├── demo.py                 # Complete pipeline demonstration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone or download the project
cd CAD_images_classification

# Activate your Python environment
pyenv activate cad-env  # or your preferred environment

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Using Kaggle API (recommended)
kaggle datasets download -d orvile/carotid-ultrasound-images
unzip carotid-ultrasound-images.zip -d data/raw/

# OR download manually from Kaggle and extract to data/raw/
```

### 3. Run Complete Pipeline
```bash
# Option 1: Run everything with demo script
python demo.py

# Option 2: Run each step individually
python src/data_exploration.py      # Analyze and preprocess data
python src/train_model.py          # Train the CNN model
python src/inference.py --image path/to/test_image.png --visualize --report
```

## 🔬 Technical Details

### Model Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned for 4-class medical classification
- **Custom Head**: Fully connected layers with dropout for regularization
- **Input Size**: 224×224 RGB images
- **Output**: 4 stenosis severity classes

### Data Processing
- **Patient-level splits**: Train/Val/Test to prevent data leakage
- **Data Augmentation**: Rotation, flipping, color jitter, affine transforms
- **Normalization**: ImageNet statistics for transfer learning
- **Class Balance**: Handles imbalanced medical data

### Training Details
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 16 (GPU memory optimized)
- **Epochs**: 15-20 for full training

## 📈 Results

### Expected Performance
- **Target Accuracy**: >85% on test set
- **Metrics**: Precision, Recall, F1-score per class
- **Visualizations**: Training curves, confusion matrix, ROC curves

### Outputs Generated
- **Training History**: Loss and accuracy plots
- **Confusion Matrix**: Detailed classification performance
- **Clinical Reports**: AI-generated assessment reports
- **Model Checkpoints**: Best performing model saved

## 🏥 Clinical Relevance

Carotid artery stenosis is a major risk factor for stroke. This AI system can:
- **Assist radiologists** in diagnosis and screening
- **Provide consistent** stenosis severity assessment
- **Enable faster** patient triage and care
- **Support telemedicine** applications in remote areas
- **Reduce human error** in routine screening

## 🎓 Educational Value

Perfect for AI/ML classes covering:
- **Medical Image Analysis**: Real ultrasound data
- **Transfer Learning**: Pre-trained model fine-tuning
- **Computer Vision**: CNN architectures and training
- **Healthcare AI**: Ethics and clinical applications
- **Data Science**: Proper train/val/test methodology

## 📝 Usage Examples

### Training a Model
```bash
# Full training with all features
python src/train_model.py

# Results saved to models/ and results/
```

### Making Predictions
```bash
# Single image with visualization
python src/inference.py --image test_image.png --visualize --report

# Batch processing
python src/inference.py --batch images_directory/ --output results/

# Clinical report generation
python src/inference.py --image scan.png --report
```

### Data Analysis
```bash
# Explore dataset structure and statistics
python src/data_exploration.py

# Outputs: processed labels, data splits, sample visualizations
```

## ⚠️ Important Notes

- **Synthetic Labels**: This demo uses synthetic labels for educational purposes
- **Clinical Use**: Real deployment requires medical expert validation
- **Research Only**: Not intended for actual clinical diagnosis
- **Data Privacy**: Follow HIPAA guidelines for real medical data

## 🔧 Customization

### Hyperparameter Tuning
```python
# In train_model.py, modify:
trainer.train(
    num_epochs=20,           # Increase for better accuracy
    learning_rate=0.0005,    # Lower for fine-tuning
    weight_decay=1e-3        # Adjust regularization
)
```

### Model Architecture
```python
# Try different backbones:
model = CarotidClassifier(model_name='efficientnet_b0')  # Lighter model
model = CarotidClassifier(model_name='resnet50')         # Standard choice
```

### Data Augmentation
```python
# Modify transforms in train_model.py for medical data:
transforms.RandomRotation(degrees=5)        # Reduce rotation
transforms.ColorJitter(brightness=0.1)      # Subtle color changes
```

## 🚀 Future Improvements

- **Segmentation**: Add vessel wall and plaque segmentation
- **Multi-modal**: Combine with Doppler ultrasound data
- **Ensemble Methods**: Multiple model voting for reliability
- **Explainable AI**: Gradient-CAM for decision visualization
- **Real Labels**: Collaborate with medical institutions
- **3D Analysis**: Process ultrasound video sequences

## 📚 References

- **Medical Background**: Carotid stenosis and stroke risk
- **Technical**: ResNet, Transfer Learning, Medical AI
- **Dataset**: Kaggle Carotid Ultrasound Images
- **Libraries**: PyTorch, Torchvision, OpenCV, Matplotlib

## 🤝 Contributing

This is an educational project. Suggestions for improvements:
1. Better data augmentation strategies
2. Advanced model architectures
3. Clinical validation metrics
4. User interface development

---

**🎓 Perfect for AI Class Projects!**  
This project demonstrates real-world application of deep learning in healthcare with proper methodology and clinical relevance.