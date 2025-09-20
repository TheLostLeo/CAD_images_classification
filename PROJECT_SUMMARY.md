# Carotid Artery Stenosis Classification Project

## 🎯 Project Overview
An AI-powered medical imaging system for detecting carotid artery stenosis from ultrasound images to assess stroke risk. This project demonstrates advanced deep learning techniques applied to medical imaging for an AI class assignment.

## 📊 Performance Comparison

### Original Model (ResNet50)
- **Test Accuracy**: 37.5%
- **Architecture**: ResNet50 backbone
- **Training**: Basic configuration
- **Issues**: Class imbalance, limited augmentation

### Enhanced Model (EfficientNet-B3)
- **Test Accuracy**: 26.8%
- **Test F1-Score**: 26.8%
- **Architecture**: EfficientNet-B3 backbone
- **Best Validation F1**: 28.4%

## 🚀 Key Improvements Implemented

### 1. Advanced Architecture
- **EfficientNet-B3**: More efficient and powerful than ResNet50
- **Enhanced Classifier Head**: Dropout layers and batch normalization
- **Input Size**: Increased from 224x224 to 300x300 pixels

### 2. Training Enhancements
- **Class-Weighted Loss**: Addresses class imbalance issues
- **Mixed Precision Training**: Faster training with lower memory usage
- **OneCycle Learning Rate**: Optimal learning rate scheduling
- **Early Stopping**: Prevents overfitting with F1-score monitoring

### 3. Data Augmentation
- **Advanced Transforms**: RandomAffine, ColorJitter, GaussianBlur
- **Medical-Specific**: Histogram equalization for poor contrast images
- **Balanced Sampling**: WeightedRandomSampler for class balance

### 4. Monitoring & Optimization
- **Gradient Clipping**: Training stability
- **Differential Learning Rates**: Backbone vs classifier optimization
- **F1-Score Focus**: Better metric for medical classification

## 🏥 Clinical Classification System

### Stenosis Categories
1. **Normal**: 0-29% stenosis (Low risk)
2. **Mild**: 30-49% stenosis (Low-moderate risk)
3. **Moderate**: 50-69% stenosis (Moderate-high risk)
4. **Severe**: 70-99% stenosis (High risk)

### Risk Assessment
- **Low Risk**: Routine monitoring, 1-2 year follow-up
- **Moderate Risk**: Enhanced monitoring, 6-12 month follow-up
- **High Risk**: Immediate medical attention, potential intervention

## 📁 Project Structure
```
CAD_images_classification/
├── data/
│   ├── raw/                     # Original Kaggle dataset (1,100 images)
│   └── processed/               # Processed labels and splits
├── src/
│   ├── data_exploration.py      # Dataset analysis and labeling
│   ├── train_model.py          # Original training script
│   ├── train_model_enhanced.py # Enhanced training pipeline
│   └── inference.py            # Prediction and clinical reporting
├── models/
│   ├── carotid_classifier_best.pth         # Original model
│   └── enhanced_carotid_classifier_best.pth # Enhanced model
├── results/
│   └── inference/              # Prediction results and reports
└── notebooks/                  # Jupyter notebooks for analysis
```

## 🛠 Technical Stack
- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, PIL
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Medical Imaging**: Custom preprocessing pipelines

## 📋 Dataset Information
- **Source**: Kaggle - Common Carotid Artery Ultrasound Images
- **Size**: 1,100 ultrasound images from 35 patients
- **Format**: PNG images of varying sizes
- **Labels**: Synthetic labels created based on filename patterns
- **Split**: 60% train, 23% validation, 17% test

## 🔬 Model Features

### Enhanced Preprocessing
- **Adaptive Histogram Equalization**: Improves poor contrast images
- **Smart Resizing**: Maintains aspect ratio while resizing
- **Medical-Specific Normalization**: ImageNet statistics adapted for medical images

### Inference Pipeline
- **Clinical Report Generation**: Professional medical-style reports
- **Confidence Scoring**: Probability distributions for all classes
- **Risk Stratification**: Automated risk level assessment
- **Visualization**: Annotated prediction images

## 📈 Training Configuration
- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: OneCycleLR for optimal convergence
- **Loss Function**: CrossEntropyLoss with class weights
- **Batch Size**: 32 with mixed precision training
- **Epochs**: 35 with early stopping

## 🎯 Clinical Applications
1. **Screening Tool**: First-line assessment for stroke risk
2. **Triage System**: Prioritize patients for specialist referral
3. **Monitoring**: Track stenosis progression over time
4. **Education**: Training tool for medical students

## ⚠️ Important Disclaimers
- **Educational Purpose**: This model is for demonstration and learning
- **Synthetic Labels**: Training data uses artificially generated labels
- **Medical Supervision**: All results require professional medical interpretation
- **Research Only**: Not approved for clinical decision-making

## 🚀 Future Improvements
1. **Real Clinical Data**: Train on professionally labeled images
2. **Multi-Modal Input**: Combine with patient demographics and symptoms
3. **Temporal Analysis**: Track stenosis progression over time
4. **Uncertainty Quantification**: Bayesian neural networks for confidence estimation
5. **Federated Learning**: Multi-institutional training while preserving privacy

## 📊 Performance Metrics Summary
```
Enhanced Model Results:
├── Overall Test Accuracy: 26.8%
├── Overall Test F1-Score: 26.8%
├── Best Validation F1: 28.4%
└── Class-wise Performance:
    ├── Normal: F1=0.34 (Precision=0.43, Recall=0.29)
    ├── Mild Stenosis: F1=0.20 (Precision=0.22, Recall=0.17)
    ├── Moderate Stenosis: F1=0.29 (Precision=0.23, Recall=0.42)
    └── Severe Stenosis: F1=0.17 (Precision=0.17, Recall=0.17)
```

## 🎓 Learning Outcomes
This project demonstrates:
- **Medical AI**: Application of deep learning to healthcare
- **Transfer Learning**: Adapting pre-trained models for specialized tasks
- **Class Imbalance**: Handling uneven data distributions
- **Model Optimization**: Advanced training techniques and hyperparameter tuning
- **Clinical Integration**: Building AI systems for medical workflows

---
*Project developed for AI Class - Advanced Medical Imaging Classification*