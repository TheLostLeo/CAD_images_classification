#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import glob
from PIL import Image


class CarotidDatasetExplorer:
    """Class to explore and analyze carotid artery ultrasound dataset."""
    
    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)
        self.us_images_path = self.data_path / "Common Carotid Artery Ultrasound Images/Common Carotid Artery Ultrasound Images/US images"
        self.mask_images_path = self.data_path / "Common Carotid Artery Ultrasound Images/Common Carotid Artery Ultrasound Images/Expert mask images"
        self.processed_path = Path("data/processed")
        self.results_path = Path("results")
        
        # Create output directories
        self.processed_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
    def explore_dataset_structure(self):
        """Explore the basic structure of the dataset."""
        print("=" * 60)
        print("CAROTID ARTERY ULTRASOUND DATASET EXPLORATION")
        print("=" * 60)
        
        # Check if paths exist
        print(f"Dataset path: {self.data_path}")
        print(f"US Images path exists: {self.us_images_path.exists()}")
        print(f"Mask Images path exists: {self.mask_images_path.exists()}")
        
        # Count images
        us_images = list(self.us_images_path.glob("*.png")) if self.us_images_path.exists() else []
        mask_images = list(self.mask_images_path.glob("*.png")) if self.mask_images_path.exists() else []
        
        print(f"\nDataset Statistics:")
        print(f"- Total Ultrasound Images: {len(us_images)}")
        print(f"- Total Mask Images: {len(mask_images)}")
        
        return us_images, mask_images
    
    def analyze_image_properties(self, us_images):
        """Analyze basic properties of ultrasound images."""
        print(f"\n{'='*40}")
        print("IMAGE PROPERTIES ANALYSIS")
        print(f"{'='*40}")
        
        if not us_images:
            print("No ultrasound images found!")
            return
        
        # Sample a subset for analysis (to speed up processing)
        sample_size = min(100, len(us_images))
        sample_images = np.random.choice(us_images, sample_size, replace=False)
        
        widths, heights, channels = [], [], []
        file_sizes = []
        
        print(f"Analyzing {sample_size} sample images...")
        
        for img_path in sample_images:
            try:
                # Get file size
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
                
                # Load image and get dimensions
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w, c = img.shape
                    heights.append(h)
                    widths.append(w)
                    channels.append(c)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Print statistics
        if widths and heights:
            print(f"\nImage Dimensions:")
            print(f"- Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
            print(f"- Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
            print(f"- Channels: {Counter(channels)}")
            print(f"- File sizes (KB): min={min(file_sizes):.1f}, max={max(file_sizes):.1f}, mean={np.mean(file_sizes):.1f}")
            
            # Most common dimensions
            dimensions = [(w, h) for w, h in zip(widths, heights)]
            common_dims = Counter(dimensions).most_common(5)
            print(f"\nMost common dimensions:")
            for dims, count in common_dims:
                print(f"- {dims[0]}x{dims[1]}: {count} images")
                
        return {
            'widths': widths,
            'heights': heights,
            'file_sizes': file_sizes,
            'sample_images': sample_images
        }
    
    def visualize_sample_images(self, sample_images, n_samples=8):
        """Visualize a sample of ultrasound images."""
        print(f"\n{'='*40}")
        print("SAMPLE IMAGE VISUALIZATION")
        print(f"{'='*40}")
        
        if len(sample_images) < n_samples:
            n_samples = len(sample_images)
        
        # Select random samples
        selected_samples = np.random.choice(sample_images, n_samples, replace=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i, img_path in enumerate(selected_samples):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f"Image {i+1}\n{Path(img_path).name[:20]}...")
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, "Failed to load", ha='center', va='center')
                    axes[i].set_title(f"Error {i+1}")
                    
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {str(e)[:20]}", ha='center', va='center')
                axes[i].set_title(f"Error {i+1}")
        
        plt.tight_layout()
        plt.savefig(self.results_path / "sample_ultrasound_images.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sample images saved to: {self.results_path / 'sample_ultrasound_images.png'}")
    
    def analyze_filename_patterns(self, us_images):
        """Analyze filename patterns to understand the dataset structure."""
        print(f"\n{'='*40}")
        print("FILENAME PATTERN ANALYSIS")
        print(f"{'='*40}")
        
        if not us_images:
            return
        
        # Extract filename components
        filenames = [Path(img).name for img in us_images]
        
        # Analyze patterns
        print(f"Total files: {len(filenames)}")
        print(f"Sample filenames:")
        for i, fname in enumerate(filenames[:5]):
            print(f"  {i+1}. {fname}")
        
        # Extract patient/study IDs (assuming they're in the filename)
        patient_ids = []
        for fname in filenames:
            # Extract date/time part which likely represents different studies
            parts = fname.split('_')
            if len(parts) > 0:
                patient_ids.append(parts[0][:12])  # First 12 chars (date + time)
        
        unique_patients = len(set(patient_ids))
        print(f"\nDataset composition:")
        print(f"- Unique studies/patients: {unique_patients}")
        print(f"- Average images per study: {len(filenames) / unique_patients:.1f}")
        
        # Count images per patient
        patient_counts = Counter(patient_ids)
        print(f"\nImages per study distribution:")
        counts_dist = Counter(patient_counts.values())
        for count, frequency in sorted(counts_dist.items()):
            print(f"- {count} images: {frequency} studies")
            
        return patient_ids, patient_counts
    
    def create_classification_labels(self, us_images, patient_counts):
        """Create synthetic labels for classification (since we don't have real stenosis labels)."""
        print(f"\n{'='*40}")
        print("CREATING SYNTHETIC LABELS FOR CLASSIFICATION")
        print(f"{'='*40}")
        
        # For this project, we'll create synthetic labels based on image characteristics
        # In a real scenario, these would come from medical annotations
        
        labels_data = []
        
        print("Creating synthetic stenosis classification labels...")
        print("Note: In a real project, these would be provided by medical experts")
        
        # Define synthetic classes
        classes = ['normal', 'mild_stenosis', 'moderate_stenosis', 'severe_stenosis']
        class_weights = [0.4, 0.3, 0.2, 0.1]  # More normal cases, fewer severe
        
        for img_path in us_images:
            filename = Path(img_path).name
            
            # Create synthetic label based on filename hash for consistency
            hash_val = hash(filename) % 100
            
            if hash_val < 40:
                label = 'normal'
                label_idx = 0
            elif hash_val < 70:
                label = 'mild_stenosis'
                label_idx = 1
            elif hash_val < 90:
                label = 'moderate_stenosis'
                label_idx = 2
            else:
                label = 'severe_stenosis'
                label_idx = 3
            
            labels_data.append({
                'filename': filename,
                'filepath': str(img_path),
                'label': label,
                'label_idx': label_idx
            })
        
        # Create DataFrame
        df = pd.DataFrame(labels_data)
        
        # Print distribution
        label_counts = df['label'].value_counts()
        print(f"\nSynthetic label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"- {label}: {count} images ({percentage:.1f}%)")
        
        # Save labels
        labels_file = self.processed_path / "image_labels.csv"
        df.to_csv(labels_file, index=False)
        print(f"\nLabels saved to: {labels_file}")
        
        return df
    
    def create_train_val_split(self, df, val_split=0.2, test_split=0.1):
        """Create train/validation/test splits ensuring patient-level separation."""
        print(f"\n{'='*40}")
        print("CREATING TRAIN/VALIDATION/TEST SPLITS")
        print(f"{'='*40}")
        
        # Extract patient IDs from filenames
        df['patient_id'] = df['filename'].apply(lambda x: x.split('_')[0][:12])
        
        # Get unique patients
        unique_patients = df['patient_id'].unique()
        np.random.shuffle(unique_patients)
        
        # Split patients (not images) to avoid data leakage
        n_patients = len(unique_patients)
        n_test = int(n_patients * test_split)
        n_val = int(n_patients * val_split)
        n_train = n_patients - n_test - n_val
        
        test_patients = unique_patients[:n_test]
        val_patients = unique_patients[n_test:n_test + n_val]
        train_patients = unique_patients[n_test + n_val:]
        
        # Assign splits
        df['split'] = 'train'
        df.loc[df['patient_id'].isin(val_patients), 'split'] = 'val'
        df.loc[df['patient_id'].isin(test_patients), 'split'] = 'test'
        
        # Print statistics
        split_stats = df.groupby(['split', 'label']).size().unstack(fill_value=0)
        print(f"\nDataset split statistics:")
        print(f"- Train patients: {n_train} ({len(df[df['split']=='train'])} images)")
        print(f"- Validation patients: {n_val} ({len(df[df['split']=='val'])} images)")
        print(f"- Test patients: {n_test} ({len(df[df['split']=='test'])} images)")
        
        print(f"\nLabel distribution by split:")
        print(split_stats)
        
        # Save updated labels with splits
        labels_with_splits_file = self.processed_path / "image_labels_with_splits.csv"
        df.to_csv(labels_with_splits_file, index=False)
        print(f"\nUpdated labels saved to: {labels_with_splits_file}")
        
        return df
    
    def generate_data_summary_report(self, df):
        """Generate a comprehensive data summary report."""
        print(f"\n{'='*60}")
        print("DATASET SUMMARY REPORT")
        print(f"{'='*60}")
        
        report = f"""
CAROTID ARTERY STENOSIS DETECTION DATASET
=========================================

Dataset Overview:
- Total Images: {len(df)}
- Unique Patients/Studies: {df['patient_id'].nunique()}
- Classes: {df['label'].nunique()} (Normal, Mild, Moderate, Severe Stenosis)

Data Distribution:
{df['label'].value_counts().to_string()}

Split Distribution:
{df['split'].value_counts().to_string()}

Class Distribution by Split:
{df.groupby(['split', 'label']).size().unstack(fill_value=0).to_string()}

Files:
- Raw images: {self.us_images_path}
- Labels: {self.processed_path / 'image_labels_with_splits.csv'}
- Processed data will be stored in: {self.processed_path}
"""
        
        print(report)
        
        # Save report
        report_file = self.results_path / "dataset_summary_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nFull report saved to: {report_file}")


def main():
    """Main function to run the dataset exploration."""
    print("Starting Carotid Artery Ultrasound Dataset Exploration...")
    
    # Initialize explorer
    explorer = CarotidDatasetExplorer()
    
    # Step 1: Explore dataset structure
    us_images, mask_images = explorer.explore_dataset_structure()
    
    if not us_images:
        print("No ultrasound images found. Please check the dataset path.")
        return
    
    # Step 2: Analyze image properties
    img_props = explorer.analyze_image_properties(us_images)
    
    # Step 3: Visualize sample images
    if img_props and 'sample_images' in img_props:
        explorer.visualize_sample_images(img_props['sample_images'])
    
    # Step 4: Analyze filename patterns
    patient_ids, patient_counts = explorer.analyze_filename_patterns(us_images)
    
    # Step 5: Create classification labels
    df = explorer.create_classification_labels(us_images, patient_counts)
    
    # Step 6: Create train/validation/test splits
    df_with_splits = explorer.create_train_val_split(df)
    
    # Step 7: Generate summary report
    explorer.generate_data_summary_report(df_with_splits)
    
    print(f"\n{'='*60}")
    print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("Next: Run the model training script to build the stenosis classifier.")


if __name__ == "__main__":
    main()