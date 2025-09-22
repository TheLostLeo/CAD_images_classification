#!/usr/bin/env python3
"""
Image Preprocessing Script for Binary Classification
Organizes images into two directories: 0 (normal) and 1 (ill/disease)
Based on existing CSV labels without creating new CSV files
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def preprocess_images_binary():
    """
    Preprocess images by organizing them into binary classification structure:
    - 0/ directory: Normal images
    - 1/ directory: Disease images (mild, moderate, severe stenosis)
    """
    
    print("BINARY IMAGE PREPROCESSING")
    print("=" * 50)
    print("Organizing images into binary classification structure:")
    print("• 0/ → Normal images")
    print("• 1/ → Disease images (mild + moderate + severe stenosis)")
    print("=" * 50)
    
    # Define paths
    csv_file = "data/processed/image_labels.csv"
    source_dir = Path("data/raw/Common Carotid Artery Ultrasound Images/Common Carotid Artery Ultrasound Images/US images")
    output_dir = Path("data/binary_classified")
    
    # Create output directories
    normal_dir = output_dir / "0"
    disease_dir = output_dir / "1"
    
    # Create directories if they don't exist
    normal_dir.mkdir(parents=True, exist_ok=True)
    disease_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Normal images -> {normal_dir}")
    print(f"Disease images -> {disease_dir}")
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return False
    
    # Check if CSV file exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found: {csv_file}")
        return False
    
    # Read CSV file
    print(f"\nReading labels from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Total images in CSV: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} images")
    
    # Define label mapping for binary classification
    label_mapping = {
        'normal': 0,
        'mild_stenosis': 1,
        'moderate_stenosis': 1,
        'severe_stenosis': 1
    }
    
    # Add binary label column
    df['binary_label'] = df['label'].map(label_mapping)
    
    # Check binary distribution
    binary_counts = df['binary_label'].value_counts()
    print(f"\nBinary classification distribution:")
    print(f"  Normal (0): {binary_counts.get(0, 0)} images")
    print(f"  Disease (1): {binary_counts.get(1, 0)} images")
    
    # Process images
    print(f"\nProcessing images...")
    
    copied_count = {'normal': 0, 'disease': 0}
    missing_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        filename = row['filename']
        binary_label = row['binary_label']
        
        # Source file path
        source_file = source_dir / filename
        
        # Determine destination directory
        if binary_label == 0:
            dest_dir = normal_dir
            category = 'normal'
        else:
            dest_dir = disease_dir
            category = 'disease'
        
        # Destination file path
        dest_file = dest_dir / filename
        
        # Check if source file exists
        if source_file.exists():
            try:
                # Copy file to destination
                shutil.copy2(source_file, dest_file)
                copied_count[category] += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        else:
            missing_files.append(filename)
    
    # Print results
    print(f"\n" + "=" * 50)
    print("PREPROCESSING RESULTS")
    print("=" * 50)
    print(f"Successfully copied:")
    print(f"   Normal images: {copied_count['normal']} -> {normal_dir}")
    print(f"   Disease images: {copied_count['disease']} -> {disease_dir}")
    print(f"   Total copied: {copied_count['normal'] + copied_count['disease']}")
    
    if missing_files:
        print(f"\nMissing files: {len(missing_files)}")
        if len(missing_files) <= 10:
            for missing_file in missing_files:
                print(f"   - {missing_file}")
        else:
            print(f"   First 10 missing files:")
            for missing_file in missing_files[:10]:
                print(f"   - {missing_file}")
            print(f"   ... and {len(missing_files) - 10} more")
    
    # Create summary file
    summary = {
        'total_images_processed': len(df),
        'normal_images': copied_count['normal'],
        'disease_images': copied_count['disease'],
        'total_copied': copied_count['normal'] + copied_count['disease'],
        'missing_files': len(missing_files),
        'normal_directory': str(normal_dir),
        'disease_directory': str(disease_dir),
        'label_mapping': label_mapping
    }
    
    # Save summary to file
    summary_file = output_dir / "preprocessing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Binary Image Preprocessing Summary\n")
        f.write("=" * 40 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        
        if missing_files:
            f.write(f"\nMissing files:\n")
            for missing_file in missing_files:
                f.write(f"  - {missing_file}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Verify directory structure
    print(f"\nDirectory structure created:")
    print(f"   {output_dir}/")
    print(f"   ├── 0/ ({len(list(normal_dir.glob('*')))} normal images)")
    print(f"   ├── 1/ ({len(list(disease_dir.glob('*')))} disease images)")
    print(f"   └── preprocessing_summary.txt")
    
    return True


def verify_preprocessing():
    """Verify the preprocessing was successful"""
    
    print(f"\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    
    output_dir = Path("data/binary_classified")
    normal_dir = output_dir / "0"
    disease_dir = output_dir / "1"
    
    if not output_dir.exists():
        print("Binary classified directory does not exist")
        return False
    
    # Count files in each directory
    normal_count = len(list(normal_dir.glob("*.png"))) if normal_dir.exists() else 0
    disease_count = len(list(disease_dir.glob("*.png"))) if disease_dir.exists() else 0
    
    print(f"Directory verification:")
    print(f"   Normal images (0/): {normal_count} files")
    print(f"   Disease images (1/): {disease_count} files")
    print(f"   Total: {normal_count + disease_count} files")
    
    # Check some sample files
    if normal_count > 0:
        sample_normal = list(normal_dir.glob("*.png"))[:3]
        print(f"\nSample normal images:")
        for img in sample_normal:
            print(f"   - {img.name}")
    
    if disease_count > 0:
        sample_disease = list(disease_dir.glob("*.png"))[:3]
        print(f"\nSample disease images:")
        for img in sample_disease:
            print(f"   - {img.name}")
    
    # Calculate class balance
    if normal_count + disease_count > 0:
        balance_ratio = min(normal_count, disease_count) / max(normal_count, disease_count)
        print(f"\nClass balance ratio: {balance_ratio:.3f}")
        
        if balance_ratio < 0.5:
            print(" Dataset is imbalanced - consider balancing techniques")
        else:
            print("Dataset is reasonably balanced")
    
    return True


def main():
    """Main function to run preprocessing"""
    
    print("Binary Image Preprocessing for Carotid Artery Stenosis Classification")
    print("Converting 4-class problem to binary classification")
    print("Normal vs Disease (Mild + Moderate + Severe)")
    print("=" * 80)
    
    # Run preprocessing
    success = preprocess_images_binary()
    
    if success:
        # Verify results
        verify_preprocessing()
        
        print(f"\n" + "=" * 80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print(f"\n" + "=" * 80)
        print("PREPROCESSING FAILED!")
        print("Please check the error messages above and fix the issues.")
        print("=" * 80)


if __name__ == "__main__":
    main()