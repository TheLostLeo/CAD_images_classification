#!/usr/bin/env python3
"""
Data Splitting Script for Binary Classification
Takes preprocessed binary data from binary_classified/ directory and splits it into train/val/test
"""

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


def split_preprocessed_data(
    source_dir="data/binary_classified",
    output_dir="data/preprocessed",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):
    """
    Split preprocessed binary data into train/val/test directories
    
    Args:
        source_dir: Directory containing preprocessed 0/ and 1/ subdirectories
        output_dir: Output directory for split data
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)  
        test_ratio: Fraction for testing (default: 0.15)
        random_state: Random seed for reproducibility
    """
    
    # Verify ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    print("BINARY DATASET SPLITTING FROM PREPROCESSED DATA")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    print("=" * 60)
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Check source directories
    normal_dir = os.path.join(source_dir, "0")
    disease_dir = os.path.join(source_dir, "1")
    
    if not os.path.exists(normal_dir) or not os.path.exists(disease_dir):
        raise FileNotFoundError(f"Preprocessed directories not found: {normal_dir} or {disease_dir}")
    
    # Get image lists
    normal_images = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    disease_images = [f for f in os.listdir(disease_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(normal_images)} normal images in preprocessed data")
    print(f"Found {len(disease_images)} disease images in preprocessed data")
    print(f"Total preprocessed images: {len(normal_images) + len(disease_images)}")
    
    # Create output directory structure
    splits = ['train', 'val', 'test']
    classes = ['0', '1']  # normal, disease
    
    # Remove existing output directory if it exists
    if os.path.exists(output_dir):
        print(f"\nRemoving existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    for split in splits:
        for class_dir in classes:
            os.makedirs(os.path.join(output_dir, split, class_dir), exist_ok=True)
    
    print(f"\nCreated directory structure:")
    for split in splits:
        print(f"  {output_dir}/{split}/0/ (normal)")
        print(f"  {output_dir}/{split}/1/ (disease)")
    
    # Split each class separately to maintain class balance
    def split_class_images(images, class_name, source_class_dir, target_class):
        """Split images for a single class"""
        
        # First split: train vs (val + test)
        train_images, temp_images = train_test_split(
            images, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Second split: val vs test from the temp set
        val_images, test_images = train_test_split(
            temp_images,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=random_state
        )
        
        print(f"\n{class_name} class split:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(val_images)} images") 
        print(f"  Test: {len(test_images)} images")
        
        # Copy files to respective directories
        split_data = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        copy_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for split, image_list in split_data.items():
            dest_dir = os.path.join(output_dir, split, target_class)
            
            for image_name in tqdm(image_list, desc=f"Copying {class_name} {split}"):
                source_path = os.path.join(source_class_dir, image_name)
                dest_path = os.path.join(dest_dir, image_name)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    copy_counts[split] += 1
                except Exception as e:
                    print(f"Error copying {image_name}: {e}")
        
        return copy_counts
    
    # Split normal images (class 0)
    normal_counts = split_class_images(normal_images, "Normal", normal_dir, "0")
    
    # Split disease images (class 1)  
    disease_counts = split_class_images(disease_images, "Disease", disease_dir, "1")
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("DATASET SPLITTING RESULTS")
    print("=" * 60)
    
    total_counts = {}
    for split in splits:
        total_normal = normal_counts[split]
        total_disease = disease_counts[split]
        total_split = total_normal + total_disease
        total_counts[split] = total_split
        
        print(f"\n{split.upper()} SET:")
        print(f"  Normal (0): {total_normal} images")
        print(f"  Disease (1): {total_disease} images")
        print(f"  Total: {total_split} images")
        if total_disease > 0:
            print(f"  Class ratio (Normal/Disease): {total_normal/total_disease:.3f}")
    
    # Overall summary
    total_images = sum(total_counts.values())
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total images processed: {total_images}")
    print(f"  Train: {total_counts['train']} ({total_counts['train']/total_images:.1%})")
    print(f"  Val: {total_counts['val']} ({total_counts['val']/total_images:.1%})")
    print(f"  Test: {total_counts['test']} ({total_counts['test']/total_images:.1%})")
    
    # Verify directory contents
    print(f"\nVERIFICATION:")
    for split in splits:
        for class_dir in classes:
            path = os.path.join(output_dir, split, class_dir)
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {split}/{class_dir}: {count} files")
    
    # Save summary
    summary_path = os.path.join(output_dir, "split_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Binary Dataset Split Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Source directory: {source_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}\n")
        f.write(f"Random state: {random_state}\n\n")
        
        f.write("Split distribution:\n")
        for split in splits:
            f.write(f"  {split}:\n")
            f.write(f"    Normal (0): {normal_counts[split]} images\n")
            f.write(f"    Disease (1): {disease_counts[split]} images\n")
            f.write(f"    Total: {total_counts[split]} images\n\n")
        
        f.write(f"Total images: {total_images}\n")
        f.write("Split completed successfully!\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 60)
    print("PREPROCESSING AND SPLITTING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return output_dir


if __name__ == "__main__":
    try:
        output_path = split_preprocessed_data()
        print(f"\nDataset ready for training at: {output_path}")
        print("\nYou can now use this split dataset for training your binary classification models.")
        
    except Exception as e:
        print(f"Error during dataset splitting: {e}")
        import traceback
        traceback.print_exc()