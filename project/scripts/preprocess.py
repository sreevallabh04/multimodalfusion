"""
Data preprocessing script for mango fruit and leaf datasets.
Splits data into train/val/test sets (70/15/15) and organizes for training.
"""

import os
import shutil
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import numpy as np

class DataPreprocessor:
    def __init__(self, base_path="data", seed=42):
        self.base_path = Path(base_path)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define paths
        self.fruit_raw_path = self.base_path / "fruit" / "SenMangoFruitDDS_bgremoved"
        self.leaf_raw_path = self.base_path / "leaf"
        
        # Output paths
        self.fruit_processed_path = self.base_path / "processed" / "fruit"
        self.leaf_processed_path = self.base_path / "processed" / "leaf"
        
        # Classes
        self.fruit_classes = ["Healthy", "Anthracnose", "Alternaria", "Black Mould Rot", "Stem and Rot"]
        self.leaf_classes = ["Healthy", "Anthracnose", "Bacterial Canker", "Cutting Weevil", 
                           "Die Back", "Gall Midge", "Powdery Mildew", "Sooty Mould"]
    
    def create_directory_structure(self):
        """Create directory structure for processed data."""
        for split in ['train', 'val', 'test']:
            # Fruit directories
            for cls in self.fruit_classes:
                (self.fruit_processed_path / split / cls).mkdir(parents=True, exist_ok=True)
            
            # Leaf directories  
            for cls in self.leaf_classes:
                (self.leaf_processed_path / split / cls).mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created successfully!")
    
    def get_image_paths(self, dataset_path, classes):
        """Get all image paths and labels."""
        image_paths = []
        labels = []
        
        for cls in classes:
            cls_path = dataset_path / cls
            if cls_path.exists():
                images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
                print(f"Found {len(images)} images in {cls_path}")
                image_paths.extend(images)
                labels.extend([cls] * len(images))
            else:
                print(f"Class path does not exist: {cls_path}")
        
        return image_paths, labels
    
    def resize_and_save_image(self, src_path, dst_path, target_size=(224, 224)):
        """Resize image and save to destination."""
        try:
            # Read image
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"⚠️  Failed to read image: {src_path}")
                return False
            
            # Resize image
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Create destination directory if it doesn't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            cv2.imwrite(str(dst_path), img_resized)
            return True
        except Exception as e:
            print(f"❌ Error processing {src_path}: {e}")
            return False
    
    def split_and_process_dataset(self, dataset_name="fruit"):
        """Split dataset into train/val/test and process images."""
        if dataset_name == "fruit":
            raw_path = self.fruit_raw_path
            processed_path = self.fruit_processed_path
            classes = self.fruit_classes
        else:
            raw_path = self.leaf_raw_path
            processed_path = self.leaf_processed_path
            classes = self.leaf_classes
        
        print(f"🚀 Processing {dataset_name} dataset...")
        
        # Get all image paths and labels
        image_paths, labels = self.get_image_paths(raw_path, classes)
        
        if not image_paths:
            print(f"❌ No images found in {raw_path}")
            return
        
        print(f"📊 Found {len(image_paths)} images across {len(set(labels))} classes")
        
        # Create DataFrame for easier handling
        df = pd.DataFrame({'path': image_paths, 'label': labels})
        
        # Print class distribution
        print("\n📈 Class distribution:")
        print(df['label'].value_counts())
        
        # Split data: 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=self.seed, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=self.seed, stratify=temp_df['label'])
        
        print(f"\n📊 Data splits:")
        print(f"Train: {len(train_df)} images")
        print(f"Val: {len(val_df)} images") 
        print(f"Test: {len(test_df)} images")
        
        # Process and save images
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f"\n🔄 Processing {split_name} set...")
            
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
                src_path = row['path']
                label = row['label']
                
                # Create destination path
                filename = f"{src_path.stem}_{idx}{src_path.suffix}"
                dst_path = processed_path / split_name / label / filename
                
                # Resize and save image
                self.resize_and_save_image(src_path, dst_path)
        
        # Save metadata
        metadata = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for split_name, split_df in metadata.items():
            metadata_path = processed_path / f"{split_name}_metadata.csv"
            split_df.to_csv(metadata_path, index=False)
        
        print(f"✅ {dataset_name.capitalize()} dataset processing completed!")
    
    def generate_summary_report(self):
        """Generate a summary report of the processed datasets."""
        print("\n" + "="*60)
        print("📋 DATASET SUMMARY REPORT")
        print("="*60)
        
        for dataset_name in ["fruit", "leaf"]:
            if dataset_name == "fruit":
                processed_path = self.fruit_processed_path
                classes = self.fruit_classes
            else:
                processed_path = self.leaf_processed_path  
                classes = self.leaf_classes
            
            print(f"\n🥭 {dataset_name.upper()} DATASET:")
            print(f"Classes: {len(classes)}")
            print(f"Class names: {', '.join(classes)}")
            
            total_images = 0
            for split in ['train', 'val', 'test']:
                split_count = 0
                for cls in classes:
                    cls_path = processed_path / split / cls
                    if cls_path.exists():
                        cls_count = len(list(cls_path.glob("*.jpg"))) + len(list(cls_path.glob("*.jpeg"))) + len(list(cls_path.glob("*.png")))
                        split_count += cls_count
                
                print(f"  {split.capitalize()}: {split_count} images")
                total_images += split_count
            
            print(f"  Total: {total_images} images")
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        print("🚀 Starting Multi-Modal Mango Dataset Preprocessing")
        print("="*60)
        
        # Create directory structure
        self.create_directory_structure()
        
        # Process both datasets
        self.split_and_process_dataset("fruit")
        self.split_and_process_dataset("leaf")
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n✅ Preprocessing completed successfully!")
        print("📁 Processed data available in:")
        print(f"   - Fruit: {self.fruit_processed_path}")
        print(f"   - Leaf: {self.leaf_processed_path}")


def main():
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing
    preprocessor.run_preprocessing()


if __name__ == "__main__":
    main() 