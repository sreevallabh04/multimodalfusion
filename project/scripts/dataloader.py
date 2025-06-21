"""
Multi-modal dataloader for mango fruit classification.
Loads RGB images, thermal maps, and optional acoustic/texture maps.
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class MultiModalMangoDataset(Dataset):
    """
    Multi-modal dataset for mango fruit classification.
    Loads RGB images, thermal maps, and optional acoustic/texture features.
    """
    
    def __init__(self,
                 rgb_data_path: str,
                 thermal_data_path: str,
                 split: str = 'train',
                 image_size: int = 224,
                 use_acoustic: bool = False,
                 acoustic_data_path: Optional[str] = None,
                 transform_type: str = 'train'):
        """
        Args:
            rgb_data_path: Path to RGB fruit images
            thermal_data_path: Path to thermal maps
            split: Data split ('train', 'val', 'test')
            image_size: Input image size
            use_acoustic: Whether to include acoustic/texture features
            acoustic_data_path: Path to acoustic maps (if use_acoustic=True)
            transform_type: Type of transforms ('train', 'val', 'test')
        """
        self.rgb_data_path = Path(rgb_data_path)
        self.thermal_data_path = Path(thermal_data_path) / 'thermal'
        self.split = split
        self.image_size = image_size
        self.use_acoustic = use_acoustic
        self.acoustic_data_path = Path(acoustic_data_path) if acoustic_data_path else None
        
        # Define class names and create label encoder
        self.class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        self.num_classes = len(self.class_names)
        
        # Load data paths and labels
        self.data_samples = self._load_data_paths()
        
        # Define transforms
        self.rgb_transform = self._get_rgb_transforms(transform_type)
        self.thermal_transform = self._get_thermal_transforms()
        if self.use_acoustic:
            self.acoustic_transform = self._get_acoustic_transforms()
        
        print(f"✅ Loaded {len(self.data_samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _load_data_paths(self) -> List[Dict]:
        """Load paths for RGB, thermal, and acoustic data."""
        samples = []
        
        rgb_split_path = self.rgb_data_path / self.split
        thermal_split_path = self.thermal_data_path / self.split
        
        if not rgb_split_path.exists():
            raise FileNotFoundError(f"RGB data path not found: {rgb_split_path}")
        
        for class_name in self.class_names:
            rgb_class_path = rgb_split_path / class_name
            thermal_class_path = thermal_split_path / class_name
            
            if not rgb_class_path.exists():
                print(f"⚠️  RGB class path not found: {rgb_class_path}")
                continue
            
            # Get RGB image paths
            rgb_images = list(rgb_class_path.glob("*.jpg")) + \
                        list(rgb_class_path.glob("*.jpeg")) + \
                        list(rgb_class_path.glob("*.png"))
            
            for rgb_path in rgb_images:
                # Find corresponding thermal map
                thermal_path = thermal_class_path / rgb_path.name
                
                sample = {
                    'rgb_path': str(rgb_path),
                    'thermal_path': str(thermal_path) if thermal_path.exists() else None,
                    'label': class_name,
                    'class_idx': self.label_encoder.transform([class_name])[0]
                }
                
                # Add acoustic path if available
                if self.use_acoustic and self.acoustic_data_path:
                    acoustic_class_path = self.acoustic_data_path / self.split / class_name
                    acoustic_path = acoustic_class_path / rgb_path.name
                    sample['acoustic_path'] = str(acoustic_path) if acoustic_path.exists() else None
                
                samples.append(sample)
        
        return samples
    
    def _get_rgb_transforms(self, transform_type: str):
        """Get RGB image transforms."""
        if transform_type == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _get_thermal_transforms(self):
        """Get thermal map transforms."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale to [-1, 1]
            ToTensorV2()
        ])
    
    def _get_acoustic_transforms(self):
        """Get acoustic/texture map transforms."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    
    def _load_image(self, image_path: str, is_grayscale: bool = False) -> np.ndarray:
        """Load and preprocess image."""
        if not os.path.exists(image_path):
            # Return black image if file doesn't exist
            if is_grayscale:
                return np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        if is_grayscale:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _generate_acoustic_map(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Generate pseudo-acoustic map based on surface texture.
        Uses Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP).
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if len(rgb_image.shape) == 3 else rgb_image
        
        # Calculate Local Binary Patterns for texture
        def local_binary_pattern(image, P=8, R=1):
            """Simple LBP implementation."""
            height, width = image.shape
            lbp = np.zeros_like(image)
            
            for i in range(R, height - R):
                for j in range(R, width - R):
                    center = image[i, j]
                    code = 0
                    for p in range(P):
                        angle = 2 * np.pi * p / P
                        x = int(i + R * np.cos(angle))
                        y = int(j + R * np.sin(angle))
                        if 0 <= x < height and 0 <= y < width:
                            if image[x, y] >= center:
                                code |= (1 << p)
                    lbp[i, j] = code
            
            return lbp
        
        # Calculate texture features
        lbp = local_binary_pattern(gray)
        
        # Calculate gradient magnitude for firmness simulation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Combine features to simulate acoustic properties
        # High texture variation = softer fruit (higher acoustic response)
        texture_variation = cv2.GaussianBlur(lbp.astype(np.float32), (5, 5), 0)
        gradient_smooth = cv2.GaussianBlur(gradient_magnitude.astype(np.float32), (5, 5), 0)
        
        # Normalize and combine
        texture_norm = (texture_variation - texture_variation.min()) / (texture_variation.max() - texture_variation.min() + 1e-8)
        gradient_norm = (gradient_smooth - gradient_smooth.min()) / (gradient_smooth.max() - gradient_smooth.min() + 1e-8)
        
        # Acoustic map: combination of texture and gradient
        acoustic_map = 0.6 * texture_norm + 0.4 * gradient_norm
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, acoustic_map.shape)
        acoustic_map = np.clip(acoustic_map + noise, 0, 1)
        
        # Convert to uint8
        acoustic_map = (acoustic_map * 255).astype(np.uint8)
        
        return acoustic_map
    
    def _print_class_distribution(self):
        """Print class distribution for this split."""
        class_counts = {}
        for sample in self.data_samples:
            class_name = sample['label']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n📊 Class distribution for {self.split} split:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple containing (rgb_tensor, thermal_tensor, [acoustic_tensor], label)
        """
        sample = self.data_samples[idx]
        
        # Load RGB image
        rgb_image = self._load_image(sample['rgb_path'], is_grayscale=False)
        rgb_tensor = self.rgb_transform(image=rgb_image)['image']
        
        # Load thermal map
        thermal_image = self._load_image(sample['thermal_path'], is_grayscale=True) if sample['thermal_path'] else \
                       np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        thermal_tensor = self.thermal_transform(image=thermal_image)['image']
        
        # Handle label
        label = torch.tensor(sample['class_idx'], dtype=torch.long)
        
        if self.use_acoustic:
            # Load or generate acoustic map
            if self.acoustic_data_path and sample.get('acoustic_path') and os.path.exists(sample['acoustic_path']):
                acoustic_image = self._load_image(sample['acoustic_path'], is_grayscale=True)
            else:
                # Generate acoustic map from RGB image
                acoustic_image = self._generate_acoustic_map(rgb_image)
            
            acoustic_tensor = self.acoustic_transform(image=acoustic_image)['image']
            return rgb_tensor, thermal_tensor, acoustic_tensor, label
        else:
            return rgb_tensor, thermal_tensor, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets."""
        class_counts = np.zeros(self.num_classes)
        for sample in self.data_samples:
            class_counts[sample['class_idx']] += 1
        
        # Inverse frequency weighting
        total_samples = len(self.data_samples)
        weights = total_samples / (self.num_classes * class_counts)
        weights = weights / weights.sum() * self.num_classes  # Normalize
        
        return torch.FloatTensor(weights)


def create_dataloaders(rgb_data_path: str,
                      thermal_data_path: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      image_size: int = 224,
                      use_acoustic: bool = False,
                      acoustic_data_path: Optional[str] = None) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        rgb_data_path: Path to RGB fruit images
        thermal_data_path: Path to thermal maps
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Input image size
        use_acoustic: Whether to include acoustic features
        acoustic_data_path: Path to acoustic maps
        
    Returns:
        Dictionary containing 'train', 'val', 'test' dataloaders
    """
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        transform_type = 'train' if split == 'train' else 'val'
        
        datasets[split] = MultiModalMangoDataset(
            rgb_data_path=rgb_data_path,
            thermal_data_path=thermal_data_path,
            split=split,
            image_size=image_size,
            use_acoustic=use_acoustic,
            acoustic_data_path=acoustic_data_path,
            transform_type=transform_type
        )
    
    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')
        )
    
    print(f"\n✅ Created dataloaders:")
    for split, dataloader in dataloaders.items():
        print(f"  {split}: {len(dataloader)} batches, {len(dataloader.dataset)} samples")
    
    return dataloaders


def test_dataloader():
    """Test the multi-modal dataloader."""
    print("🧪 Testing Multi-Modal Dataloader...")
    
    # Test parameters
    rgb_data_path = "data/processed/fruit"
    thermal_data_path = "data/thermal"
    batch_size = 4
    use_acoustic = True
    
    try:
        # Create dataloaders
        dataloaders = create_dataloaders(
            rgb_data_path=rgb_data_path,
            thermal_data_path=thermal_data_path,
            batch_size=batch_size,
            num_workers=0,  # Use 0 for testing
            use_acoustic=use_acoustic
        )
        
        # Test a batch from train dataloader
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        if use_acoustic:
            rgb_batch, thermal_batch, acoustic_batch, labels = batch
            print(f"RGB batch shape: {rgb_batch.shape}")
            print(f"Thermal batch shape: {thermal_batch.shape}")
            print(f"Acoustic batch shape: {acoustic_batch.shape}")
            print(f"Labels shape: {labels.shape}")
        else:
            rgb_batch, thermal_batch, labels = batch
            print(f"RGB batch shape: {rgb_batch.shape}")
            print(f"Thermal batch shape: {thermal_batch.shape}")
            print(f"Labels shape: {labels.shape}")
        
        print("✅ Dataloader test passed!")
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")


if __name__ == "__main__":
    test_dataloader() 