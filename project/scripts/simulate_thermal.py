"""
Thermal simulation script that generates pseudo-thermal maps for fruit images
using a lesion detector trained on leaf images.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.lesion_detector import LesionDetector, create_lesion_detector


class FruitDataset(Dataset):
    """
    Dataset for loading fruit images for thermal simulation.
    Args:
        image_paths (list): List of image file paths.
        transform (callable, optional): Transform to apply to images.
    """
    def __init__(self, image_paths: list, transform: callable = None) -> None:
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self) -> int:
        return len(self.image_paths)
    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(image_path)


class ThermalSimulator:
    """
    Simulates thermal maps for fruit images using a pre-trained lesion detector.
    Args:
        lesion_model_path (str): Path to the lesion detector model.
        device (torch.device, optional): Device to use.
        input_size (int): Input image size.
    """
    def __init__(self, lesion_model_path: str, device: torch.device = None, input_size: int = 224) -> None:
        self.logger = logging.getLogger("ThermalSimulator")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.lesion_model = self._load_lesion_model(lesion_model_path)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.logger.info(f"\u2705 Thermal simulator initialized on device: {self.device}")
    def _load_lesion_model(self, model_path: str) -> LesionDetector:
        """
        Load the pre-trained lesion detector model.
        Args:
            model_path (str): Path to the model checkpoint.
        Returns:
            LesionDetector: Loaded lesion detector model.
        """
        if not os.path.exists(model_path):
            self.logger.warning(f"\u26a0\ufe0f  Lesion model not found at {model_path}")
            self.logger.info("Creating a new model for demonstration...")
            model = create_lesion_detector(num_classes=8)
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = create_lesion_detector(num_classes=8)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"\u2705 Loaded lesion detector from {model_path}")
        model.to(self.device)
        model.eval()
        return model
    def generate_thermal_map(self, fruit_image: torch.Tensor) -> np.ndarray:
        """
        Generate pseudo-thermal map for a fruit image.
        Args:
            fruit_image (torch.Tensor): Normalized tensor of shape (3, 224, 224)
        Returns:
            np.ndarray: Thermal map as numpy array of shape (224, 224)
        """
        with torch.no_grad():
            if fruit_image.dim() == 3:
                fruit_image = fruit_image.unsqueeze(0)
            fruit_image = fruit_image.to(self.device)
            lesion_map = self.lesion_model.get_lesion_probability(fruit_image)
            lesion_map = lesion_map.squeeze().cpu().numpy()
            thermal_map = self._simulate_heat_distribution(lesion_map)
            return thermal_map
    def _simulate_heat_distribution(self, lesion_map: np.ndarray) -> np.ndarray:
        """
        Simulate realistic heat distribution based on lesion probability.
        Args:
            lesion_map (np.ndarray): Lesion probability map of shape (224, 224)
        Returns:
            np.ndarray: Thermal map simulating heat distribution
        """
        base_temp = 0.3
        disease_temp_increase = lesion_map * 0.7
        thermal_map = base_temp + disease_temp_increase
        thermal_map = cv2.GaussianBlur(thermal_map, (15, 15), 3.0)
        noise = np.random.normal(0, 0.05, thermal_map.shape)
        thermal_map = thermal_map + noise
        thermal_map = np.clip(thermal_map, 0, 1)
        return thermal_map
    def save_thermal_map(self, thermal_map: np.ndarray, output_path: str) -> None:
        """
        Save thermal map as grayscale image.
        Args:
            thermal_map (np.ndarray): Thermal map.
            output_path (str): Path to save the image.
        """
        thermal_uint8 = (thermal_map * 255).astype(np.uint8)
        cv2.imwrite(output_path, thermal_uint8)
    def visualize_thermal_overlay(self, original_image: np.ndarray, thermal_map: np.ndarray, output_path: str = None, alpha: float = 0.6) -> np.ndarray:
        """
        Create a visualization overlaying thermal map on original image.
        Args:
            original_image (np.ndarray): Original RGB image.
            thermal_map (np.ndarray): Thermal map.
            output_path (str, optional): Path to save the visualization.
            alpha (float): Transparency of thermal overlay.
        Returns:
            np.ndarray: Overlay image.
        """
        if original_image.shape[:2] != thermal_map.shape:
            thermal_map = cv2.resize(thermal_map, (original_image.shape[1], original_image.shape[0]))
        thermal_colored = cm.jet(thermal_map)[:, :, :3]
        thermal_colored = (thermal_colored * 255).astype(np.uint8)
        overlay = cv2.addWeighted(original_image, 1-alpha, thermal_colored, alpha, 0)
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return overlay
    def process_fruit_dataset(self, fruit_data_path: str, output_path: str, batch_size: int = 16, save_visualizations: bool = True) -> None:
        """
        Process entire fruit dataset to generate thermal maps.
        Args:
            fruit_data_path (str): Path to processed fruit dataset.
            output_path (str): Path to save thermal maps.
            batch_size (int): Batch size for processing.
            save_visualizations (bool): Whether to save overlay visualizations.
        """
        fruit_data_path = Path(fruit_data_path)
        output_path = Path(output_path)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for class_name in ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']:
                thermal_dir = output_path / 'thermal' / split / class_name
                thermal_dir.mkdir(parents=True, exist_ok=True)
                
                if save_visualizations:
                    viz_dir = output_path / 'visualizations' / split / class_name
                    viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            self.logger.info(f"\n🔥 Processing {split} set for thermal simulation...")
            
            # Get all image paths for this split
            split_path = fruit_data_path / split
            if not split_path.exists():
                self.logger.warning(f"⚠️  Split path {split_path} does not exist, skipping...")
                continue
            
            all_image_paths = []
            for class_name in ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']:
                class_path = split_path / class_name
                if class_path.exists():
                    image_paths = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
                    all_image_paths.extend(image_paths)
            
            if not all_image_paths:
                self.logger.warning(f"⚠️  No images found in {split_path}")
                continue
            
            # Create dataset and dataloader
            dataset = FruitDataset(all_image_paths, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            # Process images
            for batch_images, batch_paths in tqdm(dataloader, desc=f"Generating thermal maps for {split}"):
                batch_images = batch_images.to(self.device)
                
                for i, (image_tensor, image_path) in enumerate(zip(batch_images, batch_paths)):
                    # Generate thermal map
                    thermal_map = self.generate_thermal_map(image_tensor)
                    
                    # Determine output paths
                    image_path = Path(image_path)
                    class_name = image_path.parent.name
                    
                    thermal_output_path = output_path / 'thermal' / split / class_name / image_path.name
                    
                    # Save thermal map
                    self.save_thermal_map(thermal_map, str(thermal_output_path))
                    
                    # Save visualization if requested
                    if save_visualizations:
                        # Load original image for visualization
                        original_image = cv2.imread(str(image_path))
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        original_image = cv2.resize(original_image, (224, 224))
                        
                        viz_output_path = output_path / 'visualizations' / split / class_name / f"overlay_{image_path.name}"
                        self.visualize_thermal_overlay(original_image, thermal_map, str(viz_output_path))
        
        self.logger.info(f"\n✅ Thermal simulation completed!")
        self.logger.info(f"📁 Thermal maps saved to: {output_path / 'thermal'}")
        if save_visualizations:
            self.logger.info(f"📁 Visualizations saved to: {output_path / 'visualizations'}")


def main():
    """Main function to run thermal simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate pseudo-thermal maps for fruit images')
    parser.add_argument('--lesion_model', type=str, default='models/lesion_detector_best.pth',
                       help='Path to trained lesion detector model')
    parser.add_argument('--fruit_data', type=str, default='data/processed/fruit',
                       help='Path to processed fruit dataset')
    parser.add_argument('--output', type=str, default='data/thermal',
                       help='Output path for thermal maps')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip saving visualization overlays')
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    # Initialize thermal simulator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    thermal_simulator = ThermalSimulator(
        lesion_model_path=args.lesion_model,
        device=device
    )
    
    # Process fruit dataset
    thermal_simulator.process_fruit_dataset(
        fruit_data_path=args.fruit_data,
        output_path=args.output,
        batch_size=args.batch_size,
        save_visualizations=not args.no_viz
    )


if __name__ == "__main__":
    main() 