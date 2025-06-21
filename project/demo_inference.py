#!/usr/bin/env python3
"""
Demonstration script for multi-modal mango disease classification inference.
Shows how to use the trained RGB and fusion models on new images.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.rgb_branch import create_rgb_branch
from models.fusion_model import create_fusion_model
from scripts.simulate_thermal import ThermalSimulator


class MangoClassificationDemo:
    """Demo class for mango fruit disease classification."""
    
    def __init__(self, 
                 rgb_model_path: str,
                 fusion_model_path: str = None,
                 device: str = 'auto'):
        """
        Initialize the demo with trained models.
        
        Args:
            rgb_model_path: Path to trained RGB model
            fusion_model_path: Path to trained fusion model (optional)
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Class names
        self.class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
        
        # Load RGB model
        self.rgb_model = self._load_model(rgb_model_path, 'rgb')
        print(f"✅ Loaded RGB model from {rgb_model_path}")
        
        # Load fusion model if provided
        self.fusion_model = None
        self.thermal_simulator = None
        if fusion_model_path:
            self.fusion_model = self._load_model(fusion_model_path, 'fusion')
            print(f"✅ Loaded fusion model from {fusion_model_path}")
            
            # Initialize thermal simulator
            self.thermal_simulator = ThermalSimulator(
                lesion_model_path='dummy_path.pth',  # Uses demo model
                device=self.device
            )
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.thermal_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _load_model(self, model_path: str, model_type: str):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if model_type == 'rgb':
            model = create_rgb_branch(
                num_classes=5,
                backbone='resnet18',
                feature_dim=512
            )
        else:  # fusion
            model = create_fusion_model(
                num_classes=5,
                feature_dim=512,
                use_acoustic=False,
                fusion_type='attention'
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_rgb(self, image_path: str):
        """Predict using RGB-only model."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.rgb_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'image': np.array(image)
        }
    
    def predict_fusion(self, image_path: str):
        """Predict using fusion model."""
        if self.fusion_model is None:
            raise ValueError("Fusion model not loaded")
        
        # Load and preprocess RGB image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate thermal map
        thermal_map = self.thermal_simulator.generate_thermal_map(image_tensor)
        thermal_tensor = torch.from_numpy(thermal_map).unsqueeze(0).unsqueeze(0).float().to(self.device)
        # Normalize thermal tensor
        thermal_tensor = (thermal_tensor - 0.5) / 0.5
        
        # Predict
        with torch.no_grad():
            outputs = self.fusion_model(image_tensor, thermal_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'image': np.array(image),
            'thermal_map': thermal_map
        }
    
    def visualize_prediction(self, result: dict, save_path: str = None):
        """Visualize prediction results."""
        if 'thermal_map' in result:
            # Fusion model visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            ax1.imshow(result['image'])
            ax1.set_title('Original RGB Image')
            ax1.axis('off')
            
            # Thermal map
            ax2.imshow(result['thermal_map'], cmap='jet')
            ax2.set_title('Simulated Thermal Map')
            ax2.axis('off')
            
            # Thermal overlay
            overlay = np.copy(result['image'])
            thermal_colored = cm.jet(result['thermal_map'])[:, :, :3] * 255
            overlay = cv2.addWeighted(overlay.astype(np.uint8), 0.6, 
                                    thermal_colored.astype(np.uint8), 0.4, 0)
            ax3.imshow(overlay)
            ax3.set_title('RGB + Thermal Overlay')
            ax3.axis('off')
            
            # Probabilities
            ax4.bar(self.class_names, result['probabilities'], alpha=0.7)
            ax4.set_title(f'Prediction: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}')
            ax4.set_ylabel('Probability')
            ax4.tick_params(axis='x', rotation=45)
            
            model_type = "Fusion Model"
        else:
            # RGB-only visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax1.imshow(result['image'])
            ax1.set_title('Original RGB Image')
            ax1.axis('off')
            
            # Probabilities
            ax2.bar(self.class_names, result['probabilities'], alpha=0.7)
            ax2.set_title(f'Prediction: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}')
            ax2.set_ylabel('Probability')
            ax2.tick_params(axis='x', rotation=45)
            
            model_type = "RGB-only Model"
        
        plt.suptitle(f'{model_type} Prediction Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, image_path: str, save_path: str = None):
        """Compare RGB and fusion model predictions on the same image."""
        if self.fusion_model is None:
            print("⚠️  Fusion model not available for comparison")
            return self.predict_rgb(image_path)
        
        # Get predictions from both models
        rgb_result = self.predict_rgb(image_path)
        fusion_result = self.predict_fusion(image_path)
        
        # Create comparison visualization
        fig = plt.figure(figsize=(16, 10))
        
        # Original image (large)
        ax1 = plt.subplot(3, 4, (1, 5))
        ax1.imshow(rgb_result['image'])
        ax1.set_title('Original RGB Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Thermal map
        ax2 = plt.subplot(3, 4, (2, 6))
        ax2.imshow(fusion_result['thermal_map'], cmap='jet')
        ax2.set_title('Simulated Thermal Map', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # RGB probabilities
        ax3 = plt.subplot(3, 4, (3, 4))
        bars1 = ax3.bar(range(len(self.class_names)), rgb_result['probabilities'], alpha=0.7, color='lightblue')
        ax3.set_title(f'RGB Model\nPrediction: {rgb_result["predicted_class"]}\nConfidence: {rgb_result["confidence"]}:.3f}')
        ax3.set_ylabel('Probability')
        ax3.set_xticks(range(len(self.class_names)))
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Fusion probabilities
        ax4 = plt.subplot(3, 4, (7, 8))
        bars2 = ax4.bar(range(len(self.class_names)), fusion_result['probabilities'], alpha=0.7, color='lightcoral')
        ax4.set_title(f'Fusion Model\nPrediction: {fusion_result["predicted_class"]}\nConfidence: {fusion_result["confidence"]}:.3f}')
        ax4.set_ylabel('Probability')
        ax4.set_xticks(range(len(self.class_names)))
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Comparison summary
        ax5 = plt.subplot(3, 4, (9, 12))
        ax5.axis('off')
        
        # Create comparison text
        agreement = "✅ AGREE" if rgb_result['predicted_class'] == fusion_result['predicted_class'] else "❌ DISAGREE"
        confidence_diff = fusion_result['confidence'] - rgb_result['confidence']
        improvement = "↗️ HIGHER" if confidence_diff > 0 else "↘️ LOWER" if confidence_diff < 0 else "➡️ SAME"
        
        comparison_text = f"""
MODEL COMPARISON SUMMARY
========================

Agreement: {agreement}

RGB Model:
• Prediction: {rgb_result['predicted_class']}
• Confidence: {rgb_result['confidence']:.3f}

Fusion Model:
• Prediction: {fusion_result['predicted_class']}
• Confidence: {fusion_result['confidence']:.3f}

Confidence Difference: {improvement}
({confidence_diff:+.3f})
        """
        
        ax5.text(0.1, 0.5, comparison_text, fontsize=12, fontfamily='monospace',
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.suptitle('🥭 Multi-Modal Mango Disease Classification Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison saved to {save_path}")
        
        plt.show()
        
        return {
            'rgb_result': rgb_result,
            'fusion_result': fusion_result,
            'agreement': rgb_result['predicted_class'] == fusion_result['predicted_class'],
            'confidence_difference': confidence_diff
        }


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Mango Disease Classification Demo')
    parser.add_argument('--rgb_model', type=str, required=True,
                       help='Path to RGB model checkpoint')
    parser.add_argument('--fusion_model', type=str,
                       help='Path to fusion model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='demo_result.png',
                       help='Output path for visualization')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = MangoClassificationDemo(
        rgb_model_path=args.rgb_model,
        fusion_model_path=args.fusion_model,
        device=args.device
    )
    
    print(f"\n🥭 Mango Disease Classification Demo")
    print(f"📸 Processing image: {args.image}")
    print("="*50)
    
    if args.fusion_model:
        # Compare both models
        comparison = demo.compare_models(args.image, args.output)
        
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"RGB Model: {comparison['rgb_result']['predicted_class']} ({comparison['rgb_result']['confidence']:.3f})")
        print(f"Fusion Model: {comparison['fusion_result']['predicted_class']} ({comparison['fusion_result']['confidence']:.3f})")
        print(f"Agreement: {'Yes' if comparison['agreement'] else 'No'}")
        print(f"Confidence Difference: {comparison['confidence_difference']:+.3f}")
    else:
        # RGB-only prediction
        result = demo.predict_rgb(args.image)
        demo.visualize_prediction(result, args.output)
        
        print(f"\n📊 RGB MODEL RESULTS:")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    print(f"\n✅ Demo completed! Results saved to {args.output}")


if __name__ == "__main__":
    main() 