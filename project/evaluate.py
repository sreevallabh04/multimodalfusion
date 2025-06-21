"""
Comprehensive evaluation script for multi-modal mango disease classification.
Includes metrics calculation, confusion matrix, CAM visualizations, and model comparison.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import cv2
from PIL import Image
from tqdm import tqdm

# Add project modules to path
sys.path.append(str(Path(__file__).parent))
from models.rgb_branch import RGBBranch, create_rgb_branch
from models.fusion_model import MultiModalFusionModel, create_fusion_model
from scripts.dataloader import create_dataloaders


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 class_names: List[str],
                 model_type: str = 'rgb'):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model_type = model_type
        self.num_classes = len(class_names)
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate_dataset(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate model on dataset and return comprehensive metrics."""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        sample_paths = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if self.model_type == 'fusion':
                    if len(batch) == 4:  # With acoustic
                        rgb_images, thermal_images, acoustic_images, labels = batch
                        rgb_images = rgb_images.to(self.device)
                        thermal_images = thermal_images.to(self.device)
                        acoustic_images = acoustic_images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(rgb_images, thermal_images, acoustic_images)
                    else:  # Without acoustic
                        rgb_images, thermal_images, labels = batch
                        rgb_images = rgb_images.to(self.device)
                        thermal_images = thermal_images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(rgb_images, thermal_images)
                else:  # RGB only
                    if len(batch) == 3:  # RGB + thermal + label
                        rgb_images, _, labels = batch
                    elif len(batch) == 4:  # RGB + thermal + acoustic + label
                        rgb_images, _, _, labels = batch
                    else:  # Just RGB + label
                        rgb_images, labels = batch
                    
                    rgb_images = rgb_images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(rgb_images)
                
                # Get predictions and probabilities
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        return metrics
    
    def _calculate_metrics(self, labels: List[int], predictions: List[int], probabilities: List[List[float]]) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        labels = np.array(labels)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Basic metrics
        accuracy = (predictions == labels).mean() * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # ROC AUC (for multiclass)
        try:
            labels_binarized = label_binarize(labels, classes=range(self.num_classes))
            if labels_binarized.shape[1] == 1:  # Binary case
                auc_score = roc_auc_score(labels, probabilities[:, 1])
            else:  # Multiclass case
                auc_score = roc_auc_score(labels_binarized, probabilities, multi_class='ovr', average='macro')
        except:
            auc_score = 0.0
        
        # Classification report
        class_report = classification_report(
            labels, predictions, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'auc_score': auc_score,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'labels': labels.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        return metrics
    
    def plot_confusion_matrix(self, metrics: Dict, save_path: str = None, title: str = None):
        """Plot and save confusion matrix."""
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annotations = []
        for i in range(cm.shape[0]):
            row_annotations = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percent = cm_percent[i, j]
                annotation = f'{count}\n({percent:.1f}%)'
                row_annotations.append(annotation)
            annotations.append(row_annotations)
        
        # Plot heatmap
        sns.heatmap(
            cm_percent,
            annot=annotations,
            fmt='',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title(title or f'Confusion Matrix - {self.model_type.upper()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_class_metrics(self, metrics: Dict, save_path: str = None):
        """Plot per-class precision, recall, and F1-score."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        x_pos = np.arange(len(self.class_names))
        
        # Precision
        ax1.bar(x_pos, metrics['precision_per_class'], alpha=0.8, color='lightblue')
        ax1.set_title('Precision per Class')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(metrics['precision_per_class']):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Recall
        ax2.bar(x_pos, metrics['recall_per_class'], alpha=0.8, color='lightgreen')
        ax2.set_title('Recall per Class')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        for i, v in enumerate(metrics['recall_per_class']):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1-score
        ax3.bar(x_pos, metrics['f1_per_class'], alpha=0.8, color='lightcoral')
        ax3.set_title('F1-Score per Class')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        for i, v in enumerate(metrics['f1_per_class']):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_cam_visualizations(self, 
                                  dataloader: torch.utils.data.DataLoader,
                                  num_samples: int = 8,
                                  save_dir: str = None) -> None:
        """Generate Class Activation Map (CAM) visualizations."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
        # Get random samples
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            if self.model_type == 'fusion':
                if len(batch) == 4:  # With acoustic
                    rgb_images, thermal_images, acoustic_images, labels = batch
                    rgb_images = rgb_images.to(self.device)
                    thermal_images = thermal_images.to(self.device)
                    acoustic_images = acoustic_images.to(self.device)
                    labels = labels.to(self.device)
                else:  # Without acoustic
                    rgb_images, thermal_images, labels = batch
                    rgb_images = rgb_images.to(self.device)
                    thermal_images = thermal_images.to(self.device)
                    labels = labels.to(self.device)
                    acoustic_images = None
            else:  # RGB only
                if len(batch) == 3:  # RGB + thermal + label
                    rgb_images, _, labels = batch
                elif len(batch) == 4:  # RGB + thermal + acoustic + label
                    rgb_images, _, _, labels = batch
                else:  # Just RGB + label
                    rgb_images, labels = batch
                
                rgb_images = rgb_images.to(self.device)
                labels = labels.to(self.device)
            
            batch_size = rgb_images.size(0)
            
            for i in range(min(batch_size, num_samples - sample_count)):
                self._generate_single_cam(
                    rgb_images[i:i+1], 
                    thermal_images[i:i+1] if self.model_type == 'fusion' else None,
                    acoustic_images[i:i+1] if self.model_type == 'fusion' and acoustic_images is not None else None,
                    labels[i:i+1],
                    sample_count,
                    save_dir
                )
                sample_count += 1
    
    def _generate_single_cam(self, 
                           rgb_image: torch.Tensor,
                           thermal_image: torch.Tensor = None,
                           acoustic_image: torch.Tensor = None,
                           label: torch.Tensor = None,
                           sample_idx: int = 0,
                           save_dir: Path = None):
        """Generate CAM for a single sample."""
        try:
            # Get model prediction
            if self.model_type == 'fusion':
                if acoustic_image is not None:
                    outputs = self.model(rgb_image, thermal_image, acoustic_image)
                else:
                    outputs = self.model(rgb_image, thermal_image)
            else:
                # Try to get attention map from RGB model
                try:
                    outputs, attention = self.model(rgb_image, return_features=False, return_attention=True)
                except:
                    outputs = self.model(rgb_image)
                    attention = None
            
            # Get prediction
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Convert images for visualization
            rgb_np = rgb_image.squeeze().cpu().numpy()
            rgb_np = np.transpose(rgb_np, (1, 2, 0))
            
            # Denormalize RGB image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_np = std * rgb_np + mean
            rgb_np = np.clip(rgb_np, 0, 1)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3 if self.model_type == 'fusion' else 2, figsize=(15, 5))
            
            # Original RGB image
            axes[0].imshow(rgb_np)
            axes[0].set_title(f'Original RGB\nTrue: {self.class_names[label.item()]}\nPred: {self.class_names[predicted.item()]}\nConf: {probabilities[0, predicted].item():.3f}')
            axes[0].axis('off')
            
            # Thermal image (if available)
            if self.model_type == 'fusion' and thermal_image is not None:
                thermal_np = thermal_image.squeeze().cpu().numpy()
                axes[1].imshow(thermal_np, cmap='jet')
                axes[1].set_title('Thermal Map')
                axes[1].axis('off')
            
            # Attention/CAM visualization
            cam_idx = 2 if self.model_type == 'fusion' else 1
            if 'attention' in locals() and attention is not None:
                attention_np = attention.squeeze().cpu().numpy()
                im = axes[cam_idx].imshow(rgb_np)
                axes[cam_idx].imshow(attention_np, alpha=0.5, cmap='jet')
                axes[cam_idx].set_title('RGB Attention Map')
            else:
                # Fallback: show prediction confidence as text
                axes[cam_idx].text(0.5, 0.5, f'Prediction Confidence\n{probabilities[0, predicted].item():.3f}', 
                                 ha='center', va='center', transform=axes[cam_idx].transAxes,
                                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[cam_idx].set_title('Prediction Confidence')
            axes[cam_idx].axis('off')
            
            plt.tight_layout()
            
            if save_dir:
                save_path = save_dir / f'cam_sample_{sample_idx:03d}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            print(f"Error generating CAM for sample {sample_idx}: {e}")


def load_model(model_path: str, model_type: str, device: torch.device, class_names: List[str]):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    if model_type == 'rgb':
        model = create_rgb_branch(
            num_classes=len(class_names),
            backbone=checkpoint.get('model_config', {}).get('backbone', 'resnet18'),
            feature_dim=checkpoint.get('model_config', {}).get('feature_dim', 512)
        )
    elif model_type == 'fusion':
        model_config = checkpoint.get('model_config', {})
        model = create_fusion_model(
            num_classes=len(class_names),
            feature_dim=model_config.get('feature_dim', 512),
            use_acoustic=model_config.get('use_acoustic', False),
            fusion_type=model_config.get('fusion_type', 'attention')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded {model_type} model from {model_path}")
    
    return model


def compare_models(rgb_metrics: Dict, fusion_metrics: Dict, save_path: str = None):
    """Compare RGB and fusion model performance."""
    # Prepare data for comparison
    metrics_comparison = {
        'Model': ['RGB-only', 'Fusion'],
        'Accuracy (%)': [rgb_metrics['accuracy'], fusion_metrics['accuracy']],
        'F1-Score (Macro)': [rgb_metrics['f1_macro'], fusion_metrics['f1_macro']],
        'F1-Score (Weighted)': [rgb_metrics['f1_weighted'], fusion_metrics['f1_weighted']],
        'AUC Score': [rgb_metrics['auc_score'], fusion_metrics['auc_score']]
    }
    
    df = pd.DataFrame(metrics_comparison)
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    bars1 = ax1.bar(df['Model'], df['Accuracy (%)'], color=['lightblue', 'lightcoral'], alpha=0.8)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    for bar, value in zip(bars1, df['Accuracy (%)']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    x_pos = np.arange(len(df['Model']))
    width = 0.35
    
    bars2 = ax2.bar(x_pos - width/2, df['F1-Score (Macro)'], width, label='Macro', alpha=0.8, color='lightgreen')
    bars3 = ax2.bar(x_pos + width/2, df['F1-Score (Weighted)'], width, label='Weighted', alpha=0.8, color='orange')
    
    ax2.set_title('F1-Score Comparison')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df['Model'])
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Per-class F1-score comparison
    class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
    x_pos_class = np.arange(len(class_names))
    
    bars4 = ax3.bar(x_pos_class - width/2, rgb_metrics['f1_per_class'], width, 
                   label='RGB-only', alpha=0.8, color='lightblue')
    bars5 = ax3.bar(x_pos_class + width/2, fusion_metrics['f1_per_class'], width,
                   label='Fusion', alpha=0.8, color='lightcoral')
    
    ax3.set_title('Per-Class F1-Score Comparison')
    ax3.set_ylabel('F1-Score')
    ax3.set_xticks(x_pos_class)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Improvement analysis
    improvement = [
        fusion_metrics['accuracy'] - rgb_metrics['accuracy'],
        fusion_metrics['f1_macro'] - rgb_metrics['f1_macro'],
        fusion_metrics['f1_weighted'] - rgb_metrics['f1_weighted'],
        fusion_metrics['auc_score'] - rgb_metrics['auc_score']
    ]
    
    improvement_metrics = ['Accuracy', 'F1-Macro', 'F1-Weighted', 'AUC']
    colors = ['green' if x > 0 else 'red' for x in improvement]
    
    bars6 = ax4.bar(improvement_metrics, improvement, color=colors, alpha=0.7)
    ax4.set_title('Fusion Model Improvement over RGB-only')
    ax4.set_ylabel('Improvement')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars6, improvement):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.001),
                f'{value:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary
    print("\n📊 MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(df.to_string(index=False))
    print("\n📈 IMPROVEMENTS (Fusion vs RGB-only):")
    for metric, imp in zip(improvement_metrics, improvement):
        print(f"  {metric}: {imp:+.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-modal mango disease classification models')
    
    # Model paths
    parser.add_argument('--rgb_model_path', type=str, required=True,
                       help='Path to trained RGB model')
    parser.add_argument('--fusion_model_path', type=str,
                       help='Path to trained fusion model')
    
    # Data arguments
    parser.add_argument('--rgb_data_path', type=str, default='data/processed/fruit',
                       help='Path to processed RGB fruit data')
    parser.add_argument('--thermal_data_path', type=str, default='data/thermal',
                       help='Path to thermal map data')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_acoustic', action='store_true',
                       help='Include acoustic/texture features for fusion model')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_cam_samples', type=int, default=8,
                       help='Number of samples for CAM visualization')
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Class names
    class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
    
    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        rgb_data_path=args.rgb_data_path,
        thermal_data_path=args.thermal_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_acoustic=args.use_acoustic
    )
    
    # Evaluate RGB model
    print("\n🔍 Evaluating RGB Model...")
    rgb_model = load_model(args.rgb_model_path, 'rgb', device, class_names)
    rgb_evaluator = ModelEvaluator(rgb_model, device, class_names, 'rgb')
    rgb_metrics = rgb_evaluator.evaluate_dataset(dataloaders['test'])
    
    print(f"RGB Model Test Accuracy: {rgb_metrics['accuracy']:.2f}%")
    print(f"RGB Model F1-Score (Macro): {rgb_metrics['f1_macro']:.3f}")
    
    # Save RGB results
    rgb_evaluator.plot_confusion_matrix(rgb_metrics, str(output_dir / 'rgb_confusion_matrix.png'), 'RGB Model')
    rgb_evaluator.plot_per_class_metrics(rgb_metrics, str(output_dir / 'rgb_per_class_metrics.png'))
    rgb_evaluator.generate_cam_visualizations(dataloaders['test'], args.num_cam_samples, str(output_dir / 'rgb_cam'))
    
    # Save RGB metrics
    with open(output_dir / 'rgb_metrics.json', 'w') as f:
        json.dump(rgb_metrics, f, indent=2)
    
    fusion_metrics = None
    
    # Evaluate fusion model if provided
    if args.fusion_model_path and os.path.exists(args.fusion_model_path):
        print("\n🔍 Evaluating Fusion Model...")
        fusion_model = load_model(args.fusion_model_path, 'fusion', device, class_names)
        fusion_evaluator = ModelEvaluator(fusion_model, device, class_names, 'fusion')
        fusion_metrics = fusion_evaluator.evaluate_dataset(dataloaders['test'])
        
        print(f"Fusion Model Test Accuracy: {fusion_metrics['accuracy']:.2f}%")
        print(f"Fusion Model F1-Score (Macro): {fusion_metrics['f1_macro']:.3f}")
        
        # Save fusion results
        fusion_evaluator.plot_confusion_matrix(fusion_metrics, str(output_dir / 'fusion_confusion_matrix.png'), 'Fusion Model')
        fusion_evaluator.plot_per_class_metrics(fusion_metrics, str(output_dir / 'fusion_per_class_metrics.png'))
        fusion_evaluator.generate_cam_visualizations(dataloaders['test'], args.num_cam_samples, str(output_dir / 'fusion_cam'))
        
        # Save fusion metrics
        with open(output_dir / 'fusion_metrics.json', 'w') as f:
            json.dump(fusion_metrics, f, indent=2)
        
        # Compare models
        print("\n🏆 Comparing Models...")
        compare_models(rgb_metrics, fusion_metrics, str(output_dir / 'model_comparison.png'))
    
    # Generate comprehensive report
    report = {
        'evaluation_date': str(datetime.now()),
        'dataset_info': {
            'test_samples': len(dataloaders['test'].dataset),
            'classes': class_names,
            'image_size': args.image_size
        },
        'rgb_model': {
            'path': args.rgb_model_path,
            'metrics': rgb_metrics
        }
    }
    
    if fusion_metrics:
        report['fusion_model'] = {
            'path': args.fusion_model_path,
            'metrics': fusion_metrics
        }
        
        report['comparison'] = {
            'accuracy_improvement': fusion_metrics['accuracy'] - rgb_metrics['accuracy'],
            'f1_macro_improvement': fusion_metrics['f1_macro'] - rgb_metrics['f1_macro'],
            'f1_weighted_improvement': fusion_metrics['f1_weighted'] - rgb_metrics['f1_weighted']
        }
    
    # Save comprehensive report
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Evaluation completed! Results saved to: {output_dir}")
    print("\n📁 Generated files:")
    for file_path in sorted(output_dir.glob('*')):
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    from datetime import datetime
    main() 