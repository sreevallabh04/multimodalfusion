"""
Main training script for multi-modal mango fruit disease classification.
Trains RGB baseline and fusion models with comprehensive logging and evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Add project modules to path
sys.path.append(str(Path(__file__).parent))
from models.rgb_branch import RGBBranch, RGBTrainer, create_rgb_branch
from models.fusion_model import MultiModalFusionModel, FusionTrainer, create_fusion_model
from scripts.dataloader import create_dataloaders


class TrainingLogger:
    """Enhanced logging for training process."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.log_dir / f"{experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        """Log epoch results."""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['learning_rates'].append(lr)
        
        self.logger.info(f"Epoch {epoch}")
        self.logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        self.logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        self.logger.info(f"  Learning Rate: {lr:.6f}")
    
    def save_plots(self, save_path: str):
        """Save training plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(epochs, self.history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # F1 score comparison (if available)
        if len(self.history['val_acc']) > 0:
            ax4.plot(epochs, self.history['val_acc'], 'r-', linewidth=2)
            ax4.set_title('Validation Accuracy Over Time')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy (%)')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def train_rgb_baseline(args):
    """Train RGB-only baseline model."""
    print("🚀 Training RGB Baseline Model")
    print("=" * 60)
    
    # Setup logging
    experiment_name = f"rgb_baseline_{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(args.log_dir, experiment_name)
    logger.logger.info(f"Starting RGB baseline training with config: {vars(args)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        rgb_data_path=args.rgb_data_path,
        thermal_data_path=args.thermal_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_acoustic=False
    )
    
    # Create model
    logger.logger.info("Creating RGB model...")
    class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
    model = create_rgb_branch(
        num_classes=len(class_names),
        backbone=args.backbone,
        pretrained=True,
        feature_dim=args.feature_dim
    )
    
    # Create trainer
    trainer = RGBTrainer(model, device, class_names)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Training loop
    best_val_acc = 0.0
    model_save_dir = Path(args.model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(dataloaders['train'], optimizer, epoch)
        
        # Validate
        val_metrics = trainer.validate(dataloaders['val'])
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
        
        # Log epoch
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_path = model_save_dir / f"{experiment_name}_best.pth"
            trainer.save_model(str(best_model_path), epoch, val_metrics)
        
        # Check early stopping
        if early_stopping(val_metrics['loss'], model):
            logger.logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    logger.logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = model_save_dir / f"{experiment_name}_final.pth"
    trainer.save_model(str(final_model_path), epoch, val_metrics)
    
    # Save training plots
    plot_path = logger.log_dir / f"{experiment_name}_plots.png"
    logger.save_plots(str(plot_path))
    
    # Test evaluation
    logger.logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(dataloaders['test'])
    logger.logger.info(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Save configuration
    config = {
        'experiment_name': experiment_name,
        'model_type': 'rgb_baseline',
        'best_val_acc': best_val_acc,
        'test_acc': test_metrics['accuracy'],
        'training_time': training_time,
        'args': vars(args)
    }
    
    config_path = logger.log_dir / f"{experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(best_model_path), test_metrics


def train_fusion_model(args, rgb_model_path: str = None):
    """Train multi-modal fusion model."""
    print("🚀 Training Multi-Modal Fusion Model")
    print("=" * 60)
    
    # Setup logging
    fusion_suffix = "with_acoustic" if args.use_acoustic else "rgb_thermal"
    experiment_name = f"fusion_{fusion_suffix}_{args.fusion_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(args.log_dir, experiment_name)
    logger.logger.info(f"Starting fusion training with config: {vars(args)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        rgb_data_path=args.rgb_data_path,
        thermal_data_path=args.thermal_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_acoustic=args.use_acoustic
    )
    
    # Create fusion model
    logger.logger.info("Creating fusion model...")
    class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
    model = create_fusion_model(
        num_classes=len(class_names),
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        use_acoustic=args.use_acoustic,
        fusion_type=args.fusion_type
    )
    
    # Load pre-trained RGB weights if available
    if rgb_model_path and os.path.exists(rgb_model_path):
        logger.logger.info(f"Loading pre-trained RGB weights from {rgb_model_path}")
        model.load_rgb_pretrained(rgb_model_path)
        
        if args.freeze_rgb_epochs > 0:
            logger.logger.info(f"Freezing RGB branch for {args.freeze_rgb_epochs} epochs")
            model.freeze_rgb_branch()
    
    # Create trainer
    trainer = FusionTrainer(model, device, class_names)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Training loop
    best_val_acc = 0.0
    model_save_dir = Path(args.model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Unfreeze RGB branch after specified epochs
        if epoch == args.freeze_rgb_epochs + 1 and args.freeze_rgb_epochs > 0:
            logger.logger.info("Unfreezing RGB branch")
            model.unfreeze_rgb_branch()
            # Reduce learning rate when unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # Train
        train_metrics = trainer.train_epoch(dataloaders['train'], optimizer, epoch)
        
        # Validate
        val_metrics = trainer.validate(dataloaders['val'])
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
        
        # Log epoch
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_path = model_save_dir / f"{experiment_name}_best.pth"
            trainer.save_model(str(best_model_path), epoch, val_metrics)
        
        # Check early stopping
        if early_stopping(val_metrics['loss'], model):
            logger.logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    logger.logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = model_save_dir / f"{experiment_name}_final.pth"
    trainer.save_model(str(final_model_path), epoch, val_metrics)
    
    # Save training plots
    plot_path = logger.log_dir / f"{experiment_name}_plots.png"
    logger.save_plots(str(plot_path))
    
    # Test evaluation
    logger.logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(dataloaders['test'])
    logger.logger.info(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Save configuration
    config = {
        'experiment_name': experiment_name,
        'model_type': 'fusion',
        'fusion_type': args.fusion_type,
        'use_acoustic': args.use_acoustic,
        'best_val_acc': best_val_acc,
        'test_acc': test_metrics['accuracy'],
        'training_time': training_time,
        'rgb_model_path': rgb_model_path,
        'args': vars(args)
    }
    
    config_path = logger.log_dir / f"{experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(best_model_path), test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-modal mango disease classification models')
    
    # Data arguments
    parser.add_argument('--rgb_data_path', type=str, default='data/processed/fruit',
                       help='Path to processed RGB fruit data')
    parser.add_argument('--thermal_data_path', type=str, default='data/thermal',
                       help='Path to thermal map data')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1'],
                       help='Backbone architecture')
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='Feature dimension for fusion')
    parser.add_argument('--fusion_type', type=str, default='attention',
                       choices=['attention', 'concat', 'average'],
                       help='Type of fusion mechanism')
    parser.add_argument('--use_acoustic', action='store_true',
                       help='Include acoustic/texture features')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--freeze_rgb_epochs', type=int, default=10,
                       help='Number of epochs to freeze RGB branch in fusion model')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--model_save_dir', type=str, default='models/checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logging')
    
    # Training strategy
    parser.add_argument('--train_mode', type=str, default='both',
                       choices=['rgb_only', 'fusion_only', 'both'],
                       help='Which models to train')
    parser.add_argument('--skip_rgb_pretrain', action='store_true',
                       help='Skip RGB pretraining for fusion model')
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Print configuration
    print("🔧 Training Configuration:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-" * 40)
    
    # Training pipeline
    rgb_model_path = None
    
    if args.train_mode in ['rgb_only', 'both']:
        print("\n" + "="*60)
        print("PHASE 1: RGB BASELINE TRAINING")
        print("="*60)
        
        rgb_model_path, rgb_test_metrics = train_rgb_baseline(args)
        
        print(f"\n✅ RGB baseline training completed!")
        print(f"📊 RGB Test Accuracy: {rgb_test_metrics['accuracy']:.2f}%")
    
    if args.train_mode in ['fusion_only', 'both']:
        print("\n" + "="*60)
        print("PHASE 2: FUSION MODEL TRAINING")
        print("="*60)
        
        if args.skip_rgb_pretrain:
            rgb_model_path = None
        
        fusion_model_path, fusion_test_metrics = train_fusion_model(args, rgb_model_path)
        
        print(f"\n✅ Fusion model training completed!")
        print(f"📊 Fusion Test Accuracy: {fusion_test_metrics['accuracy']:.2f}%")
        
        # Compare results
        if args.train_mode == 'both':
            improvement = fusion_test_metrics['accuracy'] - rgb_test_metrics['accuracy']
            print(f"\n📈 Performance Comparison:")
            print(f"RGB-only:  {rgb_test_metrics['accuracy']:.2f}%")
            print(f"Fusion:    {fusion_test_metrics['accuracy']:.2f}%")
            print(f"Improvement: {improvement:+.2f}%")
    
    print("\n🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 