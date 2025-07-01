#!/usr/bin/env python3
"""
Ultra-Optimized Training for Maximum Accuracy
Implements the most aggressive optimization techniques.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

# Add project modules to path
sys.path.append(str(Path(__file__).parent))
from models.rgb_branch import RGBBranch, RGBTrainer, create_rgb_branch
from scripts.dataloader import create_dataloaders

class UltraOptimizedTrainer:
    """Ultra-optimized trainer with aggressive techniques."""
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.to(device)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Ultra-optimized loss with focal loss + label smoothing
        self.alpha = torch.tensor([1.0, 2.0, 1.5, 2.0, 1.5]).to(device)  # Class weights
        self.focal_gamma = 2.0
        self.label_smoothing = 0.15
        
    def focal_loss_with_smoothing(self, inputs, targets):
        """Focal loss with label smoothing for hard examples."""
        # Label smoothing
        num_classes = inputs.size(1)
        smooth_targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Convert to one-hot
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def train_epoch_ultra(self, dataloader, optimizer, scheduler, epoch):
        """Ultra-optimized training epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            rgb_images, thermal_images, labels = batch
            rgb_images = rgb_images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(rgb_images)
                loss = self.focal_loss_with_smoothing(outputs, labels)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }

def create_ultra_optimized_model(num_classes=5, backbone='efficientnet_b3'):
    """Create ultra-optimized model with best practices."""
    import timm
    
    # Use EfficientNet-B3 for maximum accuracy
    base_model = timm.create_model(
        backbone,
        pretrained=True,
        num_classes=0,
        global_pool='avg'
    )
    
    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        features = base_model(dummy_input)
        feature_dim = features.shape[1]
    
    # Ultra-optimized classifier with multiple techniques
    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(feature_dim, feature_dim // 2),
        nn.LayerNorm(feature_dim // 2),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(feature_dim // 2, feature_dim // 4),
        nn.LayerNorm(feature_dim // 4),
        nn.GELU(), 
        nn.Dropout(0.1),
        nn.Linear(feature_dim // 4, num_classes)
    )
    
    # Combine into full model
    model = nn.Sequential(base_model, classifier)
    
    return model

def ultra_optimized_training():
    """Run ultra-optimized training with maximum techniques."""
    print("🚀 ULTRA-OPTIMIZED TRAINING FOR MAXIMUM ACCURACY")
    print("=" * 70)
    
    # Setup device with optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"🔧 Device: {device}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    
    # Create ultra-optimized dataloaders
    print("📊 Creating ultra-optimized dataloaders...")
    dataloaders = create_dataloaders(
        rgb_data_path='data/processed/fruit',
        thermal_data_path='data/thermal',
        batch_size=32,  # Larger batch size for stability
        num_workers=4,
        image_size=224,
        use_acoustic=False
    )
    
    # Create ultra-optimized model
    print("🤖 Creating ultra-optimized model...")
    class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
    
    model = create_ultra_optimized_model(
        num_classes=len(class_names),
        backbone='efficientnet_b3'  # Best backbone for accuracy
    )
    
    # Create ultra trainer
    trainer = UltraOptimizedTrainer(model, device, class_names)
    
    # Ultra-optimized optimizer (AdamW with better settings)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,  # Higher learning rate with OneCycleLR
        weight_decay=1e-3,  # Stronger regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # OneCycleLR for maximum performance
    total_steps = len(dataloaders['train']) * 50  # 50 epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=total_steps,
        pct_start=0.1,  # Warm up for 10% of training
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95
    )
    
    print("🔧 ULTRA-OPTIMIZED CONFIGURATION:")
    print("✅ EfficientNet-B3 backbone (state-of-the-art)")
    print("✅ Focal loss + Label smoothing (0.15)")
    print("✅ Mixed precision training")
    print("✅ OneCycleLR scheduler")
    print("✅ Aggressive gradient clipping (0.5)")
    print("✅ Class-weighted loss")
    print("✅ Multi-layer classifier with LayerNorm")
    
    # Training loop
    best_val_acc = 0.0
    model_save_dir = Path("models/checkpoints")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = f"ultra_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n🚀 Starting ultra-optimized training...")
    print(f"🎯 TARGET: 95%+ accuracy")
    
    start_time = time.time()
    
    for epoch in range(1, 51):  # 50 epochs with aggressive schedule
        # Ultra-optimized training
        train_metrics = trainer.train_epoch_ultra(
            dataloaders['train'], optimizer, scheduler, epoch
        )
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloaders['val']:
                rgb_images, thermal_images, labels = batch
                rgb_images = rgb_images.to(device)
                labels = labels.to(device)
                
                with autocast():
                    outputs = model(rgb_images)
                    loss = trainer.focal_loss_with_smoothing(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100. * val_correct / val_total
        
        # Log progress
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d} | Train: {train_metrics['accuracy']:5.2f}% | Val: {val_accuracy:5.2f}% | LR: {current_lr:.6f}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = model_save_dir / f"{experiment_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_accuracy,
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_model_path)
            print(f"🔥 NEW BEST: {val_accuracy:.2f}% (saved)")
        
        # Early celebration if we hit target
        if val_accuracy >= 95.0:
            print(f"🎉 TARGET ACHIEVED: {val_accuracy:.2f}% >= 95%!")
            break
    
    training_time = time.time() - start_time
    
    # Final test evaluation
    print(f"\n📊 FINAL EVALUATION:")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in dataloaders['test']:
            rgb_images, thermal_images, labels = batch
            rgb_images = rgb_images.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(rgb_images)
            
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_accuracy = 100. * test_correct / test_total
    
    print(f"🏆 ULTRA-OPTIMIZED RESULTS:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Compare to previous best
    previous_best = 92.06
    if test_accuracy > previous_best:
        improvement = test_accuracy - previous_best
        print(f"🚀 IMPROVEMENT: +{improvement:.2f}% over baseline!")
    
    if test_accuracy >= 95.0:
        print("🎉 MISSION ACCOMPLISHED: 95%+ ACCURACY ACHIEVED!")
    elif test_accuracy >= 93.0:
        print("🥇 EXCELLENT: 93%+ accuracy achieved!")
    elif test_accuracy >= 90.0:
        print("🥈 VERY GOOD: 90%+ accuracy achieved!")
    
    return best_val_acc, test_accuracy

if __name__ == "__main__":
    ultra_optimized_training() 