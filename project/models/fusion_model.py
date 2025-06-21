"""
Multi-modal fusion model for mango fruit disease classification.
Combines RGB, thermal, and optional acoustic features using attention-based fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional, Dict
import numpy as np

from .rgb_branch import RGBBranch


class ThermalBranch(nn.Module):
    """CNN branch for processing thermal images."""
    
    def __init__(self,
                 backbone: str = 'resnet18',
                 pretrained: bool = False,  # No pretrained for single channel
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3):
        super(ThermalBranch, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Create backbone and modify first layer for single channel input
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,  # Start from scratch for thermal
            num_classes=0,
            global_pool='avg',
            in_chans=1  # Single channel for thermal
        )
        
        # Get backbone output features
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)
            backbone_out = self.backbone(dummy_input)
            self.backbone_features = backbone_out.shape[1]
        
        # Feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.feature_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for thermal branch."""
        features = self.backbone(x)
        projected_features = self.feature_projector(features)
        return projected_features


class AcousticBranch(nn.Module):
    """CNN branch for processing acoustic/texture images."""
    
    def __init__(self,
                 backbone: str = 'resnet18',
                 pretrained: bool = False,
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3):
        super(AcousticBranch, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Create lightweight backbone for acoustic features
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            global_pool='avg',
            in_chans=1  # Single channel for acoustic
        )
        
        # Get backbone output features
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)
            backbone_out = self.backbone(dummy_input)
            self.backbone_features = backbone_out.shape[1]
        
        # Feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.feature_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for acoustic branch."""
        features = self.backbone(x)
        projected_features = self.feature_projector(features)
        return projected_features


class AttentionFusion(nn.Module):
    """Attention-based fusion module for combining multi-modal features."""
    
    def __init__(self,
                 feature_dim: int = 512,
                 num_modalities: int = 2,
                 dropout_rate: float = 0.2):
        super(AttentionFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Self-attention for each modality
        self.modality_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, modality_features: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-modal features using attention.
        
        Args:
            modality_features: List of feature tensors from different modalities
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        batch_size = modality_features[0].size(0)
        
        # Apply self-attention to each modality
        attended_features = []
        attention_weights = []
        
        for i, features in enumerate(modality_features):
            # Self-attention weights
            attn_weights = self.modality_attention[i](features)
            attended_feat = features * attn_weights
            
            attended_features.append(attended_feat)
            attention_weights.append(attn_weights)
        
        # Stack features for cross-attention
        stacked_features = torch.stack(attended_features, dim=1)  # (B, num_modalities, feature_dim)
        
        # Cross-modal attention
        cross_attended, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Concatenate all features
        concatenated = cross_attended.reshape(batch_size, -1)
        
        # Final fusion
        fused_features = self.fusion_layer(concatenated)
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights, dim=1)  # (B, num_modalities, 1)
        
        return fused_features, attention_weights


class MultiModalFusionModel(nn.Module):
    """
    Multi-modal fusion model for mango fruit disease classification.
    Combines RGB, thermal, and optional acoustic features.
    """
    
    def __init__(self,
                 num_classes: int = 5,
                 rgb_backbone: str = 'resnet18',
                 thermal_backbone: str = 'resnet18',
                 acoustic_backbone: str = 'resnet18',
                 feature_dim: int = 512,
                 use_acoustic: bool = False,
                 fusion_type: str = 'attention',
                 dropout_rate: float = 0.3,
                 pretrained_rgb: bool = True):
        """
        Args:
            num_classes: Number of disease classes
            rgb_backbone: Backbone for RGB branch
            thermal_backbone: Backbone for thermal branch
            acoustic_backbone: Backbone for acoustic branch
            feature_dim: Feature dimension for fusion
            use_acoustic: Whether to include acoustic branch
            fusion_type: Type of fusion ('attention', 'concat', 'average')
            dropout_rate: Dropout rate
            pretrained_rgb: Whether to use pretrained RGB backbone
        """
        super(MultiModalFusionModel, self).__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_acoustic = use_acoustic
        self.fusion_type = fusion_type
        self.num_modalities = 3 if use_acoustic else 2
        
        # Create branches
        self.rgb_branch = RGBBranch(
            num_classes=num_classes,
            backbone=rgb_backbone,
            pretrained=pretrained_rgb,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        self.thermal_branch = ThermalBranch(
            backbone=thermal_backbone,
            pretrained=False,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        if use_acoustic:
            self.acoustic_branch = AcousticBranch(
                backbone=acoustic_backbone,
                pretrained=False,
                feature_dim=feature_dim,
                dropout_rate=dropout_rate
            )
        
        # Fusion module
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(
                feature_dim=feature_dim,
                num_modalities=self.num_modalities,
                dropout_rate=dropout_rate
            )
        elif fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * self.num_modalities, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
        elif fusion_type == 'average':
            self.fusion = None  # Simple averaging
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize fusion and classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, 
               rgb_images: torch.Tensor,
               thermal_images: torch.Tensor,
               acoustic_images: Optional[torch.Tensor] = None,
               return_attention: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the fusion model.
        
        Args:
            rgb_images: RGB input tensor (B, 3, 224, 224)
            thermal_images: Thermal input tensor (B, 1, 224, 224)
            acoustic_images: Acoustic input tensor (B, 1, 224, 224)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple containing (class_logits, attention_weights if requested)
        """
        # Extract features from each branch
        rgb_features = self.rgb_branch(rgb_images, return_features=True)[1]
        thermal_features = self.thermal_branch(thermal_images)
        
        modality_features = [rgb_features, thermal_features]
        
        if self.use_acoustic and acoustic_images is not None:
            acoustic_features = self.acoustic_branch(acoustic_images)
            modality_features.append(acoustic_features)
        
        # Fusion
        attention_weights = None
        
        if self.fusion_type == 'attention':
            fused_features, attention_weights = self.fusion(modality_features)
        elif self.fusion_type == 'concat':
            concatenated = torch.cat(modality_features, dim=1)
            fused_features = self.fusion(concatenated)
        elif self.fusion_type == 'average':
            fused_features = torch.stack(modality_features, dim=0).mean(dim=0)
        
        # Classification
        class_logits = self.classifier(fused_features)
        
        if return_attention and attention_weights is not None:
            return class_logits, attention_weights
        else:
            return class_logits
    
    def load_rgb_pretrained(self, rgb_model_path: str):
        """Load pre-trained RGB branch weights."""
        checkpoint = torch.load(rgb_model_path, map_location='cpu')
        self.rgb_branch.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded pre-trained RGB weights from {rgb_model_path}")
    
    def freeze_rgb_branch(self):
        """Freeze RGB branch parameters."""
        for param in self.rgb_branch.parameters():
            param.requires_grad = False
        print("🔒 RGB branch frozen")
    
    def unfreeze_rgb_branch(self):
        """Unfreeze RGB branch parameters."""
        for param in self.rgb_branch.parameters():
            param.requires_grad = True
        print("🔓 RGB branch unfrozen")


class FusionTrainer:
    """Trainer class for the fusion model."""
    
    def __init__(self,
                 model: MultiModalFusionModel,
                 device: torch.device,
                 class_names: list):
        self.model = model
        self.device = device
        self.class_names = class_names
        
        # Move model to device
        self.model.to(device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if self.model.use_acoustic:
                rgb_images, thermal_images, acoustic_images, labels = batch
                rgb_images = rgb_images.to(self.device)
                thermal_images = thermal_images.to(self.device)
                acoustic_images = acoustic_images.to(self.device)
                labels = labels.to(self.device)
            else:
                rgb_images, thermal_images, labels = batch
                rgb_images = rgb_images.to(self.device)
                thermal_images = thermal_images.to(self.device)
                labels = labels.to(self.device)
                acoustic_images = None
            
            # Forward pass
            optimizer.zero_grad()
            class_logits = self.model(rgb_images, thermal_images, acoustic_images)
            
            # Calculate loss
            loss = self.criterion(class_logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if self.model.use_acoustic:
                    rgb_images, thermal_images, acoustic_images, labels = batch
                    rgb_images = rgb_images.to(self.device)
                    thermal_images = thermal_images.to(self.device)
                    acoustic_images = acoustic_images.to(self.device)
                    labels = labels.to(self.device)
                else:
                    rgb_images, thermal_images, labels = batch
                    rgb_images = rgb_images.to(self.device)
                    thermal_images = thermal_images.to(self.device)
                    labels = labels.to(self.device)
                    acoustic_images = None
                
                # Forward pass
                class_logits = self.model(rgb_images, thermal_images, acoustic_images)
                
                # Calculate loss
                loss = self.criterion(class_logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = class_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return metrics
    
    def save_model(self, path: str, epoch: int, metrics: dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'class_names': self.class_names,
            'model_config': {
                'num_classes': self.model.num_classes,
                'feature_dim': self.model.feature_dim,
                'use_acoustic': self.model.use_acoustic,
                'fusion_type': self.model.fusion_type,
                'num_modalities': self.model.num_modalities
            }
        }, path)
        print(f"✅ Fusion model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Fusion model loaded from {path}")
        return checkpoint


def create_fusion_model(num_classes: int = 5,
                       backbone: str = 'resnet18',
                       feature_dim: int = 512,
                       use_acoustic: bool = False,
                       fusion_type: str = 'attention') -> MultiModalFusionModel:
    """
    Factory function to create fusion model.
    
    Args:
        num_classes: Number of disease classes
        backbone: Backbone architecture for all branches
        feature_dim: Feature dimension for fusion
        use_acoustic: Whether to include acoustic branch
        fusion_type: Type of fusion ('attention', 'concat', 'average')
        
    Returns:
        MultiModalFusionModel
    """
    model = MultiModalFusionModel(
        num_classes=num_classes,
        rgb_backbone=backbone,
        thermal_backbone=backbone,
        acoustic_backbone=backbone,
        feature_dim=feature_dim,
        use_acoustic=use_acoustic,
        fusion_type=fusion_type
    )
    
    return model


if __name__ == "__main__":
    # Test the fusion model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test without acoustic
    print("\n🧪 Testing fusion model without acoustic...")
    model = create_fusion_model(num_classes=5, backbone='resnet18', use_acoustic=False)
    model.to(device)
    
    rgb_input = torch.randn(4, 3, 224, 224).to(device)
    thermal_input = torch.randn(4, 1, 224, 224).to(device)
    
    output = model(rgb_input, thermal_input)
    print(f"Output shape: {output.shape}")
    
    # Test with attention weights
    output, attention = model(rgb_input, thermal_input, return_attention=True)
    print(f"Attention weights shape: {attention.shape}")
    
    # Test with acoustic
    print("\n🧪 Testing fusion model with acoustic...")
    model_acoustic = create_fusion_model(num_classes=5, backbone='resnet18', use_acoustic=True)
    model_acoustic.to(device)
    
    acoustic_input = torch.randn(4, 1, 224, 224).to(device)
    
    output = model_acoustic(rgb_input, thermal_input, acoustic_input)
    print(f"Output shape with acoustic: {output.shape}")
    
    print("✅ Fusion model test passed!") 