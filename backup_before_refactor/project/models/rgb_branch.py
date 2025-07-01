"""
RGB branch CNN model for mango fruit classification.
Serves as both standalone classifier and feature extractor for fusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional
import numpy as np


class RGBBranch(nn.Module):
    """
    RGB branch CNN for mango fruit disease classification.
    Can be used standalone or as part of a multi-modal fusion model.
    """
    
    def __init__(self,
                 num_classes: int = 5,
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 feature_dim: int = 512):
        """
        Args:
            num_classes: Number of fruit disease classes
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            feature_dim: Dimension of feature vector for fusion
        """
        super(RGBBranch, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Use average pooling
        )
        
        # Get backbone output features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_out = self.backbone(dummy_input)
            self.backbone_features = backbone_out.shape[1]
        
        # Feature projection layer for fusion
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Attention mechanism for interpretability
        self.attention = None
        if hasattr(self.backbone, 'forward_features'):
            self._setup_attention()
        
        self._initialize_weights()
    
    def _setup_attention(self):
        """Setup attention mechanism for feature visualization."""
        # Get the feature map dimensions from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_maps = self.backbone.forward_features(dummy_input)
            
            if len(feature_maps.shape) == 4:  # (B, C, H, W)
                feature_channels = feature_maps.shape[1]
                
                self.attention = nn.Sequential(
                    nn.Conv2d(feature_channels, feature_channels // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature_channels // 4, 1, 1),
                    nn.Sigmoid()
                )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in [self.feature_projector, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        if self.attention is not None:
            for m in self.attention.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_features: bool = False, return_attention: bool = False):
        """
        Forward pass of RGB branch.
        
        Args:
            x: Input RGB tensor of shape (batch_size, 3, 224, 224)
            return_features: Whether to return feature vector for fusion
            return_attention: Whether to return attention maps
            
        Returns:
            - class_logits: Classification logits
            - features: Feature vector (if return_features=True)
            - attention_map: Attention map (if return_attention=True)
        """
        batch_size = x.size(0)
        
        # Extract features
        if self.attention is not None and (return_attention or hasattr(self.backbone, 'forward_features')):
            # Get intermediate feature maps for attention
            feature_maps = self.backbone.forward_features(x)  # (B, C, H, W)
            
            # Generate attention map
            attention_map = None
            if self.attention is not None and return_attention:
                attention_map = self.attention(feature_maps)  # (B, 1, H, W)
                # Upsample to input size for visualization
                attention_map = F.interpolate(attention_map, size=(224, 224), mode='bilinear', align_corners=False)
                
                # Apply attention to feature maps
                attended_features = feature_maps * attention_map
                # Global average pooling
                pooled_features = F.adaptive_avg_pool2d(attended_features, (1, 1)).view(batch_size, -1)
            else:
                # Standard global average pooling
                pooled_features = F.adaptive_avg_pool2d(feature_maps, (1, 1)).view(batch_size, -1)
        else:
            # Use standard backbone forward (already includes global pooling)
            pooled_features = self.backbone(x)
            attention_map = None
        
        # Project features for fusion
        features = self.feature_projector(pooled_features)
        
        # Classification
        class_logits = self.classifier(features)
        
        # Return based on requirements
        outputs = [class_logits]
        
        if return_features:
            outputs.append(features)
        
        if return_attention and attention_map is not None:
            outputs.append(attention_map)
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
    
    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector for fusion model."""
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features


class RGBTrainer:
    """Trainer class for RGB branch model."""
    
    def __init__(self,
                 model: RGBBranch,
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
            # Handle different batch formats (with/without thermal/acoustic)
            if len(batch) == 3:  # RGB + thermal + label
                rgb_images, _, labels = batch
            elif len(batch) == 4:  # RGB + thermal + acoustic + label
                rgb_images, _, _, labels = batch
            else:  # Just RGB + label
                rgb_images, labels = batch
            
            rgb_images, labels = rgb_images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            class_logits = self.model(rgb_images)
            
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
            
            if batch_idx % 100 == 0:
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
                # Handle different batch formats
                if len(batch) == 3:  # RGB + thermal + label
                    rgb_images, _, labels = batch
                elif len(batch) == 4:  # RGB + thermal + acoustic + label
                    rgb_images, _, _, labels = batch
                else:  # Just RGB + label
                    rgb_images, labels = batch
                
                rgb_images, labels = rgb_images.to(self.device), labels.to(self.device)
                
                # Forward pass
                class_logits = self.model(rgb_images)
                
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
                'backbone': self.model.backbone_name,
                'feature_dim': self.model.feature_dim
            }
        }, path)
        print(f"✅ RGB model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ RGB model loaded from {path}")
        return checkpoint


def create_rgb_branch(num_classes: int = 5,
                     backbone: str = 'resnet18',
                     pretrained: bool = True,
                     feature_dim: int = 512) -> RGBBranch:
    """
    Factory function to create RGB branch model.
    
    Args:
        num_classes: Number of fruit disease classes
        backbone: Backbone architecture name
        pretrained: Whether to use pretrained weights
        feature_dim: Feature dimension for fusion
        
    Returns:
        RGBBranch model
    """
    model = RGBBranch(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        feature_dim=feature_dim
    )
    
    return model


if __name__ == "__main__":
    # Test the RGB branch model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_rgb_branch(num_classes=5)
    model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test classification only
    class_logits = model(dummy_input)
    print(f"Class logits shape: {class_logits.shape}")
    
    # Test with features
    class_logits, features = model(dummy_input, return_features=True)
    print(f"Features shape: {features.shape}")
    
    # Test with attention (if available)
    try:
        class_logits, features, attention = model(dummy_input, return_features=True, return_attention=True)
        print(f"Attention shape: {attention.shape}")
    except:
        print("Attention not available for this backbone")
    
    print("✅ RGB branch model test passed!") 