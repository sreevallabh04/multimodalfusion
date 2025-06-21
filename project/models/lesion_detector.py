"""
Lesion detector CNN model for detecting mango leaf diseases.
This model will be used to generate pseudo-thermal maps on fruit images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional
import numpy as np


class LesionDetector(nn.Module):
    """
    CNN model for detecting lesions in mango leaf images.
    Outputs probability of disease presence which is used to simulate thermal maps.
    """
    
    def __init__(self, 
                 num_classes: int = 8,
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super(LesionDetector, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Remove global pooling
        )
        
        # Get backbone feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_out = self.backbone(dummy_input)
            self.backbone_features = backbone_out.shape[1]
            self.feature_map_size = backbone_out.shape[-1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Attention module for generating heatmaps
        self.attention = nn.Sequential(
            nn.Conv2d(self.backbone_features, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in [self.classifier, self.attention]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the lesion detector.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            return_attention: Whether to return attention maps
            
        Returns:
            Tuple of (class_logits, attention_map) if return_attention=True
            Otherwise just class_logits
        """
        # Extract features using backbone
        features = self.backbone(x)  # Shape: (batch_size, backbone_features, H, W)
        
        # Classification
        class_logits = self.classifier(features)
        
        if return_attention:
            # Generate attention map for lesion localization
            attention_map = self.attention(features)  # Shape: (batch_size, 1, H, W)
            # Upsample attention map to input size
            attention_map = F.interpolate(attention_map, size=(224, 224), mode='bilinear', align_corners=False)
            return class_logits, attention_map
        
        return class_logits
    
    def get_lesion_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability of lesion presence for thermal map generation.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Lesion probability map of shape (batch_size, 1, 224, 224)
        """
        self.eval()
        with torch.no_grad():
            _, attention_map = self.forward(x, return_attention=True)
            
            # Convert to disease probability (1 - healthy probability)
            class_logits = self.forward(x, return_attention=False)
            class_probs = F.softmax(class_logits, dim=1)
            
            # Assume index 0 is healthy class
            healthy_prob = class_probs[:, 0:1]  # Shape: (batch_size, 1)
            disease_prob = 1 - healthy_prob  # Disease probability
            
            # Combine spatial attention with disease probability
            disease_prob = disease_prob.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, 1, 1, 1)
            lesion_map = attention_map * disease_prob
            
            return lesion_map


class LesionDetectorTrainer:
    """Trainer class for the lesion detector model."""
    
    def __init__(self, 
                 model: LesionDetector,
                 device: torch.device,
                 class_names: list):
        self.model = model
        self.device = device
        self.class_names = class_names
        
        # Move model to device
        self.model.to(device)
        
        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        self.attention_criterion = nn.MSELoss()
    
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            class_logits, attention_maps = self.model(images, return_attention=True)
            
            # Classification loss
            class_loss = self.criterion(class_logits, labels)
            
            # Total loss
            loss = class_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
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
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                class_logits = self.model(images, return_attention=False)
                
                # Loss
                loss = self.criterion(class_logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = class_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def save_model(self, path: str, epoch: int, metrics: dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'class_names': self.class_names
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {path}")
        return checkpoint


def create_lesion_detector(num_classes: int = 8, 
                          backbone: str = 'resnet18',
                          pretrained: bool = True) -> LesionDetector:
    """
    Factory function to create a lesion detector model.
    
    Args:
        num_classes: Number of leaf disease classes
        backbone: Backbone architecture name
        pretrained: Whether to use pretrained weights
        
    Returns:
        LesionDetector model
    """
    model = LesionDetector(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_lesion_detector(num_classes=8)
    model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test classification
    class_logits = model(dummy_input)
    print(f"Class logits shape: {class_logits.shape}")
    
    # Test attention
    class_logits, attention_map = model(dummy_input, return_attention=True)
    print(f"Attention map shape: {attention_map.shape}")
    
    # Test lesion probability
    lesion_prob = model.get_lesion_probability(dummy_input)
    print(f"Lesion probability shape: {lesion_prob.shape}")
    
    print("✅ Lesion detector model test passed!") 