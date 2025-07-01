"""
Ensemble model for maximum accuracy in mango disease classification.
Combines multiple trained models for robust predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from .rgb_branch import RGBBranch, create_rgb_branch
from .fusion_model import AdvancedMultiModalFusionModel


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that learns optimal weights for combining model predictions.
    """
    
    def __init__(self, num_models: int, num_classes: int = 5):
        super(AdaptiveEnsemble, self).__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
        # Confidence-based weighting network
        self.confidence_network = nn.Sequential(
            nn.Linear(num_classes * num_models, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_models),
            nn.Softmax(dim=1)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, model_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine predictions from multiple models adaptively.
        
        Args:
            model_predictions: List of prediction tensors from different models
            
        Returns:
            Tuple of (ensemble_prediction, confidence_weights)
        """
        batch_size = model_predictions[0].size(0)
        
        # Stack predictions
        stacked_predictions = torch.stack(model_predictions, dim=1)  # (B, num_models, num_classes)
        
        # Convert to probabilities
        probabilities = [F.softmax(pred / self.temperature, dim=1) for pred in model_predictions]
        stacked_probs = torch.stack(probabilities, dim=1)
        
        # Calculate confidence-based weights
        flattened_probs = stacked_probs.view(batch_size, -1)
        confidence_weights = self.confidence_network(flattened_probs)
        
        # Combine with learned ensemble weights
        final_weights = confidence_weights * self.ensemble_weights.unsqueeze(0)
        final_weights = F.softmax(final_weights, dim=1)
        
        # Weighted ensemble prediction
        ensemble_prob = torch.sum(stacked_probs * final_weights.unsqueeze(2), dim=1)
        ensemble_logits = torch.log(ensemble_prob + 1e-8)
        
        return ensemble_logits, final_weights


class MultiModelEnsemble:
    """
    Ensemble system that combines multiple trained models for maximum accuracy.
    """
    
    def __init__(self, 
                 model_configs: List[Dict],
                 device: torch.device,
                 num_classes: int = 5,
                 use_adaptive_weights: bool = True):
        """
        Args:
            model_configs: List of model configuration dictionaries
            device: Compute device
            num_classes: Number of output classes
            use_adaptive_weights: Whether to use adaptive ensemble weighting
        """
        self.device = device
        self.num_classes = num_classes
        self.use_adaptive_weights = use_adaptive_weights
        self.models = []
        self.model_types = []
        
        # Load all models
        for config in model_configs:
            model = self._load_model(config)
            if model is not None:
                self.models.append(model)
                self.model_types.append(config['type'])
        
        print(f"✅ Loaded {len(self.models)} models for ensemble")
        
        # Initialize adaptive ensemble if requested
        if self.use_adaptive_weights and len(self.models) > 1:
            self.adaptive_ensemble = AdaptiveEnsemble(
                num_models=len(self.models),
                num_classes=num_classes
            ).to(device)
        else:
            self.adaptive_ensemble = None
        
        # Simple ensemble weights (uniform by default)
        self.ensemble_weights = torch.ones(len(self.models)) / len(self.models)
    
    def _load_model(self, config: Dict) -> Optional[nn.Module]:
        """Load a single model from configuration."""
        try:
            model_path = config['model_path']
            model_type = config['type']
            
            if not Path(model_path).exists():
                print(f"⚠️  Model file not found: {model_path}")
                return None
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if model_type == 'rgb':
                # RGB model
                model = create_rgb_branch(
                    num_classes=self.num_classes,
                    backbone=config.get('backbone', 'resnet50'),
                    pretrained=False,
                    feature_dim=config.get('feature_dim', 512)
                )
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
            elif model_type == 'fusion':
                # Fusion model
                model = AdvancedMultiModalFusionModel(
                    num_classes=self.num_classes,
                    rgb_backbone=config.get('rgb_backbone', 'convnext_tiny'),
                    thermal_backbone=config.get('thermal_backbone', 'convnext_tiny'),
                    feature_dim=config.get('feature_dim', 512),
                    use_acoustic=config.get('use_acoustic', False),
                    fusion_type=config.get('fusion_type', 'advanced_attention')
                )
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                print(f"⚠️  Unknown model type: {model_type}")
                return None
            
            model.to(self.device)
            model.eval()
            
            print(f"✅ Loaded {model_type} model: {Path(model_path).name}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model {config.get('model_path', 'unknown')}: {e}")
            return None
    
    def predict(self, 
               rgb_images: torch.Tensor,
               thermal_images: Optional[torch.Tensor] = None,
               acoustic_images: Optional[torch.Tensor] = None,
               return_individual: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Make ensemble predictions.
        
        Args:
            rgb_images: RGB input tensor
            thermal_images: Thermal input tensor (for fusion models)
            acoustic_images: Acoustic input tensor (for fusion models)
            return_individual: Whether to return individual model predictions
            
        Returns:
            Tuple of (ensemble_predictions, individual_predictions_dict)
        """
        individual_predictions = []
        individual_probs = []
        model_names = []
        
        with torch.no_grad():
            for i, (model, model_type) in enumerate(zip(self.models, self.model_types)):
                try:
                    if model_type == 'rgb':
                        # RGB-only model
                        predictions = model(rgb_images)
                    elif model_type == 'fusion':
                        # Multi-modal fusion model
                        if thermal_images is not None:
                            predictions = model(rgb_images, thermal_images, acoustic_images)
                        else:
                            # If no thermal images, skip fusion models
                            continue
                    else:
                        continue
                    
                    individual_predictions.append(predictions)
                    individual_probs.append(F.softmax(predictions, dim=1))
                    model_names.append(f"{model_type}_{i}")
                    
                except Exception as e:
                    print(f"⚠️  Error with model {i} ({model_type}): {e}")
                    continue
        
        if not individual_predictions:
            raise ValueError("No models produced valid predictions")
        
        # Ensemble combination
        if self.adaptive_ensemble is not None and len(individual_predictions) > 1:
            # Adaptive ensemble
            ensemble_logits, adaptive_weights = self.adaptive_ensemble(individual_predictions)
        else:
            # Simple weighted average
            stacked_probs = torch.stack(individual_probs, dim=0)
            weights = self.ensemble_weights[:len(individual_probs)].to(self.device)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Weighted average
            ensemble_prob = torch.sum(stacked_probs * weights.view(-1, 1, 1), dim=0)
            ensemble_logits = torch.log(ensemble_prob + 1e-8)
            adaptive_weights = weights.unsqueeze(0).repeat(ensemble_prob.size(0), 1)
        
        # Prepare return values
        result = ensemble_logits
        
        if return_individual:
            individual_dict = {
                'predictions': individual_predictions,
                'probabilities': individual_probs,
                'model_names': model_names,
                'weights': adaptive_weights if self.adaptive_ensemble else weights
            }
            return result, individual_dict
        
        return result, None
    
    def predict_with_uncertainty(self, 
                                rgb_images: torch.Tensor,
                                thermal_images: Optional[torch.Tensor] = None,
                                acoustic_images: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Returns:
            Tuple of (predictions, uncertainty, confidence)
        """
        predictions, individual_dict = self.predict(
            rgb_images, thermal_images, acoustic_images, return_individual=True
        )
        
        if individual_dict is None:
            # Single model case
            probs = F.softmax(predictions, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            uncertainty = 1.0 - confidence
            return predictions, uncertainty, confidence
        
        # Multi-model case
        individual_probs = torch.stack(individual_dict['probabilities'], dim=0)
        
        # Calculate disagreement as uncertainty measure
        mean_probs = individual_probs.mean(dim=0)
        disagreement = torch.var(individual_probs, dim=0).sum(dim=1)
        
        # Confidence from ensemble
        ensemble_probs = F.softmax(predictions, dim=1)
        confidence = torch.max(ensemble_probs, dim=1)[0]
        
        # Combined uncertainty
        uncertainty = disagreement + (1.0 - confidence)
        
        return predictions, uncertainty, confidence
    
    def calibrate_temperature(self, 
                             dataloader: torch.utils.data.DataLoader,
                             max_iter: int = 100) -> float:
        """
        Calibrate temperature for better probability estimates.
        
        Returns:
            Optimal temperature value
        """
        if self.adaptive_ensemble is None:
            print("⚠️  Temperature calibration requires adaptive ensemble")
            return 1.0
        
        # Collect validation predictions
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    rgb_images, thermal_images, labels = batch
                    acoustic_images = None
                elif len(batch) == 4:
                    rgb_images, thermal_images, acoustic_images, labels = batch
                else:
                    continue
                
                rgb_images = rgb_images.to(self.device)
                if thermal_images is not None:
                    thermal_images = thermal_images.to(self.device)
                if acoustic_images is not None:
                    acoustic_images = acoustic_images.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.predict(rgb_images, thermal_images, acoustic_images)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1).to(self.device))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            loss = F.cross_entropy(all_logits / temperature, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        optimal_temp = temperature.item()
        print(f"🌡️  Optimal temperature: {optimal_temp:.3f}")
        
        # Update ensemble temperature
        self.adaptive_ensemble.temperature.data = temperature.data
        
        return optimal_temp
    
    def save_ensemble_config(self, config_path: str):
        """Save ensemble configuration for reproducibility."""
        config = {
            'num_models': len(self.models),
            'model_types': self.model_types,
            'use_adaptive_weights': self.use_adaptive_weights,
            'ensemble_weights': self.ensemble_weights.tolist() if isinstance(self.ensemble_weights, torch.Tensor) else self.ensemble_weights,
            'num_classes': self.num_classes
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Ensemble configuration saved to {config_path}")


def create_ensemble_from_checkpoints(checkpoint_dir: str, 
                                    device: torch.device,
                                    use_adaptive_weights: bool = True) -> MultiModelEnsemble:
    """
    Create ensemble from all available model checkpoints.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        device: Compute device
        use_adaptive_weights: Whether to use adaptive ensemble weighting
        
    Returns:
        MultiModelEnsemble instance
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Find all model checkpoints
    model_configs = []
    
    # Look for RGB models
    rgb_models = list(checkpoint_path.glob("*rgb*_best.pth"))
    for model_path in rgb_models:
        config = {
            'model_path': str(model_path),
            'type': 'rgb',
            'backbone': 'convnext_tiny',  # Update based on your models
            'feature_dim': 512
        }
        model_configs.append(config)
    
    # Look for fusion models
    fusion_models = list(checkpoint_path.glob("*fusion*_best.pth"))
    for model_path in fusion_models:
        config = {
            'model_path': str(model_path),
            'type': 'fusion',
            'rgb_backbone': 'convnext_tiny',
            'thermal_backbone': 'convnext_tiny',
            'feature_dim': 512,
            'use_acoustic': 'acoustic' in model_path.name,
            'fusion_type': 'advanced_attention'
        }
        model_configs.append(config)
    
    print(f"🔍 Found {len(model_configs)} model checkpoints")
    
    # Create ensemble
    ensemble = MultiModelEnsemble(
        model_configs=model_configs,
        device=device,
        use_adaptive_weights=use_adaptive_weights
    )
    
    return ensemble


def evaluate_ensemble(ensemble: MultiModelEnsemble,
                     dataloader: torch.utils.data.DataLoader,
                     class_names: List[str]) -> Dict:
    """
    Evaluate ensemble performance.
    
    Args:
        ensemble: MultiModelEnsemble instance
        dataloader: Test dataloader
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    ensemble_predictions = []
    true_labels = []
    uncertainties = []
    confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                rgb_images, thermal_images, labels = batch
                acoustic_images = None
            elif len(batch) == 4:
                rgb_images, thermal_images, acoustic_images, labels = batch
            else:
                continue
            
            rgb_images = rgb_images.to(ensemble.device)
            if thermal_images is not None:
                thermal_images = thermal_images.to(ensemble.device)
            if acoustic_images is not None:
                acoustic_images = acoustic_images.to(ensemble.device)
            labels = labels.to(ensemble.device)
            
            # Get ensemble predictions with uncertainty
            pred_logits, uncertainty, confidence = ensemble.predict_with_uncertainty(
                rgb_images, thermal_images, acoustic_images
            )
            
            predictions = pred_logits.argmax(dim=1)
            ensemble_predictions.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            uncertainties.extend(uncertainty.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    
    accuracy = accuracy_score(true_labels, ensemble_predictions)
    f1 = f1_score(true_labels, ensemble_predictions, average='weighted')
    precision = precision_score(true_labels, ensemble_predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, ensemble_predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    report = classification_report(true_labels, ensemble_predictions, target_names=class_names, output_dict=True)
    
    results = {
        'accuracy': accuracy * 100,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': report,
        'mean_uncertainty': np.mean(uncertainties),
        'mean_confidence': np.mean(confidences),
        'predictions': ensemble_predictions,
        'true_labels': true_labels,
        'uncertainties': uncertainties,
        'confidences': confidences
    }
    
    return results


if __name__ == "__main__":
    # Test ensemble creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ensemble on {device}")
    
    print("✅ Ensemble module created successfully!") 