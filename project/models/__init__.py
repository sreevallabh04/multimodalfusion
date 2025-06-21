"""
Models package for multi-modal mango disease classification.
"""

from .lesion_detector import LesionDetector, create_lesion_detector
from .rgb_branch import RGBBranch, create_rgb_branch
from .fusion_model import MultiModalFusionModel, create_fusion_model

__all__ = [
    'LesionDetector', 'create_lesion_detector',
    'RGBBranch', 'create_rgb_branch', 
    'MultiModalFusionModel', 'create_fusion_model'
] 