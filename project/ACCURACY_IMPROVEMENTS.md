# 📈 Accuracy Improvement Strategies

**Current Performance**: 88.89% (Fusion) vs 82.54% (RGB-only)  
**Target**: 93-96% accuracy with systematic improvements

## 🎯 Quick Wins (Expected +3-5% accuracy)

### 1. **Extended Training** ⏰
```bash
# Current: Only 3 epochs - severely undertrained!
python train.py --epochs 50 --batch_size 32

# Even better: Full training
python train.py --epochs 100 --batch_size 32 --patience 20
```

**Current Issue**: Models trained for only 3 epochs are barely past random initialization.  
**Expected Gain**: +2-4% accuracy just from proper training duration.

### 2. **Larger Model Architecture** 🏗️
```bash
# Current: ResNet18 (11M parameters)
# Upgrade to ResNet50 (26M parameters)
python train.py --backbone resnet50 --epochs 50

# Or try EfficientNet (better accuracy/parameter ratio)
python train.py --backbone efficientnet_b1 --epochs 50
```

**Expected Gain**: +1-3% accuracy from increased model capacity.

### 3. **Optimized Hyperparameters** ⚙️
```bash
# Better learning rate schedule
python train.py \
  --learning_rate 0.0005 \
  --scheduler cosine \
  --batch_size 64 \
  --weight_decay 0.00001
```

## 🚀 Advanced Improvements (Expected +2-4% accuracy)

### 4. **Enhanced Data Augmentation**
Create `scripts/advanced_augmentation.py`:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_advanced_train_transforms(image_size=224):
    """Enhanced augmentation for better generalization."""
    return A.Compose([
        # Geometric augmentations
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7
        ),
        
        # Color augmentations
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
        ),
        A.RandomBrightnessContrast(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8
        ),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        
        # Advanced augmentations
        A.CLAHE(clip_limit=2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RandomToneCurve(p=0.3),
        
        # Cutout/Erasing
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, 
            min_holes=1, fill_value=0, p=0.3
        ),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms(image_size=224):
    """Test-time augmentation transforms."""
    return [
        A.Compose([A.Normalize(), ToTensorV2()]),  # Original
        A.Compose([A.HorizontalFlip(), A.Normalize(), ToTensorV2()]),
        A.Compose([A.VerticalFlip(), A.Normalize(), ToTensorV2()]),
        A.Compose([A.RandomRotate90(), A.Normalize(), ToTensorV2()]),
    ]
```

### 5. **Improved Fusion Architecture**
Create `models/advanced_fusion.py`:

```python
class CrossModalTransformer(nn.Module):
    """Advanced transformer-based fusion."""
    
    def __init__(self, feature_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        
        # Modality embeddings
        self.rgb_embed = nn.Linear(feature_dim, feature_dim)
        self.thermal_embed = nn.Linear(feature_dim, feature_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, rgb_features, thermal_features):
        # Embed features
        rgb_embed = self.rgb_embed(rgb_features).unsqueeze(1)  # (B, 1, D)
        thermal_embed = self.thermal_embed(thermal_features).unsqueeze(1)  # (B, 1, D)
        
        # Concatenate for cross-attention
        features = torch.cat([rgb_embed, thermal_embed], dim=1)  # (B, 2, D)
        
        # Apply transformer
        fused = self.transformer(features)  # (B, 2, D)
        
        # Global pooling and projection
        pooled = fused.mean(dim=1)  # (B, D)
        output = self.output_proj(pooled)
        
        return output
```

### 6. **Class Balancing & Weighted Loss**
```python
# In train.py, add class weighting
def get_class_weights(dataloader):
    """Calculate inverse frequency weights."""
    class_counts = torch.zeros(5)
    for _, labels in dataloader:
        for label in labels:
            class_counts[label] += 1
    
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    return weights

# Use weighted loss
class_weights = get_class_weights(train_loader)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

## 🔬 Advanced Techniques (Expected +3-6% accuracy)

### 7. **Ensemble Methods**
```python
class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for better accuracy."""
    
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
    
    def forward(self, *inputs):
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(*inputs)
                outputs.append(F.softmax(output, dim=1))
        
        # Weighted average
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += self.weights[i] * output
        
        return ensemble_output

# Train multiple models with different configurations
models = [
    # ResNet50 + ResNet18
    train_model(backbone='resnet50', fusion_type='attention'),
    train_model(backbone='resnet18', fusion_type='transformer'),
    train_model(backbone='efficientnet_b1', fusion_type='attention'),
]
ensemble = ModelEnsemble(models, weights=[0.4, 0.3, 0.3])
```

### 8. **Progressive Training Strategy**
```bash
# Stage 1: Train with smaller images (faster)
python train.py --image_size 192 --epochs 30 --batch_size 64

# Stage 2: Fine-tune with full resolution
python train.py --image_size 224 --epochs 20 --batch_size 32 \
  --pretrained_model "models/checkpoints/stage1_best.pth" \
  --learning_rate 0.0001

# Stage 3: Final fine-tuning with TTA
python train.py --image_size 256 --epochs 10 --batch_size 16 \
  --pretrained_model "models/checkpoints/stage2_best.pth" \
  --learning_rate 0.00005 --use_tta
```

### 9. **Advanced Thermal Simulation**
Improve `scripts/simulate_thermal.py`:

```python
class AdvancedThermalSimulator:
    """Enhanced thermal simulation with physical modeling."""
    
    def __init__(self):
        self.disease_thermal_profiles = {
            'healthy': {'base_temp': 0.3, 'variation': 0.1},
            'anthracnose': {'base_temp': 0.7, 'variation': 0.2},
            'alternaria': {'base_temp': 0.8, 'variation': 0.15},
            'black_mould': {'base_temp': 0.9, 'variation': 0.25},
            'stem_rot': {'base_temp': 0.85, 'variation': 0.2}
        }
    
    def simulate_thermal_with_physics(self, rgb_image, lesion_attention):
        """Simulate thermal signature based on disease physics."""
        # Extract fruit contour for realistic thermal modeling
        fruit_mask = self.segment_fruit(rgb_image)
        
        # Simulate heat diffusion from lesion centers
        thermal_map = self.heat_diffusion_model(lesion_attention, fruit_mask)
        
        # Add environmental factors
        thermal_map = self.add_environmental_effects(thermal_map)
        
        return thermal_map
```

### 10. **Knowledge Distillation**
```python
class KnowledgeDistillationLoss(nn.Module):
    """Distill knowledge from ensemble to single model."""
    
    def __init__(self, alpha=0.7, temperature=4):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Standard classification loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Knowledge distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss
```

## 📊 Training Optimization

### 11. **Mixed Precision Training**
```python
# In train.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop with mixed precision
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(rgb_images, thermal_images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 12. **Curriculum Learning**
```python
class CurriculumScheduler:
    """Gradually increase training difficulty."""
    
    def __init__(self, start_ratio=0.5, end_ratio=1.0, total_epochs=100):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.total_epochs = total_epochs
    
    def get_difficulty_ratio(self, epoch):
        """Return fraction of hardest samples to use."""
        progress = epoch / self.total_epochs
        return self.start_ratio + progress * (self.end_ratio - self.start_ratio)
```

## 🎯 Systematic Improvement Plan

### Phase 1: Quick Wins (1-2 days)
```bash
# 1. Extended training with better hyperparameters
python train.py --epochs 100 --batch_size 32 --learning_rate 0.0005 \
  --scheduler cosine --backbone resnet50

# Expected: 91-93% accuracy
```

### Phase 2: Architecture Improvements (2-3 days)
```bash
# 2. Advanced fusion with larger models
python train.py --epochs 100 --backbone efficientnet_b1 \
  --fusion_type transformer --feature_dim 768

# Expected: 92-94% accuracy
```

### Phase 3: Advanced Techniques (3-5 days)
```bash
# 3. Ensemble + Advanced augmentation
python train_ensemble.py --num_models 3 --use_advanced_aug \
  --use_tta --epochs 50

# Expected: 94-96% accuracy
```

## 📈 Performance Targets by Improvement

| Improvement | Expected Accuracy | Implementation Time |
|-------------|------------------|-------------------|
| **Current** | 88.89% | - |
| Extended Training (50+ epochs) | 91-92% | 2-4 hours |
| ResNet50 backbone | 92-93% | 4-6 hours |
| Advanced augmentation | 93-94% | 1 day |
| Transformer fusion | 94-95% | 2 days |
| Ensemble methods | 95-96% | 3 days |
| Full optimization | **96-98%** | 1 week |

## 🚀 Immediate Action Plan

### Step 1: Quick Training Fix
```bash
# Stop undertraining - run for proper epochs
cd project
python train.py --train_mode both --epochs 50 --batch_size 32 \
  --backbone resnet50 --learning_rate 0.0005
```

### Step 2: Implement Advanced Augmentation
```bash
# Install additional dependencies
pip install albumentations timm efficientnet-pytorch

# Run with enhanced augmentation
python train.py --epochs 100 --use_advanced_aug --backbone efficientnet_b1
```

### Step 3: Ensemble Training
```bash
# Train multiple models for ensemble
python train.py --backbone resnet50 --epochs 80 --experiment_name "resnet50_v1"
python train.py --backbone efficientnet_b1 --epochs 80 --experiment_name "effnet_v1"
python train.py --backbone resnet34 --fusion_type transformer --epochs 80
```

## 🔧 Monitoring Improvements

### Key Metrics to Track:
- **Validation accuracy** (target: >94%)
- **Per-class F1 scores** (target: all >90%)
- **Confusion matrix** (minimize off-diagonal)
- **Training stability** (smooth convergence)

### Expected Weak Points to Address:
1. **Anthracnose class** (currently 68% recall) - needs better features
2. **Healthy vs Disease confusion** - improve thermal simulation
3. **Class imbalance** - use weighted loss or resampling

## 💡 Why These Improvements Work

1. **Extended Training**: Current 3 epochs is severely undertrained
2. **Larger Models**: More parameters = better feature learning capacity  
3. **Better Fusion**: Transformer attention > simple concatenation
4. **Advanced Augmentation**: Improves generalization significantly
5. **Ensemble**: Combines multiple perspectives for robustness
6. **Physics-based Thermal**: More realistic multi-modal features

## 🎯 Expected Final Performance

With systematic implementation of these improvements:
- **Single Model**: 94-96% accuracy
- **Ensemble**: 96-98% accuracy  
- **Per-class F1**: >92% for all classes
- **Practical Impact**: Production-ready accuracy for real deployment

The current 88.89% can realistically be improved to **95%+** with proper implementation of these strategies. 