# 🚀 Multi-Modal Mango Disease Classification - Setup Guide

Complete step-by-step instructions to get the project running on your system.

## 📋 Table of Contents
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Detailed Installation](#-detailed-installation)
- [Data Setup](#-data-setup)
- [Running the Pipeline](#-running-the-pipeline)
- [Usage Examples](#-usage-examples)
- [Troubleshooting](#-troubleshooting)
- [Advanced Configuration](#-advanced-configuration)

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional but recommended (CUDA-compatible NVIDIA GPU)

### Recommended Requirements
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Storage**: 20GB+ free space (for datasets and models)
- **CPU**: Multi-core processor (4+ cores recommended)

---

## ⚡ Quick Start

For users who want to get running immediately:

### 1. Clone and Navigate
```bash
# Download the project (if from Git)
git clone <repository-url>
cd multimodalfusion/project

# Or if you have the files locally
cd path/to/multimodalfusion/project
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Run the test suite
python test_pipeline.py
```

### 4. Quick Demo (if models exist)
```bash
# Run inference demo
python demo_inference.py \
  --rgb_model "models/checkpoints/rgb_baseline_resnet18_*_best.pth" \
  --fusion_model "models/checkpoints/fusion_rgb_thermal_*_best.pth" \
  --image "data/processed/fruit/test/Healthy/healthy_001.jpg"
```

---

## 🔧 Detailed Installation

### Step 1: Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create new environment
conda create -n mango-classification python=3.10

# Activate environment
conda activate mango-classification

# Install PyTorch (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install PyTorch (GPU version - if you have CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Option B: Using pip + venv
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Verify Installation
```bash
# Run comprehensive tests
python test_pipeline.py

# Expected output should show all tests passing:
# ✅ Imports..................... PASS
# ✅ Model Creation.............. PASS
# ✅ Thermal Simulation.......... PASS
# ✅ Dataloader.................. PASS
# ✅ Training Components......... PASS
# ✅ Evaluation Components....... PASS
```

---

## 📁 Data Setup

### Dataset Structure
Your data should be organized as follows:
```
project/
├── data/
│   ├── fruit/
│   │   └── SenMangoFruitDDS_bgremoved/
│   │       ├── Healthy/
│   │       ├── Anthracnose/
│   │       ├── Alternaria/
│   │       ├── Black Mould Rot/
│   │       └── Stem and Rot/
│   └── leaf/
│       ├── Healthy/
│       ├── Anthracnose/
│       ├── Bacterial Canker/
│       ├── Cutting Weevil/
│       ├── Die Back/
│       ├── Gall Midge/
│       ├── Powdery Mildew/
│       └── Sooty Mould/
```

### Option 1: Using Provided Datasets
If you already have the MangoFruitDDS and MangoLeafBD datasets:

1. **Place MangoFruitDDS** in `data/fruit/SenMangoFruitDDS_bgremoved/`
2. **Place MangoLeafBD** in `data/leaf/`

### Option 2: Download Datasets
```bash
# Download MangoFruitDDS dataset
# [Provide download link or instructions]

# Download MangoLeafBD dataset  
# [Provide download link or instructions]

# Extract and organize datasets
# Follow the directory structure above
```

### Data Preprocessing
```bash
# Process and split datasets (70/15/15 train/val/test)
python scripts/preprocess.py

# Expected output:
# ✅ Processed 838 fruit images
# ✅ Processed 4000 leaf images
# ✅ Created train/val/test splits
```

---

## 🏃‍♂️ Running the Pipeline

### Complete Training Pipeline

#### Step 1: Data Preprocessing (if not done)
```bash
python scripts/preprocess.py
```

#### Step 2: Generate Thermal Maps
```bash
# Generate simulated thermal maps
python scripts/simulate_thermal.py \
  --fruit_data data/processed/fruit \
  --output data/thermal \
  --batch_size 16
```

#### Step 3: Train Models
```bash
# Train both RGB baseline and fusion models
python train.py \
  --train_mode both \
  --epochs 50 \
  --batch_size 32 \
  --backbone resnet18 \
  --fusion_type attention

# For quick testing (reduced epochs):
python train.py --train_mode both --epochs 5 --batch_size 16
```

#### Step 4: Evaluate Models
```bash
# Run comprehensive evaluation
python evaluate.py \
  --rgb_model_path "models/checkpoints/rgb_baseline_*_best.pth" \
  --fusion_model_path "models/checkpoints/fusion_*_best.pth" \
  --output_dir evaluation_results
```

### Individual Components

#### Train Only RGB Model
```bash
python train.py --train_mode rgb_only --epochs 30
```

#### Train Only Fusion Model
```bash
python train.py --train_mode fusion_only --epochs 30
```

#### Generate Only Thermal Maps
```bash
python scripts/simulate_thermal.py --output data/thermal --no_viz
```

---

## 🎮 Usage Examples

### Basic Inference
```bash
# Classify a single image with RGB model
python demo_inference.py \
  --rgb_model "models/checkpoints/rgb_baseline_resnet18_20250621_150726_best.pth" \
  --image "path/to/your/mango_image.jpg" \
  --output "result.png"
```

### Multi-Modal Comparison
```bash
# Compare RGB vs Fusion models
python demo_inference.py \
  --rgb_model "models/checkpoints/rgb_baseline_*_best.pth" \
  --fusion_model "models/checkpoints/fusion_*_best.pth" \
  --image "data/processed/fruit/test/Healthy/healthy_001.jpg" \
  --output "comparison.png"
```

### Batch Processing
```bash
# Process multiple images
for image in data/processed/fruit/test/*/*.jpg; do
  python demo_inference.py \
    --rgb_model "models/checkpoints/rgb_*_best.pth" \
    --image "$image" \
    --output "results/$(basename $image).png"
done
```

### Custom Training Configuration
```bash
# Advanced training with custom parameters
python train.py \
  --train_mode both \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --backbone resnet50 \
  --fusion_type attention \
  --use_acoustic \
  --freeze_rgb_epochs 20
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'torch'
# Solution: Install PyTorch
pip install torch torchvision

# Error: No module named 'models.rgb_branch'
# Solution: Run from project directory
cd project
python train.py
```

#### 2. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False but you have GPU:
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Memory Issues
```bash
# Reduce batch size if out of memory
python train.py --batch_size 8

# Use CPU if GPU memory insufficient
python train.py --device cpu
```

#### 4. Dataset Not Found
```bash
# Check data directory structure
ls -la data/fruit/SenMangoFruitDDS_bgremoved/
ls -la data/leaf/

# Re-run preprocessing if needed
python scripts/preprocess.py
```

#### 5. Model Loading Errors
```bash
# Check if model files exist
ls models/checkpoints/

# Verify model paths in commands
python demo_inference.py --rgb_model "$(ls models/checkpoints/rgb_*_best.pth | head -1)"
```

### Performance Optimization

#### For Training Speed
```bash
# Use multiple workers (if sufficient RAM)
python train.py --num_workers 8

# Enable mixed precision (if GPU supports)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### For Memory Usage
```bash
# Reduce image size
python train.py --image_size 192

# Use gradient accumulation
python train.py --batch_size 8 --accumulate_grad_batches 4
```

### Debugging Mode
```bash
# Run with detailed output
python -u train.py --train_mode both 2>&1 | tee training.log

# Test with minimal data
python train.py --epochs 1 --batch_size 2
```

---

## ⚙️ Advanced Configuration

### Environment Variables
```bash
# Set PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

# Set logging level
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Custom Model Architectures
```python
# Modify train.py for different backbones
python train.py --backbone efficientnet_b0  # Smaller model
python train.py --backbone resnet50         # Larger model
```

### Hyperparameter Tuning
```bash
# Grid search example
for lr in 0.001 0.0005 0.0001; do
  for bs in 16 32 64; do
    python train.py --learning_rate $lr --batch_size $bs --epochs 10
  done
done
```

### Docker Deployment (Optional)
```dockerfile
# Create Dockerfile
FROM pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "demo_inference.py", "--port", "8000"]
```

```bash
# Build and run
docker build -t mango-classifier .
docker run -p 8000:8000 mango-classifier
```

---

## 📊 Expected Results

### After Successful Setup
- **Preprocessing**: ~5-10 minutes for full datasets
- **Training (5 epochs)**: ~30-60 minutes on CPU, ~10-15 minutes on GPU
- **Training (50 epochs)**: ~5-10 hours on CPU, ~2-3 hours on GPU
- **Evaluation**: ~5-10 minutes
- **Inference**: ~2-3 seconds per image

### Performance Targets
- **RGB Model**: ~82-85% accuracy
- **Fusion Model**: ~88-91% accuracy
- **Improvement**: ~6-8% gain with multi-modal fusion

---

## 🆘 Getting Help

### Check Logs
```bash
# Training logs
ls logs/
cat logs/rgb_baseline_*.log

# System information
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"
```

### Common Commands
```bash
# Verify setup
python test_pipeline.py

# Quick test run
python train.py --epochs 1 --batch_size 4

# Check GPU usage
nvidia-smi  # If available
```

### Support Resources
- **Documentation**: Check `README.md` and `explanation.md`
- **Issues**: Review error messages carefully
- **Community**: Search for similar PyTorch/computer vision issues online

---

## ✅ Setup Checklist

Before running the full pipeline, ensure:

- [ ] ✅ Python 3.10+ installed
- [ ] ✅ All dependencies installed (`pip install -r requirements.txt`)
- [ ] ✅ Test pipeline passes (`python test_pipeline.py`)
- [ ] ✅ Datasets in correct directories
- [ ] ✅ Data preprocessing completed
- [ ] ✅ Sufficient disk space (10GB+)
- [ ] ✅ GPU setup verified (if using)

### Ready to Start!
```bash
# Your first training run
python train.py --train_mode both --epochs 10 --batch_size 16

# Then evaluate
python evaluate.py \
  --rgb_model_path "models/checkpoints/rgb_*_best.pth" \
  --fusion_model_path "models/checkpoints/fusion_*_best.pth"
```

---

*This setup guide will help you get the Multi-Modal Mango Disease Classification system running on your machine. For detailed explanations of the technology, see `explanation.md`.* 