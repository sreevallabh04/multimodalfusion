#!/usr/bin/env python3
"""
Prepare Multi-Modal Mango Disease Classification project for GitHub upload.
Provides multiple options for handling datasets and large files.
"""

import os
import shutil
import zipfile
import argparse
from pathlib import Path
import subprocess
import json


def get_directory_size(directory):
    """Calculate total size of directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def format_size(size_bytes):
    """Format bytes to human readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"


def analyze_project_size():
    """Analyze the size of different project components."""
    print("📊 PROJECT SIZE ANALYSIS")
    print("=" * 50)
    
    total_project_size = get_directory_size('project')
    print(f"Total Project Size: {format_size(total_project_size)}")
    print()
    
    # Analyze datasets
    if os.path.exists('project/data'):
        data_size = get_directory_size('project/data')
        print(f"📁 Datasets: {format_size(data_size)}")
        
        # Break down by dataset type
        for subdir in ['fruit', 'leaf', 'thermal', 'processed']:
            subdir_path = f'project/data/{subdir}'
            if os.path.exists(subdir_path):
                subdir_size = get_directory_size(subdir_path)
                print(f"   {subdir}: {format_size(subdir_size)}")
    
    print("\n🎯 GitHub Recommendations:")
    if total_project_size > 1024**3:  # > 1GB
        print("   ⚠️  Project is >1GB - consider dataset alternatives")
    elif total_project_size > 100 * 1024**2:  # > 100MB
        print("   ⚠️  Project is >100MB - consider Git LFS for datasets")
    else:
        print("   ✅ Project size is acceptable for GitHub")
    
    return total_project_size


def create_dataset_archives():
    """Create compressed archives of datasets."""
    print("\n📦 CREATING DATASET ARCHIVES")
    print("=" * 40)
    
    archives_dir = Path('dataset_archives')
    archives_dir.mkdir(exist_ok=True)
    
    datasets = {
        'mango_fruit_dataset.zip': 'project/data/fruit',
        'mango_leaf_dataset.zip': 'project/data/leaf',
        'thermal_maps.zip': 'project/data/thermal',
        'processed_data.zip': 'project/data/processed'
    }
    
    created_archives = []
    
    for archive_name, source_dir in datasets.items():
        if os.path.exists(source_dir):
            archive_path = archives_dir / archive_name
            
            print(f"Creating {archive_name}...")
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
            
            # Check archive size
            archive_size = archive_path.stat().st_size
            original_size = get_directory_size(source_dir)
            compression_ratio = (1 - archive_size/original_size) * 100
            
            print(f"   ✅ {archive_name}: {format_size(archive_size)} "
                  f"(compressed {compression_ratio:.1f}%)")
    
    return created_archives


def setup_git_lfs():
    """Set up Git LFS for large files."""
    print("\n🔧 SETTING UP GIT LFS")
    print("=" * 30)
    
    try:
        # Check if git lfs is installed
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
        print("✅ Git LFS is installed")
        
        # Initialize git lfs
        subprocess.run(['git', 'lfs', 'install'], check=True, capture_output=True)
        print("✅ Git LFS initialized")
        
        # Set up tracking for large files
        lfs_patterns = [
            '*.zip',
            '*.pth',
            '*.pkl',
            '*.h5',
            '*.hdf5',
            'project/data/**/*',
            'project/models/checkpoints/**/*'
        ]
        
        for pattern in lfs_patterns:
            subprocess.run(['git', 'lfs', 'track', pattern], check=True, capture_output=True)
            print(f"✅ Tracking {pattern} with Git LFS")
        
        print("\n📝 Git LFS setup complete!")
        print("   Large files will be stored in Git LFS")
        print("   Cloning will be faster and repository will be smaller")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git LFS not installed or not available")
        print("   Install Git LFS: https://git-lfs.github.io/")
        return False


def create_dataset_info():
    """Create dataset information file."""
    dataset_info = {
        "name": "Multi-Modal Mango Disease Classification Dataset",
        "description": "RGB and thermal image datasets for mango fruit disease classification",
        "datasets": {
            "fruit": {
                "name": "MangoFruitDDS",
                "description": "RGB images of mango fruits with disease annotations",
                "classes": ["Healthy", "Anthracnose", "Alternaria", "Black Mould Rot", "Stem and Rot"],
                "size": "~100MB",
                "format": "JPEG images, 224x224 processed"
            },
            "leaf": {
                "name": "MangoLeafBD",
                "description": "RGB images of mango leaves for lesion detection",
                "classes": ["Healthy", "Anthracnose", "Bacterial Canker", "Cutting Weevil", 
                           "Die Back", "Gall Midge", "Powdery Mildew", "Sooty Mould"],
                "size": "~150MB",
                "format": "JPEG images, 224x224 processed"
            },
            "thermal": {
                "name": "Simulated Thermal Maps",
                "description": "Synthetic thermal maps generated using lesion detection",
                "size": "~30MB",
                "format": "Grayscale images, 224x224"
            }
        },
        "download_instructions": {
            "github_lfs": "git lfs pull (if using Git LFS)",
            "archives": "Download from releases or external hosting",
            "setup": "Run python scripts/preprocess.py after downloading"
        },
        "citations": [
            "MangoFruitDDS: [Original paper citation needed]",
            "MangoLeafBD: [Original paper citation needed]"
        ]
    }
    
    with open('project/DATASET_INFO.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("✅ Created DATASET_INFO.json")


def create_github_instructions():
    """Create instructions for GitHub setup."""
    instructions = """# 🚀 GitHub Setup Instructions

## 📋 Repository Setup

### Option 1: Code Only (Recommended for most users)
```bash
# 1. Clone repository
git clone https://github.com/yourusername/multimodal-mango-classification.git
cd multimodal-mango-classification

# 2. Install dependencies
pip install -r project/requirements.txt

# 3. Download datasets (see DATASET_INFO.json for sources)
# Place datasets in project/data/ following the structure in setup.md

# 4. Run preprocessing
python project/scripts/preprocess.py

# 5. Start training
python project/train.py --epochs 50 --backbone resnet50
```

### Option 2: With Git LFS (For contributors)
```bash
# 1. Install Git LFS
# Download from: https://git-lfs.github.io/

# 2. Clone with LFS
git lfs clone https://github.com/yourusername/multimodal-mango-classification.git
cd multimodal-mango-classification

# 3. Pull LFS files
git lfs pull

# 4. Follow setup instructions
# See setup.md for detailed instructions
```

### Option 3: Download Archives
```bash
# 1. Clone repository
git clone https://github.com/yourusername/multimodal-mango-classification.git

# 2. Download dataset archives from releases
# Extract to project/data/

# 3. Follow setup instructions
```

## 📊 Repository Structure
```
multimodal-mango-classification/
├── project/                 # Main project code
│   ├── models/             # Model architectures
│   ├── scripts/            # Data processing scripts
│   ├── data/               # Datasets (via LFS or manual download)
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── requirements.txt    # Dependencies
├── README.md               # Project overview
├── setup.md                # Detailed setup instructions
└── DATASET_INFO.json       # Dataset information
```

## 🔧 Development Setup
```bash
# For contributors
git clone https://github.com/yourusername/multimodal-mango-classification.git
cd multimodal-mango-classification
pip install -r project/requirements.txt
python project/test_pipeline.py  # Verify setup
```

## 📝 Next Steps
1. ⭐ Star the repository
2. 🍴 Fork for contributions
3. 📖 Read setup.md for detailed instructions
4. 🚀 Start with quick_accuracy_boost.py for immediate improvements
"""

    with open('GITHUB_SETUP.md', 'w') as f:
        f.write(instructions)
    
    print("✅ Created GITHUB_SETUP.md")


def main():
    parser = argparse.ArgumentParser(description='Prepare project for GitHub')
    parser.add_argument('--mode', choices=['analyze', 'archive', 'lfs', 'prepare'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--create-archives', action='store_true', 
                       help='Create compressed dataset archives')
    parser.add_argument('--setup-lfs', action='store_true',
                       help='Set up Git LFS for large files')
    
    args = parser.parse_args()
    
    print("🚀 MULTIMODAL MANGO CLASSIFICATION - GITHUB PREPARATION")
    print("=" * 70)
    
    # Analyze project size
    total_size = analyze_project_size()
    
    print(f"\n💡 Recommendations:")
    print(f"1. Code-only repository (recommended)")
    print(f"2. Dataset archives in releases")
    print(f"3. Git LFS for datasets (if needed)")
    
    if args.mode == 'analyze' or args.mode == 'prepare':
        if args.mode == 'prepare':
            print("\n🎯 RECOMMENDED APPROACH:")
            print("=" * 40)
            
            if total_size > 1024**3:  # > 1GB
                print("📦 Large project - use dataset archives + external hosting")
                if input("Create dataset archives? (y/n): ").lower() == 'y':
                    create_dataset_archives()
            elif total_size > 100 * 1024**2:  # > 100MB
                print("🔧 Medium project - use Git LFS for datasets")
                if input("Set up Git LFS? (y/n): ").lower() == 'y':
                    setup_git_lfs()
            else:
                print("✅ Small project - can upload directly to GitHub")
    
    if args.create_archives or args.mode == 'archive':
        create_dataset_archives()
    
    if args.setup_lfs or args.mode == 'lfs':
        setup_git_lfs()
    
    if args.mode == 'prepare':
        create_dataset_info()
        create_github_instructions()
        
        print("\n🎉 PROJECT PREPARED FOR GITHUB!")
        print("=" * 40)
        print("Next steps:")
        print("1. 📝 Create GitHub repository")
        print("2. 🔄 git add . && git commit -m 'Initial commit'")
        print("3. 🚀 git push origin main")
        print("4. 📖 Update README.md with your repository URL")
        print("5. 🏷️ Create releases for dataset archives (if created)")


if __name__ == "__main__":
    main() 