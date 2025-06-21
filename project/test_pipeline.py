#!/usr/bin/env python3
"""
Pipeline test script to verify all components work correctly.
This script tests the complete multi-modal fusion pipeline without full training.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project modules to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Core dependencies
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import cv2
        import PIL
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        import albumentations as A
        import timm
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print("✅ All core dependencies imported successfully")
        
        # Project modules
        from models.lesion_detector import LesionDetector, create_lesion_detector
        from models.rgb_branch import RGBBranch, create_rgb_branch
        from models.fusion_model import MultiModalFusionModel, create_fusion_model
        from scripts.dataloader import MultiModalMangoDataset, create_dataloaders
        
        print("✅ All project modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


# Import the functions at module level so they're available to all test functions
try:
    from models.lesion_detector import create_lesion_detector
    from models.rgb_branch import create_rgb_branch  
    from models.fusion_model import create_fusion_model
except ImportError:
    # Will be handled in test_imports
    pass


def test_model_creation():
    """Test that all models can be created and run forward passes."""
    print("\n🧪 Testing model creation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Test lesion detector
        print("  Testing lesion detector...")
        lesion_model = create_lesion_detector(num_classes=8)
        lesion_model.to(device)
        
        dummy_leaf = torch.randn(2, 3, 224, 224).to(device)
        lesion_output = lesion_model(dummy_leaf)
        print(f"    ✅ Lesion detector output shape: {lesion_output.shape}")
        
        # Test lesion probability generation
        lesion_prob = lesion_model.get_lesion_probability(dummy_leaf)
        print(f"    ✅ Lesion probability shape: {lesion_prob.shape}")
        
        # Test RGB branch
        print("  Testing RGB branch...")
        rgb_model = create_rgb_branch(num_classes=5)
        rgb_model.to(device)
        
        dummy_rgb = torch.randn(2, 3, 224, 224).to(device)
        rgb_output = rgb_model(dummy_rgb)
        print(f"    ✅ RGB branch output shape: {rgb_output.shape}")
        
        # Test with features
        rgb_output, rgb_features = rgb_model(dummy_rgb, return_features=True)
        print(f"    ✅ RGB features shape: {rgb_features.shape}")
        
        # Test fusion model (without acoustic)
        print("  Testing fusion model (RGB + Thermal)...")
        fusion_model = create_fusion_model(num_classes=5, use_acoustic=False)
        fusion_model.to(device)
        
        dummy_thermal = torch.randn(2, 1, 224, 224).to(device)
        fusion_output = fusion_model(dummy_rgb, dummy_thermal)
        print(f"    ✅ Fusion output shape: {fusion_output.shape}")
        
        # Test fusion model (with acoustic)
        print("  Testing fusion model (RGB + Thermal + Acoustic)...")
        fusion_acoustic_model = create_fusion_model(num_classes=5, use_acoustic=True)
        fusion_acoustic_model.to(device)
        
        dummy_acoustic = torch.randn(2, 1, 224, 224).to(device)
        fusion_acoustic_output = fusion_acoustic_model(dummy_rgb, dummy_thermal, dummy_acoustic)
        print(f"    ✅ Fusion with acoustic output shape: {fusion_acoustic_output.shape}")
        
        print("✅ All models created and tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_thermal_simulation():
    """Test thermal map generation."""
    print("\n🧪 Testing thermal simulation...")
    
    try:
        from scripts.simulate_thermal import ThermalSimulator
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a dummy lesion model path (won't exist, but should handle gracefully)
        thermal_sim = ThermalSimulator(
            lesion_model_path='dummy_path.pth',
            device=device
        )
        
        # Test thermal map generation with dummy data
        dummy_rgb = torch.randn(1, 3, 224, 224)
        thermal_map = thermal_sim.generate_thermal_map(dummy_rgb)
        
        print(f"    ✅ Generated thermal map shape: {thermal_map.shape}")
        print(f"    ✅ Thermal map value range: [{thermal_map.min():.3f}, {thermal_map.max():.3f}]")
        
        print("✅ Thermal simulation tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Thermal simulation error: {e}")
        return False


def test_dataloader():
    """Test dataloader with dummy data."""
    print("\n🧪 Testing dataloader...")
    
    try:
        from scripts.dataloader import MultiModalMangoDataset
        
        # Create dummy directory structure for testing
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # Create minimal test directories
        for split in ['train', 'val', 'test']:
            for class_name in ['Healthy', 'Anthracnose']:
                rgb_dir = test_data_dir / 'rgb' / split / class_name
                rgb_dir.mkdir(parents=True, exist_ok=True)
                
                thermal_dir = test_data_dir / 'thermal' / 'thermal' / split / class_name
                thermal_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                dummy_thermal = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
                
                for i in range(2):  # Create 2 dummy images per class
                    rgb_path = rgb_dir / f"dummy_{i}.jpg"
                    thermal_path = thermal_dir / f"dummy_{i}.jpg"
                    
                    import cv2
                    cv2.imwrite(str(rgb_path), dummy_rgb)
                    cv2.imwrite(str(thermal_path), dummy_thermal)
        
        # Test dataset creation
        dataset = MultiModalMangoDataset(
            rgb_data_path=str(test_data_dir / 'rgb'),
            thermal_data_path=str(test_data_dir / 'thermal'),
            split='train',
            use_acoustic=False,
            transform_type='val'  # Use simpler transforms for testing
        )
        
        print(f"    ✅ Dataset created with {len(dataset)} samples")
        
        # Test data loading
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"    ✅ Sample loaded: RGB shape {sample[0].shape}, Thermal shape {sample[1].shape}")
        
        # Clean up test data
        import shutil
        shutil.rmtree(test_data_dir)
        
        print("✅ Dataloader tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Dataloader error: {e}")
        # Clean up on error
        try:
            import shutil
            shutil.rmtree("test_data")
        except:
            pass
        return False


def test_training_components():
    """Test training-related components."""
    print("\n🧪 Testing training components...")
    
    try:
        from models.rgb_branch import RGBTrainer
        from models.fusion_model import FusionTrainer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
        
        # Test RGB trainer
        rgb_model = create_rgb_branch(num_classes=5)
        rgb_trainer = RGBTrainer(rgb_model, device, class_names)
        print("    ✅ RGB trainer created successfully")
        
        # Test Fusion trainer
        fusion_model = create_fusion_model(num_classes=5, use_acoustic=False)
        fusion_trainer = FusionTrainer(fusion_model, device, class_names)
        print("    ✅ Fusion trainer created successfully")
        
        print("✅ Training components tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Training components error: {e}")
        return False


def test_evaluation_components():
    """Test evaluation-related components."""
    print("\n🧪 Testing evaluation components...")
    
    try:
        from evaluate import ModelEvaluator
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_names = ['Healthy', 'Anthracnose', 'Alternaria', 'Black Mould Rot', 'Stem and Rot']
        
        # Test with RGB model
        rgb_model = create_rgb_branch(num_classes=5)
        evaluator = ModelEvaluator(rgb_model, device, class_names, 'rgb')
        print("    ✅ Model evaluator created successfully")
        
        # Test metrics calculation
        dummy_labels = [0, 1, 2, 3, 4, 0, 1, 2]
        dummy_predictions = [0, 1, 2, 3, 3, 0, 1, 1]  # Some errors for testing
        dummy_probabilities = np.random.rand(8, 5)
        
        metrics = evaluator._calculate_metrics(dummy_labels, dummy_predictions, dummy_probabilities)
        print(f"    ✅ Metrics calculated: accuracy = {metrics['accuracy']:.2f}%")
        
        print("✅ Evaluation components tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation components error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Multi-Modal Mango Disease Classification Pipeline Test")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Thermal Simulation", test_thermal_simulation),
        ("Dataloader", test_dataloader),
        ("Training Components", test_training_components),
        ("Evaluation Components", test_evaluation_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("\n📋 Next steps:")
        print("1. Run: python scripts/preprocess.py")
        print("2. Run: python train.py --train_mode both")
        print("3. Run: python evaluate.py --rgb_model_path <path> --fusion_model_path <path>")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        print("Make sure all dependencies are installed correctly.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 