#!/usr/bin/env python3
"""
Training Progress Monitor
Monitors the enhanced training pipeline and reports results.
"""

import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import re

def monitor_log_files():
    """Monitor training log files for progress."""
    log_dir = Path("logs")
    
    if not log_dir.exists():
        print("📁 No logs directory found yet...")
        return None
    
    # Find the most recent log files
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        print("📄 No log files found yet...")
        return None
    
    # Get the most recent log file
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    
    return latest_log

def parse_training_progress(log_file):
    """Parse training progress from log file."""
    if not log_file or not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract epoch information
        epoch_pattern = r'Epoch (\d+).*?Val.*?Acc: ([\d.]+)%.*?Val F1: ([\d.]+)'
        matches = re.findall(epoch_pattern, content, re.DOTALL)
        
        if matches:
            epochs = []
            accuracies = []
            f1_scores = []
            
            for match in matches:
                epochs.append(int(match[0]))
                accuracies.append(float(match[1]))
                f1_scores.append(float(match[2]))
            
            return {
                'epochs': epochs,
                'accuracies': accuracies,
                'f1_scores': f1_scores,
                'latest_epoch': epochs[-1] if epochs else 0,
                'latest_accuracy': accuracies[-1] if accuracies else 0,
                'latest_f1': f1_scores[-1] if f1_scores else 0,
                'best_accuracy': max(accuracies) if accuracies else 0
            }
    except Exception as e:
        print(f"Error parsing log: {e}")
    
    return None

def check_model_checkpoints():
    """Check for saved model checkpoints."""
    checkpoint_dir = Path("models/checkpoints")
    
    if not checkpoint_dir.exists():
        return []
    
    # Find all model files
    models = []
    for pattern in ["*rgb*_best.pth", "*fusion*_best.pth"]:
        models.extend(list(checkpoint_dir.glob(pattern)))
    
    return models

def display_progress_summary():
    """Display current training progress summary."""
    print("\n" + "="*70)
    print("🎯 ENHANCED TRAINING PIPELINE - PROGRESS MONITOR")
    print("="*70)
    
    # Check log files
    latest_log = monitor_log_files()
    
    if latest_log:
        print(f"📄 Monitoring log: {latest_log.name}")
        
        # Parse progress
        progress = parse_training_progress(latest_log)
        
        if progress:
            print(f"\n📊 TRAINING PROGRESS:")
            print(f"Current Epoch: {progress['latest_epoch']}")
            print(f"Latest Validation Accuracy: {progress['latest_accuracy']:.2f}%")
            print(f"Latest F1 Score: {progress['latest_f1']:.4f}")
            print(f"Best Accuracy So Far: {progress['best_accuracy']:.2f}%")
            
            # Show improvement
            if progress['latest_accuracy'] > 92.06:  # Previous best
                improvement = progress['latest_accuracy'] - 92.06
                print(f"🚀 IMPROVEMENT: +{improvement:.2f}% over baseline!")
            
            # Progress visualization
            if len(progress['epochs']) > 1:
                print(f"\n📈 ACCURACY TREND:")
                epochs = progress['epochs'][-5:]  # Last 5 epochs
                accs = progress['accuracies'][-5:]
                
                for i, (ep, acc) in enumerate(zip(epochs, accs)):
                    trend = ""
                    if i > 0:
                        if acc > accs[i-1]:
                            trend = "📈"
                        elif acc < accs[i-1]:
                            trend = "📉"
                        else:
                            trend = "➡️"
                    
                    print(f"  Epoch {ep}: {acc:.2f}% {trend}")
        else:
            print("⏳ Training in progress, waiting for results...")
    else:
        print("⏳ Waiting for training to start...")
    
    # Check saved models
    models = check_model_checkpoints()
    
    if models:
        print(f"\n💾 SAVED MODELS ({len(models)} found):")
        for model in models:
            model_type = "RGB" if "rgb" in model.name else "Fusion"
            size_mb = model.stat().st_size / (1024*1024)
            print(f"  ✅ {model_type} Model: {model.name} ({size_mb:.1f} MB)")
    else:
        print("\n💾 No models saved yet...")
    
    # Performance targets
    print(f"\n🎯 PERFORMANCE TARGETS:")
    print(f"RGB Model Target: 95-97% accuracy")
    print(f"Fusion Model Target: 94-96% accuracy")
    print(f"Ensemble Target: 97-98% accuracy")
    
    print("\n⏰ Estimated completion time: 2-4 hours for full pipeline")

def main():
    """Main monitoring loop."""
    print("🔍 Starting Enhanced Training Monitor...")
    
    try:
        while True:
            display_progress_summary()
            
            print(f"\n⏱️  Next update in 5 minutes... (Ctrl+C to stop)")
            print("="*70)
            
            time.sleep(300)  # Update every 5 minutes
            
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    # Run once and exit (for single check)
    display_progress_summary()
    
    # Uncomment below for continuous monitoring
    # main() 