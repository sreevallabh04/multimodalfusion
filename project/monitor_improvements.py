#!/usr/bin/env python3
"""
Monitor training improvements and compare with baseline.
Shows real-time accuracy gains from the enhanced model.
"""

import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def load_latest_logs():
    """Load the most recent training logs."""
    log_dir = Path("logs")
    
    # Find latest log files
    log_files = list(log_dir.glob("*.log"))
    config_files = list(log_dir.glob("*_config.json"))
    
    if not log_files:
        return None, None
    
    # Get the most recent ones
    latest_log = max(log_files, key=os.path.getmtime)
    latest_config = max(config_files, key=os.path.getmtime)
    
    return latest_log, latest_config


def parse_training_progress(log_file):
    """Parse training progress from log file."""
    epochs = []
    train_acc = []
    val_acc = []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "Train - Loss:" in line and "Acc:" in line:
                # Extract epoch and accuracy
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Epoch":
                        epoch = int(parts[i+1])
                    elif part == "Acc:" and "Train" in line:
                        acc = float(parts[i+1].replace('%', ''))
                        if epoch not in [e[0] for e in train_acc]:
                            train_acc.append((epoch, acc))
            
            elif "Val   - Loss:" in line and "Acc:" in line:
                # Extract validation accuracy
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Acc:":
                        acc = float(parts[i+1].replace('%', ''))
                        # Get the epoch from the most recent train entry
                        if train_acc:
                            epoch = train_acc[-1][0]
                            val_acc.append((epoch, acc))
    
    except Exception as e:
        print(f"Error parsing log: {e}")
        return [], []
    
    return train_acc, val_acc


def show_current_status():
    """Show current training status and improvements."""
    print("🔍 TRAINING PROGRESS MONITOR")
    print("=" * 50)
    
    # Load current training info
    latest_log, latest_config = load_latest_logs()
    
    if not latest_log:
        print("❌ No training logs found yet")
        return
    
    # Load configuration
    config = {}
    if latest_config and latest_config.exists():
        try:
            with open(latest_config, 'r') as f:
                config = json.load(f)
        except:
            pass
    
    print(f"📁 Latest log: {latest_log.name}")
    if config:
        args = config.get('args', {})
        print(f"🏗️ Architecture: {args.get('backbone', 'unknown')}")
        print(f"⏰ Epochs: {args.get('epochs', 'unknown')}")
        print(f"📊 Batch size: {args.get('batch_size', 'unknown')}")
        print(f"🔄 Learning rate: {args.get('learning_rate', 'unknown')}")
    
    # Parse progress
    train_progress, val_progress = parse_training_progress(latest_log)
    
    if not train_progress:
        print("⏳ Training just started, no progress yet...")
        return
    
    print("\n📈 CURRENT PROGRESS")
    print("-" * 30)
    
    current_epoch = train_progress[-1][0] if train_progress else 0
    current_train_acc = train_progress[-1][1] if train_progress else 0
    current_val_acc = val_progress[-1][1] if val_progress else 0
    
    print(f"Current Epoch: {current_epoch}")
    print(f"Train Accuracy: {current_train_acc:.2f}%")
    print(f"Val Accuracy: {current_val_acc:.2f}%")
    
    # Show improvement vs baseline
    baseline_fusion = 88.89  # Previous best
    baseline_rgb = 82.54     # RGB only
    
    if current_val_acc > 0:
        improvement_vs_fusion = current_val_acc - baseline_fusion
        improvement_vs_rgb = current_val_acc - baseline_rgb
        
        print(f"\n🎯 IMPROVEMENT ANALYSIS")
        print("-" * 30)
        print(f"vs Previous Fusion: {improvement_vs_fusion:+.2f}%")
        print(f"vs RGB Baseline: {improvement_vs_rgb:+.2f}%")
        
        if improvement_vs_fusion > 0:
            print(f"🎉 Already improved over previous best!")
        elif current_epoch < 10:
            print(f"⏳ Still warming up, expect improvements after epoch 10")
        
        # Predict final accuracy based on current progress
        if current_epoch > 5:
            growth_rate = (current_val_acc - val_progress[0][1]) / current_epoch
            predicted_final = current_val_acc + growth_rate * (20 - current_epoch)
            print(f"📊 Predicted final accuracy: {predicted_final:.1f}%")
    
    # Show recent epoch history
    if len(val_progress) > 1:
        print(f"\n📊 RECENT EPOCHS")
        print("-" * 30)
        for epoch, acc in val_progress[-5:]:  # Last 5 epochs
            status = "📈" if epoch == 1 or acc > val_progress[max(0, epoch-2)][1] else "📉"
            print(f"Epoch {epoch:2d}: {acc:.2f}% {status}")


def show_comparison_table():
    """Show comparison with previous results."""
    print("\n🔄 MODEL COMPARISON")
    print("=" * 60)
    
    # Previous results
    results = [
        ("RGB ResNet18 (3 epochs)", 82.54, "Previous baseline"),
        ("Fusion ResNet18 (3 epochs)", 88.89, "Previous best"),
        ("Current Training", "TBD", "In progress..."),
    ]
    
    print(f"{'Model':<25} {'Accuracy':<12} {'Status'}")
    print("-" * 60)
    for model, acc, status in results:
        if isinstance(acc, float):
            print(f"{model:<25} {acc:<12.2f}% {status}")
        else:
            print(f"{model:<25} {acc:<12} {status}")
    
    print("\n🎯 Expected improvements with ResNet50 + extended training:")
    print("   • Extended training (20 epochs): +3-4%")
    print("   • ResNet50 architecture: +2-3%")
    print("   • Better optimization: +1-2%")
    print("   • Total expected: 94-96% accuracy")


def main():
    """Main monitoring function."""
    
    print("🚀 ACCURACY IMPROVEMENT MONITOR")
    print("=" * 60)
    print("Monitoring training progress for immediate improvements...")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
            
            print("🚀 ACCURACY IMPROVEMENT MONITOR")
            print("=" * 60)
            print(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            print()
            
            show_current_status()
            show_comparison_table()
            
            print("\n" + "=" * 60)
            print("⏳ Refreshing in 30 seconds... (Ctrl+C to exit)")
            
            time.sleep(30)
    
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
        show_current_status()


if __name__ == "__main__":
    main() 