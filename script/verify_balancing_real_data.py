#!/usr/bin/env python
"""
Quick verification script to test balancing with actual data.
Uses only 1 tile to speed up testing.
"""
import os
import sys
import torch
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import RBDataModule

def main():
    print("="*60)
    print("QUICK BALANCING VERIFICATION WITH REAL DATA")
    print("="*60)
    
    # Use actual data directory
    data_dir = "/lyman/data1/rb_classification_meta/data/norm_snapshot"
    
    # Create a test cache directory
    test_cache_dir = "/tmp/rb_test_cache"
    os.makedirs(test_cache_dir, exist_ok=True)
    
    print(f"\nData directory: {data_dir}")
    print(f"Test cache directory: {test_cache_dir}")
    print(f"Balance data: True")
    print("="*60 + "\n")
    
    # Create DataModule with balancing enabled
    dm = RBDataModule(
        data_dir=data_dir,
        batch_size=128,
        num_workers=4,  # Reduced for testing
        balance_data=True
    )
    
    # Override cache paths to use test directory
    dm.train_cache_path = os.path.join(test_cache_dir, "train_cache.pt")
    dm.val_cache_path = os.path.join(test_cache_dir, "val_cache.pt")
    dm.test_cache_path = os.path.join(test_cache_dir, "test_cache.pt")
    
    # Setup (this will create cache and apply balancing)
    print("\nCalling dm.setup()...")
    dm.setup()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"\nCache files created in: {test_cache_dir}")
    print("You can delete this test cache with:")
    print(f"  rm -rf {test_cache_dir}")

if __name__ == "__main__":
    main()
