#!/usr/bin/env python
"""
Test script to verify balancing logic works correctly.
This creates a small synthetic dataset to quickly test balancing.
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import RBSnapshotDataset, RBDataModule

def test_balancing():
    print("="*60)
    print("TESTING BALANCING LOGIC")
    print("="*60)
    
    # Create synthetic imbalanced dataset
    n_real = 1000
    n_bogus = 100
    
    # Create fake data
    real_data = torch.randn(n_real, 1, 25, 25)
    bogus_data = torch.randn(n_bogus, 1, 25, 25)
    
    all_data = torch.cat([real_data, bogus_data], dim=0)
    all_targets = torch.cat([
        torch.ones(n_real),
        torch.zeros(n_bogus)
    ])
    
    print(f"\nOriginal synthetic dataset:")
    print(f"  Real: {(all_targets == 1).sum().item()}")
    print(f"  Bogus: {(all_targets == 0).sum().item()}")
    print(f"  Total: {len(all_targets)}")
    
    # Create dataset
    dataset = RBSnapshotDataset(data=all_data, targets=all_targets)
    
    # Create a minimal DataModule to test _balance_dataset
    class TestDataModule(RBDataModule):
        def __init__(self):
            # Skip parent init
            self.balance_data = True
    
    dm = TestDataModule()
    
    print("\n" + "="*60)
    print("APPLYING DATA BALANCING")
    print("="*60)
    
    # Test balancing
    balanced_dataset = dm._balance_dataset(dataset)
    
    print("="*60)
    
    # Verify results
    final_real = (balanced_dataset.targets == 1).sum().item()
    final_bogus = (balanced_dataset.targets == 0).sum().item()
    
    print(f"\nFinal balanced dataset:")
    print(f"  Real: {final_real}")
    print(f"  Bogus: {final_bogus}")
    print(f"  Total: {len(balanced_dataset)}")
    
    if final_real == final_bogus == n_bogus:
        print("\n✓ TEST PASSED: Balancing works correctly!")
        print(f"  Real was downsampled from {n_real} to {n_bogus}")
        print(f"  Bogus remained at {n_bogus}")
        return True
    else:
        print("\n✗ TEST FAILED: Balancing did not work correctly!")
        print(f"  Expected: {n_bogus} Real, {n_bogus} Bogus")
        print(f"  Got: {final_real} Real, {final_bogus} Bogus")
        return False

if __name__ == "__main__":
    success = test_balancing()
    sys.exit(0 if success else 1)
