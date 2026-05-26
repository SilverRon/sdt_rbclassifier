"""
Quick test to verify the visualization module imports correctly
and has all required dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        print("✓ sklearn")
    except ImportError as e:
        print(f"✗ sklearn: {e}")
        return False
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        from src.visualize_results import ResultsVisualizer, visualize_training_results
        print("✓ visualize_results module")
    except ImportError as e:
        print(f"✗ visualize_results module: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True

def test_visualizer_init():
    """Test that ResultsVisualizer can be initialized."""
    print("\nTesting ResultsVisualizer initialization...")
    
    try:
        from src.visualize_results import ResultsVisualizer
        
        # Create a temporary visualizer
        output_base = Path(__file__).parent.parent / "output"
        visualizer = ResultsVisualizer("test_v1", output_base)
        
        print(f"✓ Visualizer created")
        print(f"  - Plot dir: {visualizer.plot_dir}")
        print(f"  - Results dir: {visualizer.results_dir}")
        
        # Check directories were created
        if visualizer.plot_dir.exists():
            print(f"✓ Plot directory created")
        else:
            print(f"✗ Plot directory not created")
            return False
        
        if visualizer.results_dir.exists():
            print(f"✓ Results directory created")
        else:
            print(f"✗ Results directory not created")
            return False
        
        print("\n✓ Visualizer initialization successful!")
        return True
        
    except Exception as e:
        print(f"✗ Visualizer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("VISUALIZATION MODULE TEST")
    print("="*60)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test visualizer initialization
    if not test_visualizer_init():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    sys.exit(0 if success else 1)
