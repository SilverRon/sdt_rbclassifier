#!/usr/bin/env python3
"""
Example script showing how to programmatically access visualization results.
"""

import json
from pathlib import Path
import sys

def load_metrics(version: str, output_base: str = "output") -> dict:
    """
    Load metrics JSON for a given experiment version.
    
    Args:
        version: Experiment version (e.g., 'v1', 'experiment_001')
        output_base: Base output directory (default: 'output')
    
    Returns:
        Dictionary containing all metrics and metadata
    """
    metrics_path = Path(output_base) / "results" / version / "metrics.json"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def print_summary(metrics: dict):
    """
    Print a formatted summary of the metrics.
    
    Args:
        metrics: Metrics dictionary from load_metrics()
    """
    print("=" * 70)
    print(f"EXPERIMENT SUMMARY: {metrics['version']}")
    print("=" * 70)
    
    print("\n📊 HYPERPARAMETERS")
    print("-" * 70)
    hparams = metrics['hyperparameters']
    print(f"  Batch Size:      {hparams['batch_size']}")
    print(f"  Learning Rate:   {hparams['learning_rate']}")
    print(f"  Max Epochs:      {hparams['max_epochs']}")
    print(f"  Num Workers:     {hparams['num_workers']}")
    print(f"  Balance Data:    {hparams['balance_data']}")
    
    print("\n🎯 TEST METRICS")
    print("-" * 70)
    m = metrics['metrics']
    
    # Core metrics
    if 'test_loss' in m:
        print(f"  Loss:            {m['test_loss']:.4f}")
    if 'test_acc' in m:
        print(f"  Accuracy:        {m['test_acc']:.4f}")
    if 'test_f1' in m:
        print(f"  F1 Score:        {m['test_f1']:.4f}")
    if 'test_prec' in m:
        print(f"  Precision:       {m['test_prec']:.4f}")
    if 'test_rec' in m:
        print(f"  Recall:          {m['test_rec']:.4f}")
    if 'test_auroc' in m:
        print(f"  AUROC:           {m['test_auroc']:.4f}")
    if 'test_avg_precision' in m:
        print(f"  Avg Precision:   {m['test_avg_precision']:.4f}")
    
    # Threshold
    if 'optimal_threshold' in m:
        print(f"\n  Optimal Threshold (max F1 on val): {m['optimal_threshold']:.3f}")
    
    print("\n📁 OUTPUT PATHS")
    print("-" * 70)
    paths = metrics['output_paths']
    print(f"  Plots:           {paths['plots']}")
    print(f"  Results:         {paths['results']}")
    
    print("\n💾 CHECKPOINT")
    print("-" * 70)
    print(f"  Best Model:      {metrics['best_checkpoint']}")
    
    print("\n" + "=" * 70)


def compare_experiments(versions: list, output_base: str = "output"):
    """
    Compare metrics across multiple experiments.
    
    Args:
        versions: List of experiment versions to compare
        output_base: Base output directory
    """
    print("=" * 90)
    print("EXPERIMENT COMPARISON")
    print("=" * 90)
    
    # Load all metrics
    all_metrics = {}
    for version in versions:
        try:
            all_metrics[version] = load_metrics(version, output_base)
        except FileNotFoundError:
            print(f"⚠️  Warning: Metrics not found for version '{version}', skipping...")
    
    if not all_metrics:
        print("No valid experiments found.")
        return
    
    # Print comparison table
    print(f"\n{'Version':<20} {'Test Acc':<12} {'Test F1':<12} {'Test AUROC':<12} {'Balanced':<10}")
    print("-" * 90)
    
    for version, metrics in all_metrics.items():
        m = metrics['metrics']
        hparams = metrics['hyperparameters']
        
        acc = f"{m.get('test_acc', 0):.4f}" if 'test_acc' in m else "N/A"
        f1 = f"{m.get('test_f1', 0):.4f}" if 'test_f1' in m else "N/A"
        auroc = f"{m.get('test_auroc', 0):.4f}" if 'test_auroc' in m else "N/A"
        balanced = "Yes" if hparams.get('balance_data', False) else "No"
        
        print(f"{version:<20} {acc:<12} {f1:<12} {auroc:<12} {balanced:<10}")
    
    print("=" * 90)


def main():
    """Main function with example usage."""
    
    # Example 1: Load and print metrics for a single experiment
    print("\n🔍 EXAMPLE 1: Load Single Experiment")
    print("=" * 70)
    
    # Change this to your experiment version
    version = "v1"
    
    try:
        metrics = load_metrics(version)
        print_summary(metrics)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Tip: Run training first:")
        print(f"   python script/04_Train_Test_Validate_Snapshots.py --version {version}")
        return
    
    # Example 2: Access specific metrics programmatically
    print("\n\n🔍 EXAMPLE 2: Access Specific Metrics")
    print("=" * 70)
    
    test_auroc = metrics['metrics'].get('test_auroc', None)
    if test_auroc is not None:
        print(f"Test AUROC: {test_auroc:.4f}")
        
        if test_auroc > 0.95:
            print("✅ Excellent performance!")
        elif test_auroc > 0.90:
            print("✅ Good performance!")
        elif test_auroc > 0.85:
            print("⚠️  Acceptable performance, but could be improved.")
        else:
            print("❌ Performance needs improvement.")
    
    # Example 3: Compare multiple experiments (if available)
    print("\n\n🔍 EXAMPLE 3: Compare Multiple Experiments")
    print("=" * 70)
    
    # List all available experiment versions
    output_base = Path("output")
    results_dir = output_base / "results"
    
    if results_dir.exists():
        available_versions = [d.name for d in results_dir.iterdir() if d.is_dir()]
        
        if len(available_versions) > 1:
            print(f"Found {len(available_versions)} experiments: {', '.join(available_versions)}")
            compare_experiments(available_versions)
        elif len(available_versions) == 1:
            print(f"Only one experiment found: {available_versions[0]}")
            print("Run more experiments to enable comparison.")
        else:
            print("No experiments found.")
    else:
        print("Results directory not found. Run training first.")


if __name__ == "__main__":
    # Check if version is provided as command-line argument
    if len(sys.argv) > 1:
        version = sys.argv[1]
        try:
            metrics = load_metrics(version)
            print_summary(metrics)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Run examples
        main()
