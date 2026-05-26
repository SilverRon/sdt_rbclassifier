import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Starting reproduction script...")
    
    # Path configuration
    PROJECT_ROOT = Path('/lyman/data1/rb_classification_meta')
    OUTPUT_DIR = PROJECT_ROOT / 'output'
    META_DIR = OUTPUT_DIR / 'meta'
    SNAPSHOT_DIR = OUTPUT_DIR / 'snapshot'
    SUMMARY_DIR = OUTPUT_DIR / 'summary'
    FIGURES_DIR = SUMMARY_DIR / 'figures'

    print(f"Project root: {PROJECT_ROOT}")

    def discover_meta_runs(meta_dir: Path) -> List[Dict]:
        runs = []
        if not meta_dir.exists():
            return runs
        for run_dir in sorted(meta_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            args_file = run_dir / 'args.json'
            metrics_file = run_dir / 'metrics.json'
            if not args_file.exists() and not metrics_file.exists():
                continue
            run_info = {
                'run_id': run_dir.name,
                'model_family': 'meta',
                'run_path': run_dir,
                'args_file': args_file if args_file.exists() else None,
                'metrics_file': metrics_file if metrics_file.exists() else None,
                'features_file': run_dir / 'features.json' if (run_dir / 'features.json').exists() else None,
            }
            runs.append(run_info)
        return runs

    def discover_snapshot_runs(snapshot_dir: Path) -> List[Dict]:
        runs = []
        results_dir = snapshot_dir / 'results'
        if not results_dir.exists():
            return runs
        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_file = run_dir / 'metrics.json'
            if not metrics_file.exists():
                continue
            run_info = {
                'run_id': run_dir.name,
                'model_family': 'snapshot',
                'run_path': run_dir,
                'metrics_file': metrics_file,
            }
            runs.append(run_info)
        return runs

    print("Discovering runs...")
    meta_runs = discover_meta_runs(META_DIR)
    snapshot_runs = discover_snapshot_runs(SNAPSHOT_DIR)
    all_runs = meta_runs + snapshot_runs
    print(f"Total runs discovered: {len(all_runs)}")

    def safe_load_json(file_path: Path) -> Optional[Dict]:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load {file_path}: {e}")
            return None

    def parse_meta_artifacts(run_info: Dict) -> Dict:
        result = {'run_id': run_info['run_id'], 'model_family': 'meta', 'auroc': np.nan, 'f1_macro': np.nan}
        if run_info['metrics_file']:
            metrics = safe_load_json(run_info['metrics_file'])
            if metrics:
                result['auroc'] = metrics.get('auroc', np.nan)
                result['f1_macro'] = metrics.get('f1_macro', np.nan)
        return result

    def parse_snapshot_artifacts(run_info: Dict) -> Dict:
        result = {'run_id': run_info['run_id'], 'model_family': 'snapshot', 'auroc': np.nan, 'f1_macro': np.nan}
        if run_info['metrics_file']:
            data = safe_load_json(run_info['metrics_file'])
            if data and 'metrics' in data:
                metrics = data['metrics']
                result['auroc'] = metrics.get('test_auroc', np.nan)
                result['f1_macro'] = metrics.get('test_f1', np.nan)
        return result

    print("Extracting metrics...")
    all_metrics = []
    for run in meta_runs:
        all_metrics.append(parse_meta_artifacts(run))
    for run in snapshot_runs:
        all_metrics.append(parse_snapshot_artifacts(run))
    
    df_summary = pd.DataFrame(all_metrics)
    print(f"Summary table created with {len(df_summary)} rows")

    print("Generating visualizations...")
    df_plot = df_summary[df_summary['auroc'].notna()].copy()
    if len(df_plot) > 0:
        plt.figure(figsize=(14, 6))
        df_plot_sorted = df_plot.sort_values(['model_family', 'auroc'], ascending=[True, False])
        x_pos = np.arange(len(df_plot_sorted))
        colors = ['#3498db' if fam == 'meta' else '#e74c3c' for fam in df_plot_sorted['model_family']]
        plt.bar(x_pos, df_plot_sorted['auroc'], color=colors, alpha=0.7, edgecolor='black')
        plt.savefig(FIGURES_DIR / 'repro_auroc_comparison.png')
        print("Successfully saved auroc comparison plot")
    else:
        print("No runs with valid AUROC found")

    print("Reproduction script finished successfully!")

if __name__ == "__main__":
    main()
