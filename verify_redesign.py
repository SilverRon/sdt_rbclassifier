import os
import subprocess
import json
from pathlib import Path

def run_command(cmd):
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error: {stderr}")
    return stdout, stderr, process.returncode

def verify():
    project_root = Path('/lyman/data1/rb_classification_meta')
    output_root = project_root / 'output'
    
    # 1. Meta Smoke Test
    meta_run = "smoke_test_meta_redesign"
    run_command(f"conda run -n rbc python script/04_Train_Test_Validate_Metas.py --model rf --version {meta_run} --smoke_test")
    
    meta_dir = output_root / "meta" / meta_run
    print(f"\nChecking Meta Output: {meta_dir}")
    expected_meta = ["args.json", "features.json", "metrics.json", "model.pkl", "plots/confusion_matrix.png"]
    for f in expected_meta:
        path = meta_dir / f
        if path.exists():
            print(f"  ✓ Found {f}")
        else:
            print(f"  ✗ Missing {f}")
            
    # 2. Snapshot Smoke Test
    snapshot_run = "smoke_test_snap_redesign"
    run_command(f"conda run -n rbc python script/04_Train_Test_Validate_Snapshots.py --model simplecnn --version {snapshot_run} --fast_dev_run --max_epochs 1")
    
    snap_dir = output_root / "snapshot" / snapshot_run
    print(f"\nChecking Snapshot Output: {snap_dir}")
    # Note: fast_dev_run skips visualizations, so we check the basics
    expected_snap = ["args.json", "features.json", "checkpoints"]
    for f in expected_snap:
        path = snap_dir / f
        if path.exists():
            print(f"  ✓ Found {f}")
        else:
            print(f"  ✗ Missing {f}")

    print("\nVerification process finished.")

if __name__ == "__main__":
    verify()
