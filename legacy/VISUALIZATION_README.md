# Visualization Module - Implementation Summary

## Overview
Added comprehensive visualization and reporting capabilities to the PyTorch Lightning training pipeline without modifying any existing training, data loading, or model code.

## Changes Made

### 1. New File: `src/visualize_results.py`
A complete visualization module that generates publication-quality plots and reports.

**Key Features:**
- **ResultsVisualizer class**: Handles all visualization and reporting
- **Automatic prediction collection**: Gathers predictions from validation and test sets
- **Publication-quality plots**: All plots saved at 200+ DPI with clear labels
- **Threshold optimization**: Finds optimal classification threshold by maximizing F1 on validation set
- **Multi-format output**: PNG plots, JSON metrics, and Markdown reports

**Generated Visualizations:**

1. **Learning Curves** (`learning_curves.png`)
   - Train/val loss over epochs
   - Train/val accuracy over epochs
   - Side-by-side comparison

2. **ROC Curves** (`roc_curve_test.png`, `roc_curve_val.png`)
   - ROC curve with AUROC annotation
   - Random classifier baseline
   - For both test and validation sets

3. **Precision-Recall Curves** (`pr_curve_test.png`, `pr_curve_val.png`)
   - PR curve with Average Precision annotation
   - For both test and validation sets

4. **Confusion Matrices** (`confusion_matrices_test.png`, `confusion_matrices_val.png`)
   - Two matrices side-by-side:
     - Default threshold (0.5)
     - Optimal threshold (max F1 on validation)
   - Color-coded with value annotations

5. **Probability Histograms** (`probability_histogram_test.png`, `probability_histogram_val.png`)
   - Distribution of predicted probabilities
   - Separated by true class (Bogus vs Real)
   - Shows class separation and calibration
   - Threshold line overlay

### 2. Modified File: `script/04_Train_Test_Validate_Snapshots.py`
Minimal changes to integrate visualization:

**Line 18**: Added import
```python
from src.visualize_results import visualize_training_results
```

**Lines 121-128**: Added visualization call after testing
```python
# 8. Visualize Results (skip if fast_dev_run)
if not args.fast_dev_run:
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print("="*60)
    visualize_training_results(trainer, model, dm, args, output_base)
else:
    print("\nSkipping visualization (fast_dev_run mode)")
```

**Lines 151-152**: Added directory creation
```python
os.makedirs(output_dir / "plots" / args.version, exist_ok=True)
os.makedirs(output_dir / "results" / args.version, exist_ok=True)
```

### 3. Test File: `script/test_visualization.py`
Verification script to ensure all dependencies are available.

## Output Structure

After training completes, the following structure is created:

```
output/
├── checkpoints/{version}/
│   └── rb-classifier-{epoch}-{val_loss}.ckpt
├── logs/{version}/
│   └── [TensorBoard logs]
├── plots/{version}/
│   ├── learning_curves.png
│   ├── roc_curve_test.png
│   ├── roc_curve_val.png
│   ├── pr_curve_test.png
│   ├── pr_curve_val.png
│   ├── confusion_matrices_test.png
│   ├── confusion_matrices_val.png
│   ├── probability_histogram_test.png
│   └── probability_histogram_val.png
└── results/{version}/
    ├── metrics.json
    └── report.md
```

## Generated Reports

### 1. `metrics.json`
Comprehensive JSON summary including:
- Experiment version
- All hyperparameters (batch_size, lr, max_epochs, num_workers, balance_data)
- Best checkpoint path
- All validation metrics (loss, acc, f1, auroc)
- All test metrics (loss, acc, f1, prec, rec, auroc, avg_precision)
- Optimal threshold value
- Output paths

### 2. `report.md`
Human-readable Markdown report with:
- Configuration summary
- Dataset settings
- Training hyperparameters
- Best checkpoint path
- Validation metrics table
- Test metrics table
- List of all generated plots with descriptions

## Usage

### Standard Training Run
```bash
python script/04_Train_Test_Validate_Snapshots.py \
    --version v1 \
    --batch_size 128 \
    --lr 1e-3 \
    --max_epochs 50 \
    --num_workers 8 \
    --balance_data
```

After training completes:
- All plots will be in `output/plots/v1/`
- Metrics summary in `output/results/v1/metrics.json`
- Report in `output/results/v1/report.md`

### Fast Development Run (Skip Visualization)
```bash
python script/04_Train_Test_Validate_Snapshots.py \
    --version test \
    --fast_dev_run
```

Visualization is automatically skipped in fast_dev_run mode.

## Dependencies

The visualization module requires:
- **matplotlib**: Plotting (already available)
- **numpy**: Array operations (already available)
- **scikit-learn**: Metrics computation (installed)
- **torch**: PyTorch (already available)

All dependencies are now installed and verified.

## Key Design Decisions

1. **Non-invasive Integration**: Zero changes to training loop, data loading, or model code
2. **Automatic Execution**: Runs automatically after testing completes
3. **Multi-GPU Compatible**: Uses model.eval() and proper device handling
4. **Threshold Optimization**: Finds best threshold on validation set, applies to test set
5. **Publication Quality**: High DPI (200+), clear labels, professional styling
6. **Comprehensive Coverage**: Both validation and test sets analyzed
7. **Fast Dev Mode Aware**: Skips visualization in debugging mode
8. **Self-Contained**: All logic in separate module for maintainability

## Limitations and Notes

1. **Learning Curves**: Attempts to read from TensorBoard logs. If unavailable, learning curve plots may be empty (but won't crash).
2. **Threshold Optimization**: Computed on validation set. If validation set is very small, threshold may not generalize well.
3. **Memory**: Collects all predictions in memory. For very large test sets (>1M samples), this could use significant RAM.
4. **TensorBoard Dependency**: Optional - if TensorBoard event files can't be read, only final metrics are reported (not epoch-by-epoch history).

## Verification

Run the test script to verify everything is working:
```bash
python script/test_visualization.py
```

Expected output:
```
============================================================
VISUALIZATION MODULE TEST
============================================================
Testing imports...
✓ numpy
✓ matplotlib
✓ sklearn
✓ torch
✓ visualize_results module

✓ All imports successful!

Testing ResultsVisualizer initialization...
✓ Visualizer created
✓ Plot directory created
✓ Results directory created

✓ Visualizer initialization successful!

============================================================
✓ ALL TESTS PASSED
============================================================
```

## Next Steps

1. **Run a full training experiment** to generate real visualizations
2. **Review the generated plots** in `output/plots/{version}/`
3. **Check the metrics.json** for programmatic access to results
4. **Read the report.md** for a human-friendly summary

## Example Workflow

```bash
# 1. Run training with visualization
python script/04_Train_Test_Validate_Snapshots.py \
    --version experiment_001 \
    --batch_size 128 \
    --lr 1e-3 \
    --max_epochs 50 \
    --balance_data

# 2. View results
cat output/results/experiment_001/report.md

# 3. Open plots
# (Use your preferred image viewer)
ls output/plots/experiment_001/

# 4. Access metrics programmatically
python -c "import json; print(json.dumps(json.load(open('output/results/experiment_001/metrics.json')), indent=2))"
```

## Troubleshooting

**Issue**: "No module named 'sklearn'"
**Solution**: Run `pip install scikit-learn`

**Issue**: Learning curves are empty
**Solution**: This is expected if TensorBoard logs can't be read. The other plots will still be generated.

**Issue**: "CUDA out of memory" during prediction collection
**Solution**: Reduce batch size or use CPU for inference by modifying the visualizer to explicitly use CPU.

**Issue**: Plots look different than expected
**Solution**: Check that matplotlib backend is correctly configured. The module uses the default backend.
