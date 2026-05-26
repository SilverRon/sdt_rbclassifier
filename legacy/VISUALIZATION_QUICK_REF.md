# Quick Reference: Visualization Module

## What Was Added

### New Files
1. **`src/visualize_results.py`** - Complete visualization module (600+ lines)
2. **`script/test_visualization.py`** - Test/verification script
3. **`VISUALIZATION_README.md`** - Full documentation (this file's companion)

### Modified Files
1. **`script/04_Train_Test_Validate_Snapshots.py`** - Only 3 minimal changes:
   - Added import (line 18)
   - Added visualization call after testing (lines 121-128)
   - Added directory creation (lines 151-152)

## What It Does

After training and testing complete, automatically generates:

### 9 Publication-Quality Plots
1. Learning curves (loss + accuracy)
2. ROC curve (test set)
3. ROC curve (validation set)
4. Precision-Recall curve (test set)
5. Precision-Recall curve (validation set)
6. Confusion matrices at 2 thresholds (test set)
7. Confusion matrices at 2 thresholds (validation set)
8. Probability histogram (test set)
9. Probability histogram (validation set)

### 2 Summary Files
1. **`metrics.json`** - All metrics in JSON format
2. **`report.md`** - Human-readable summary report

## Quick Start

```bash
# Install dependency (if needed)
pip install scikit-learn

# Test the module
python script/test_visualization.py

# Run training (visualization happens automatically)
python script/04_Train_Test_Validate_Snapshots.py --version my_experiment

# View results
ls output/plots/my_experiment/
cat output/results/my_experiment/report.md
```

## Output Locations

```
output/
├── plots/{version}/          # All PNG plots (200 DPI)
└── results/{version}/        # JSON + Markdown reports
```

## Key Features

✅ **Zero impact on training** - No changes to model, data, or training loop  
✅ **Automatic execution** - Runs after testing, no manual steps  
✅ **Publication quality** - High DPI, clear labels, professional styling  
✅ **Threshold optimization** - Finds best F1 threshold on validation set  
✅ **Comprehensive metrics** - AUROC, AP, F1, Precision, Recall, Confusion Matrix  
✅ **Multi-format output** - Plots (PNG), metrics (JSON), report (Markdown)  
✅ **Fast dev mode aware** - Skips visualization when `--fast_dev_run` is used  

## What Wasn't Changed

❌ Training loop - Unchanged  
❌ Data loading - Unchanged  
❌ Model architecture - Unchanged  
❌ Balancing logic - Unchanged  
❌ Checkpointing - Unchanged  
❌ CLI arguments - Unchanged (except visualization happens automatically)  
❌ Default behavior - Unchanged (visualization is additive only)  

## Example Output

### metrics.json
```json
{
  "version": "v1",
  "hyperparameters": {
    "batch_size": 128,
    "learning_rate": 0.001,
    "max_epochs": 50,
    "num_workers": 8,
    "balance_data": true
  },
  "metrics": {
    "test_auroc": 0.9876,
    "test_f1": 0.9234,
    "optimal_threshold": 0.523
  }
}
```

### report.md
```markdown
# Experiment Report: v1

## Configuration
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Balance Data**: True

## Test Metrics
- **Test AUROC**: 0.9876
- **Test F1**: 0.9234
- **Optimal Threshold**: 0.523

## Generated Plots
All plots saved in: `output/plots/v1/`
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'sklearn'` | Run `pip install scikit-learn` |
| Empty learning curves | Expected if TensorBoard logs unavailable (other plots still work) |
| Visualization skipped | Check you're not using `--fast_dev_run` flag |
| Plots not found | Check `output/plots/{version}/` directory exists |

## Dependencies

- ✅ matplotlib (already installed)
- ✅ numpy (already installed)
- ✅ torch (already installed)
- ✅ scikit-learn (now installed)

## Verification

```bash
python script/test_visualization.py
# Should output: ✓ ALL TESTS PASSED
```

## Code Changes Summary

**Total lines added**: ~650  
**Total lines modified in existing files**: 6  
**Files created**: 3  
**Files modified**: 1  
**Breaking changes**: 0  
**New dependencies**: 1 (scikit-learn)  

## Next Steps

1. ✅ Dependencies installed
2. ✅ Module tested
3. ⏭️ Run a training experiment
4. ⏭️ Review generated visualizations
5. ⏭️ Iterate on model/hyperparameters using insights from plots
