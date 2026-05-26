# Scientific Sanity Check - Leakage Visualization

This documentation explains the leakage-check visualization feature, designed for scientific verification of the real/bogus classifier.

## Overview

We are concerned about potential data leakage because the model achieves high performance. This visualization allows for visual inspection of individual snapshot predictions on the **TEST** set, which was not used during training or model selection.

## Visualized Split: TEST

The visualization is performed exclusively on the **TEST** set. While validation data can also be visualized for comparison, scientific sanity checking must focus on the test data to ensure findings are unbiased by the model selection process.

## Methodology

### Checkpoint Selection
The visualization uses the **best model checkpoint** identified during training (based on the lowest validation loss). This ensures we are inspecting the most scientific version of the model.

### Score Computation
Predicted scores are computed as:
$$P(Real) = \sigma(\text{logits})$$
where $\sigma$ is the sigmoid function.
- A score of **1.0** indicates absolute confidence in "Real".
- A score of **0.0** indicates absolute confidence in "Bogus".
- Predictions are annotated with $\checkmark$ (Correct) or $\times$ (Incorrect) based on a default 0.5 threshold.

## Generated Figures

The system generates four distinct types of snapshot grids:

1. **`test_snapshots_random.png`**
   - 64 randomly selected samples from the test set. 
   - Provides an unbiased overview of model behavior across the entire data distribution.

2. **`test_snapshots_high_conf_real.png`**
   - 64 samples with the highest predicted probabilities for the "Real" class.
   - Useful for verifying if the model has learned the correct features of real stars/sources.

3. **`test_snapshots_high_conf_bogus.png`**
   - 64 samples with the lowest predicted probabilities (highest confidence for "Bogus").
   - Helps identify common types of artifacts the model correctly rejects.

4. **`test_snapshots_ambiguous.png`**
   - 64 samples with predicted scores closest to 0.5.
   - These are the "hardest" cases for the model. Inspecting these often reveals edge cases where the classifier is uncertain.

## Figure Details

- **Title**: Includes the experiment version and "TEST SET (Leakage Check)".
- **Color Coding**: 
  - **Green**: Correct prediction.
  - **Red**: Incorrect prediction.
- **Annotations**:
  - `T`: True Class (Real/Bogus).
  - `P`: Predicted Probability.

## Location
All figures are saved in:
`output/plots/{version}/`
