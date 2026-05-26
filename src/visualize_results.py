"""
Visualization and Reporting Module for RB Classification
Generates publication-quality plots and summary reports after training.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)


from src.utils.output_paths import OutputManager

class ResultsVisualizer:
    """
    Handles all visualization and reporting for the RB classification model.
    """
    
    def __init__(self, output_manager: OutputManager):
        """
        Initialize the visualizer.
        
        Args:
            output_manager: Unified OutputManager instance
        """
        self.output_manager = output_manager
        self.version = output_manager.run_name
        
        # Paths from OutputManager
        self.plot_dir = output_manager.get_path('plots')
        self.data_dir = output_manager.get_path('data')
        
        # Storage for metrics history
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auroc': []
        }
        
        # Storage for predictions
        self.val_predictions = {'y_true': [], 'y_score': []}
        self.test_predictions = {'y_true': [], 'y_score': []}
        
    def extract_metrics_from_logger(self, logger):
        """
        Extract training history from TensorBoard logger.
        
        Args:
            logger: PyTorch Lightning logger instance
        """
        try:
            # Try to read from the logger's metrics
            metrics = logger.experiment.get_logdir_metrics()
            
            # Extract epoch-level metrics
            for key in self.metrics_history.keys():
                if key in metrics:
                    self.metrics_history[key] = metrics[key]
        except Exception as e:
            print(f"Warning: Could not extract metrics from logger: {e}")
            print("Will attempt to read from CSV files if available.")
    
    def collect_predictions(self, trainer, model, datamodule, split='test', return_images=False):
        """
        Collect predictions and ground truth labels for a given split.
        
        Args:
            trainer: PyTorch Lightning Trainer
            model: Trained model
            datamodule: Data module
            split: 'val' or 'test'
            return_images: Whether to return the input images as well
            
        Returns:
            Tuple of (y_true, y_score, [images]) as numpy arrays
        """
        print(f"\nCollecting predictions for {split} set...")
        
        # Get the appropriate dataloader
        # Set num_workers=0 for visualization to avoid "Too many open files" error
        if split == 'val':
            dataloader = datamodule.val_dataloader()
        elif split == 'test':
            dataloader = datamodule.test_dataloader()
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Override num_workers to 0 for the temporary dataloader
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=dataloader.pin_memory
        )
        
        # Set model to eval mode
        model.eval()
        
        y_true_list = []
        y_score_list = []
        images_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                
                # Move to same device as model
                device = next(model.parameters()).device
                x = x.to(device)
                y = y.to(device)
                
                # Get predictions
                logits = model(x).squeeze(1)
                probs = torch.sigmoid(logits)
                
                # Collect
                y_true_list.append(y.cpu().numpy())
                y_score_list.append(probs.cpu().numpy())
                if return_images:
                    images_list.append(x.cpu().numpy())
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_list)
        y_score = np.concatenate(y_score_list)
        
        print(f"Collected {len(y_true)} predictions for {split} set")
        
        if return_images:
            images = np.concatenate(images_list)
            return y_true, y_score, images
        
        return y_true, y_score
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Find threshold that maximizes F1 score.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.linspace(0, 1, 101)
        f1_scores = []
        
        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal threshold (max F1): {optimal_threshold:.3f} (F1={f1_scores[optimal_idx]:.4f})")
        
        return optimal_threshold
    
    def plot_learning_curves(self, metrics_dict: Dict[str, List[float]]):
        """
        Plot training and validation learning curves.
        
        Args:
            metrics_dict: Dictionary containing metric histories
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        if 'train_loss' in metrics_dict and len(metrics_dict['train_loss']) > 0:
            epochs = range(1, len(metrics_dict['train_loss']) + 1)
            axes[0].plot(epochs, metrics_dict['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if 'val_loss' in metrics_dict and len(metrics_dict['val_loss']) > 0:
                axes[0].plot(epochs, metrics_dict['val_loss'], 'r-', label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title(f'Learning Curves - Loss ({self.version})', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        if 'train_acc' in metrics_dict and len(metrics_dict['train_acc']) > 0:
            epochs = range(1, len(metrics_dict['train_acc']) + 1)
            axes[1].plot(epochs, metrics_dict['train_acc'], 'b-', label='Train Acc', linewidth=2)
            if 'val_acc' in metrics_dict and len(metrics_dict['val_acc']) > 0:
                axes[1].plot(epochs, metrics_dict['val_acc'], 'r-', label='Val Acc', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title(f'Learning Curves - Accuracy ({self.version})', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.plot_dir / 'learning_curves.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved learning curves to {save_path}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, split='test'):
        """
        Plot ROC curve with AUROC annotation.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            split: Dataset split name
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {split.capitalize()} Set ({self.version})', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        save_path = self.plot_dir / f'roc_curve_{split}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC curve to {save_path}")
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray, split='test'):
        """
        Plot Precision-Recall curve with Average Precision annotation.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            split: Dataset split name
        """
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {split.capitalize()} Set ({self.version})', 
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        save_path = self.plot_dir / f'pr_curve_{split}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved PR curve to {save_path}")
        
        return avg_precision
    
    def plot_confusion_matrices(self, y_true: np.ndarray, y_score: np.ndarray, 
                                 optimal_threshold: Optional[float] = None, split='test'):
        """
        Plot confusion matrices at default (0.5) and optimal thresholds.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            optimal_threshold: Optimal threshold (if None, will compute)
            split: Dataset split name
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Default threshold (0.5)
        y_pred_default = (y_score >= 0.5).astype(int)
        cm_default = confusion_matrix(y_true, y_pred_default)
        
        # Optimal threshold
        if optimal_threshold is None:
            optimal_threshold = self.find_optimal_threshold(y_true, y_score)
        y_pred_optimal = (y_score >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(y_true, y_pred_optimal)
        
        # Plot default
        im1 = axes[0].imshow(cm_default, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0].set_title(f'Confusion Matrix (threshold=0.5)\n{split.capitalize()} Set', 
                          fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Bogus', 'Real'])
        axes[0].set_yticklabels(['Bogus', 'Real'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[0].text(j, i, cm_default[i, j],
                                   ha="center", va="center", color="white" if cm_default[i, j] > cm_default.max() / 2 else "black",
                                   fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot optimal
        im2 = axes[1].imshow(cm_optimal, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1].set_title(f'Confusion Matrix (threshold={optimal_threshold:.3f})\n{split.capitalize()} Set', 
                          fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['Bogus', 'Real'])
        axes[1].set_yticklabels(['Bogus', 'Real'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[1].text(j, i, cm_optimal[i, j],
                                   ha="center", va="center", color="white" if cm_optimal[i, j] > cm_optimal.max() / 2 else "black",
                                   fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        save_path = self.plot_dir / f'confusion_matrices_{split}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrices to {save_path}")
    
    def plot_probability_histogram(self, y_true: np.ndarray, y_score: np.ndarray, split='test'):
        """
        Plot histogram of predicted probabilities for positive/negative classes.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            split: Dataset split name
        """
        plt.figure(figsize=(10, 6))
        
        # Separate by class
        bogus_scores = y_score[y_true == 0]
        real_scores = y_score[y_true == 1]
        
        # Plot histograms
        plt.hist(bogus_scores, bins=50, alpha=0.6, label=f'Bogus (n={len(bogus_scores)})', 
                 color='blue', edgecolor='black')
        plt.hist(real_scores, bins=50, alpha=0.6, label=f'Real (n={len(real_scores)})', 
                 color='red', edgecolor='black')
        
        plt.xlabel('Predicted Probability (Real)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Predicted Probability Distribution - {split.capitalize()} Set ({self.version})', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Default Threshold (0.5)')
        
        # Use log scale on Y-axis as requested
        plt.yscale('log')
        
        save_path = self.plot_dir / f'probability_histogram_{split}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved probability histogram to {save_path}")
        
    def plot_prediction_grid(self, images: np.ndarray, y_true: np.ndarray, y_score: np.ndarray, 
                             indices: np.ndarray, title: str, filename: str, grid_size=(8, 8)):
        """
        Plot a grid of snapshots with labels and predicted scores.
        """
        n_rows, n_cols = grid_size
        n_samples = n_rows * n_cols
        
        if len(indices) < n_samples:
            # Adjust grid if fewer samples
            n_samples = len(indices)
            n_rows = (n_samples + n_cols - 1) // n_cols
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
        fig.suptitle(f"{title} ({self.version}, TEST SET)", fontsize=20, fontweight='bold', y=0.95)
        
        axes_flat = axes.flatten() if n_samples > 1 else [axes]
        
        for i in range(n_samples):
            idx = indices[i]
            img = images[idx].squeeze() # (C, H, W) -> (H, W)
            target = y_true[idx]
            score = y_score[idx]
            
            # Plot image
            axes_flat[i].imshow(img, cmap='gray', origin='lower')
            axes_flat[i].axis('off')
            
            # Predict label (0.5 threshold)
            pred_label = 1 if score >= 0.5 else 0
            correct = (pred_label == target)
            
            label_text = "Real" if target == 1 else "Bogus"
            color = "green" if correct else "red"
            mark = "✓" if correct else "✗"
            
            axes_flat[i].set_title(f"T:{label_text} P:{score:.3f}\n{mark}", 
                                   color=color, fontsize=10, fontweight='bold')
            
        # Hide empty axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        save_path = self.plot_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved snapshot grid to {save_path}")

    def plot_leakage_check_grids(self, images: np.ndarray, y_true: np.ndarray, y_score: np.ndarray, split='test'):
        """
        Generates specific grids for leakage check: random, high-confidence, ambiguous.
        """
        print(f"\nGenerating Leakage Check Grids for {split} set...")
        
        # 1. Random subset
        rng = np.random.RandomState(42)
        n_total = len(y_true)
        random_indices = rng.choice(n_total, min(64, n_total), replace=False)
        self.plot_prediction_grid(images, y_true, y_score, random_indices, 
                                  "Random Samples", f"{split}_snapshots_random.png")
        
        # 2. High-confidence Real
        real_indices = np.where(y_true == 1)[0]
        if len(real_indices) > 0:
            # Sort by score descending
            sorted_real = real_indices[np.argsort(y_score[real_indices])[::-1]]
            self.plot_prediction_grid(images, y_true, y_score, sorted_real[:64], 
                                      "Top Confidence Real", f"{split}_snapshots_high_conf_real.png")
            
        # 3. High-confidence Bogus
        bogus_indices = np.where(y_true == 0)[0]
        if len(bogus_indices) > 0:
            # Sort by score ascending
            sorted_bogus = bogus_indices[np.argsort(y_score[bogus_indices])]
            self.plot_prediction_grid(images, y_true, y_score, sorted_bogus[:64], 
                                      "Top Confidence Bogus", f"{split}_snapshots_high_conf_bogus.png")
            
        # 4. Most Ambiguous (near 0.5)
        ambiguous_scores = np.abs(y_score - 0.5)
        ambiguous_indices = np.argsort(ambiguous_scores)[:64]
        self.plot_prediction_grid(images, y_true, y_score, ambiguous_indices, 
                                  "Most Ambiguous Samples", f"{split}_snapshots_ambiguous.png")
        
        # 5. Real Among Bogus (y_true=0, high y_score)
        bogus_indices = np.where(y_true == 0)[0]
        if len(bogus_indices) > 0:
            # Sort by score descending (most likely to be real)
            real_among_bogus = bogus_indices[np.argsort(y_score[bogus_indices])[::-1]]
            self.plot_prediction_grid(images, y_true, y_score, real_among_bogus[:64], 
                                      "Real Among Bogus (Suspicious)", "real_among_bogus_snapshot_mosaic.png")
    
    def generate_all_plots(self, y_true_val: np.ndarray, y_score_val: np.ndarray,
                           y_true_test: np.ndarray, y_score_test: np.ndarray,
                           metrics_dict: Dict[str, List[float]]):
        """
        Generate all visualization plots.
        
        Args:
            y_true_val: Validation ground truth
            y_score_val: Validation predictions
            y_true_test: Test ground truth
            y_score_test: Test predictions
            metrics_dict: Training metrics history
        """
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Learning curves
        self.plot_learning_curves(metrics_dict)
        
        # Find optimal threshold on validation set
        optimal_threshold = self.find_optimal_threshold(y_true_val, y_score_val)
        
        # Test set visualizations
        roc_auc = self.plot_roc_curve(y_true_test, y_score_test, split='test')
        avg_precision = self.plot_precision_recall_curve(y_true_test, y_score_test, split='test')
        self.plot_confusion_matrices(y_true_test, y_score_test, optimal_threshold, split='test')
        self.plot_probability_histogram(y_true_test, y_score_test, split='test')
        
        # Validation set visualizations (optional, for completeness)
        self.plot_roc_curve(y_true_val, y_score_val, split='val')
        self.plot_precision_recall_curve(y_true_val, y_score_val, split='val')
        self.plot_confusion_matrices(y_true_val, y_score_val, optimal_threshold, split='val')
        self.plot_probability_histogram(y_true_val, y_score_val, split='val')
        
        print("="*60 + "\n")
        
        return {
            'optimal_threshold': optimal_threshold,
            'test_auroc': roc_auc,
            'test_avg_precision': avg_precision
        }

    def save_predictions_csv(self, y_true: np.ndarray, y_score: np.ndarray, 
                             file_paths: Optional[List[str]], split='test'):
        """
        Save predictions to a CSV file.
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities
            file_paths: List of filenames/paths
            split: 'val' or 'test'
        """
        import pandas as pd
        
        # Predicted labels (default threshold 0.5)
        y_pred = (y_score >= 0.5).astype(int)
        
        # Handle missing file_paths
        if file_paths is None or len(file_paths) != len(y_true):
            print(f"Warning: File paths counts mismatch or missing for {split}. Using dummy identifiers.")
            file_paths = [f"sample_{i}" for i in range(len(y_true))]
        
        df = pd.DataFrame({
            'filename': file_paths,
            'score': y_score,
            'predicted_label': y_pred,
            'true_label': y_true
        })
        
        csv_filename = f'validation_predictions.csv' if split == 'val' else f'test_predictions.csv'
        csv_path = self.data_dir / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"Saved {split} predictions to {csv_path}")
    
    def save_metrics_json(self, metrics: Dict, args, best_ckpt_path: str):
        """
        Save metrics summary to JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            args: Training arguments
            best_ckpt_path: Path to best checkpoint
        """
        summary = {
            'version': self.version,
            'hyperparameters': {
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'max_epochs': args.max_epochs,
                'num_workers': args.num_workers,
                'balance_data': args.balance_data
            },
            'best_checkpoint': str(best_ckpt_path),
            'metrics': metrics,
            'output_paths': {
                'plots': str(self.plot_dir),
                'results': str(self.results_dir)
            }
        }
        
        self.output_manager.write_metrics(summary)
        
        print(f"Saved metrics summary to {self.output_manager.run_dir / 'metrics.json'}")
    
    def generate_report_md(self, metrics: Dict, args, best_ckpt_path: str):
        """
        Generate a markdown report summarizing the experiment.
        
        Args:
            metrics: Dictionary of metrics
            args: Training arguments
            best_ckpt_path: Path to best checkpoint
        """
        report_lines = [
            f"# Experiment Report: {self.version}",
            "",
            "## Configuration",
            "",
            "### Dataset",
            f"- **Data Directory**: `{args.data_dir}`",
            f"- **Balance Data**: {args.balance_data}",
            "",
            "### Training Hyperparameters",
            f"- **Batch Size**: {args.batch_size}",
            f"- **Learning Rate**: {args.lr}",
            f"- **Max Epochs**: {args.max_epochs}",
            f"- **Num Workers**: {args.num_workers}",
            "",
            "## Model Checkpoint",
            f"- **Best Checkpoint**: `{best_ckpt_path}`",
            "",
            "## Validation Metrics",
            ""
        ]
        
        # Add validation metrics if available
        if 'val_loss' in metrics:
            report_lines.append(f"- **Val Loss**: {metrics['val_loss']:.4f}")
        if 'val_acc' in metrics:
            report_lines.append(f"- **Val Accuracy**: {metrics['val_acc']:.4f}")
        if 'val_f1' in metrics:
            report_lines.append(f"- **Val F1**: {metrics['val_f1']:.4f}")
        if 'val_auroc' in metrics:
            report_lines.append(f"- **Val AUROC**: {metrics['val_auroc']:.4f}")
        
        report_lines.extend([
            "",
            "## Test Metrics",
            ""
        ])
        
        # Add test metrics
        if 'test_loss' in metrics:
            report_lines.append(f"- **Test Loss**: {metrics['test_loss']:.4f}")
        if 'test_acc' in metrics:
            report_lines.append(f"- **Test Accuracy**: {metrics['test_acc']:.4f}")
        if 'test_f1' in metrics:
            report_lines.append(f"- **Test F1**: {metrics['test_f1']:.4f}")
        if 'test_prec' in metrics:
            report_lines.append(f"- **Test Precision**: {metrics['test_prec']:.4f}")
        if 'test_rec' in metrics:
            report_lines.append(f"- **Test Recall**: {metrics['test_rec']:.4f}")
        if 'test_auroc' in metrics:
            report_lines.append(f"- **Test AUROC**: {metrics['test_auroc']:.4f}")
        if 'test_avg_precision' in metrics:
            report_lines.append(f"- **Test Average Precision**: {metrics['test_avg_precision']:.4f}")
        if 'optimal_threshold' in metrics:
            report_lines.append(f"- **Optimal Threshold (max F1 on val)**: {metrics['optimal_threshold']:.3f}")
        
        report_lines.extend([
            "",
            "## Generated Plots",
            "",
            f"All plots are saved in: `{self.plot_dir}`",
            "",
            "### Available Visualizations",
            "- `learning_curves.png` - Training and validation loss/accuracy over epochs",
            "- `roc_curve_test.png` - ROC curve for test set",
            "- `roc_curve_val.png` - ROC curve for validation set",
            "- `pr_curve_test.png` - Precision-Recall curve for test set",
            "- `pr_curve_val.png` - Precision-Recall curve for validation set",
            "- `confusion_matrices_test.png` - Confusion matrices at default and optimal thresholds (test)",
            "- `confusion_matrices_val.png` - Confusion matrices at default and optimal thresholds (val)",
            "- `probability_histogram_test.png` - Distribution of predicted probabilities (test)",
            "- `probability_histogram_val.png` - Distribution of predicted probabilities (val)",
            "- `test_snapshots_random.png` - Random snapshots from TEST set for leakage check",
            "- `test_snapshots_high_conf_real.png` - Highest confidence Real snapshots for leakage check",
            "- `test_snapshots_high_conf_bogus.png` - Highest confidence Bogus snapshots for leakage check",
            "- `test_snapshots_ambiguous.png` - Most ambiguous snapshots (scores near 0.5) for leakage check",
            "",
            "---",
            f"*Report generated for: {self.output_manager.run_dir}*"
        ])
        
        report_path = self.data_dir / 'report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved experiment report to {report_path}")


def visualize_training_results(trainer, model, datamodule, args, output_manager: OutputManager, final_metrics: Optional[Dict] = None):
    """
    Main function to generate all visualizations and reports.
    
    Args:
        trainer: PyTorch Lightning Trainer
        model: Trained model
        datamodule: Data module
        args: Training arguments
        output_manager: Unified OutputManager instance
    """
    visualizer = ResultsVisualizer(output_manager)
    
    # Collect predictions
    y_true_val, y_score_val = visualizer.collect_predictions(trainer, model, datamodule, split='val')
    y_true_test, y_score_test, images_test = visualizer.collect_predictions(trainer, model, datamodule, split='test', return_images=True)
    
    # Leakage check grids (Mandatory for TEST set)
    visualizer.plot_leakage_check_grids(images_test, y_true_test, y_score_test, split='test')
    
    # Save CSV predictions
    test_paths = datamodule.test_dataset.file_paths
    visualizer.save_predictions_csv(y_true_test, y_score_test, test_paths, split='test')
    
    # Optional: Leakage check for VAL set (for comparison)
    y_true_val, y_score_val, images_val = visualizer.collect_predictions(trainer, model, datamodule, split='val', return_images=True)
    visualizer.plot_leakage_check_grids(images_val, y_true_val, y_score_val, split='val')
    
    # Save CSV predictions (val)
    val_paths = datamodule.val_dataset.file_paths
    visualizer.save_predictions_csv(y_true_val, y_score_val, val_paths, split='val')
    
    # Try to extract metrics from logger or use trainer's logged metrics
    metrics_dict = {}
    try:
        # Get metrics from trainer's callback metrics
        if hasattr(trainer, 'logged_metrics'):
            logged = trainer.logged_metrics
            # Extract final values
            metrics_dict = {
                'val_loss': logged.get('val_loss', 0.0),
                'val_acc': logged.get('val_acc', 0.0),
                'val_f1': logged.get('val_f1', 0.0),
                'val_auroc': logged.get('val_auroc', 0.0),
                'test_loss': logged.get('test_loss', 0.0),
                'test_acc': logged.get('test_acc', 0.0),
                'test_f1': logged.get('test_f1', 0.0),
                'test_prec': logged.get('test_prec', 0.0),
                'test_rec': logged.get('test_rec', 0.0),
                'test_auroc': logged.get('test_auroc', 0.0)
            }
            # Convert tensors to floats
            metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                           for k, v in metrics_dict.items()}
    except Exception as e:
        print(f"Warning: Could not extract logged metrics: {e}")
    
    # Use final_metrics if provided (overwrites logged metrics with more accurate values)
    if final_metrics:
        print("Using provided final validation/test metrics for the report.")
        metrics_dict.update(final_metrics)
        # Convert tensors to floats
        metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in metrics_dict.items()}
    
    # Read training history from TensorBoard logs if available
    training_history = {}
    try:
        import pandas as pd
        from tensorboard.backend.event_processing import event_accumulator
        
        log_dir = output_manager.get_path('logs') / args.version
        # Find the event file
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        if event_files:
            ea = event_accumulator.EventAccumulator(str(event_files[0].parent))
            ea.Reload()
            
            # Extract scalars
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                training_history[tag] = [e.value for e in events]
    except Exception as e:
        print(f"Note: Could not read TensorBoard logs for learning curves: {e}")
        print("Learning curves will be skipped if history is not available.")
    
    # Generate all plots
    plot_metrics = visualizer.generate_all_plots(
        y_true_val, y_score_val,
        y_true_test, y_score_test,
        training_history
    )
    
    # Merge metrics
    all_metrics = {**metrics_dict, **plot_metrics}
    
    # Get best checkpoint path
    best_ckpt_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else "N/A"
    
    # Save JSON summary
    visualizer.save_metrics_json(all_metrics, args, best_ckpt_path)
    
    # Generate markdown report
    visualizer.generate_report_md(all_metrics, args, best_ckpt_path)
    
    print("\n" + "="*60)
    print("VISUALIZATION AND REPORTING COMPLETE")
    print("="*60)
    print(f"Plots saved to: {visualizer.plot_dir}")
    print(f"Results saved to: {visualizer.data_dir}")
    print("="*60 + "\n")
