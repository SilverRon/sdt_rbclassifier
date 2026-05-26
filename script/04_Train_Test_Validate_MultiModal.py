
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Library
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
import os, sys
import argparse
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import RBDataModule
from src.model import RBClassificationModule
from src.visualize_results import visualize_training_results

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Main
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def main(args):
    # Fix for "Too many open files" error on Linux
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Fixed Seed
    pl.seed_everything(42)
    
    # Optimization for A100 Tensor Cores
    torch.set_float32_matmul_precision('medium')
    print(f"Set float32 matmul precision to 'medium' for Tensor Cores")

    # Device information
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()} (0-indexed)")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"GPU will be used for: Model training")
        print(f"CPU will be used for: Data loading (parallel workers)")
    print("="*60 + "\n")

    # 1. Data Module
    print("="*60)
    print("DATA MODULE INITIALIZATION")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Balance data: {args.balance_data}")
    print("="*60 + "\n")
    
    dm = RBDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_data=args.balance_data
    )
    # Setup to calculate pos_weight
    dm.setup() 
    
    # 2. Model
    print("\n" + "="*60)
    print("MODEL INITIALIZATION")
    print("="*60)
    print(f"Model architecture: {args.model}")
    pos_weight_val = dm.pos_weight.item() if isinstance(dm.pos_weight, torch.Tensor) else dm.pos_weight
    print(f"Learning rate: {args.lr}")
    print(f"Pos weight (for class imbalance): {pos_weight_val:.4f}")
    print("="*60 + "\n")
    
    model = RBClassificationModule(
        model_name=args.model,
        learning_rate=args.lr,
        pos_weight=pos_weight_val
    )

    # 3. Callbacks
    # Output paths relative to where script is
    script_dir = Path(__file__).parent
    output_base = script_dir.parent / "output"
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=str(output_base / "checkpoints" / args.version),
        filename='rb-classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 4. Logger
    logger = TensorBoardLogger(str(output_base / "logs"), name=args.version)

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        fast_dev_run=args.fast_dev_run
    )

    # 6. Fit
    print("Starting Training...")
    trainer.fit(model, dm)

    # 7. Test & Final Validation (to capture metrics for report)
    print("Starting Testing and Final Validation with best checkpoint...")
    ckpt_path = 'best' if not args.fast_dev_run else None
    
    # Run test and capture metrics
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)
    
    # Run validation and capture metrics (using best checkpoint)
    val_results = trainer.validate(model, datamodule=dm, ckpt_path=ckpt_path)
    
    # 8. Visualize Results (skip if fast_dev_run)
    if not args.fast_dev_run:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS AND REPORTS")
        print("="*60)
        
        # Explicitly load best checkpoint for manual prediction gathering
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            print(f"Loading best checkpoint for visualizations: {trainer.checkpoint_callback.best_model_path}")
            model = RBClassificationModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
        # Merge results for the visualizer
        # results are lists of dicts
        metrics = {}
        if test_results:
            metrics.update(test_results[0])
        if val_results:
            metrics.update(val_results[0])
            
        visualize_training_results(trainer, model, dm, args, output_base, final_metrics=metrics)
    else:
        print("\nSkipping visualization (fast_dev_run mode)")

if __name__ == "__main__":
    # Determine default data_dir relative to this script
    script_dir = Path(__file__).parent
    default_data_dir = script_dir.parent / "data" / "norm_snapshot"

    parser = argparse.ArgumentParser(description="Train Test Validate RB Classification")
    parser.add_argument("--version", type=str, default="v1", help="Experiment version name")
    parser.add_argument("--model", type=str, default="simplecnn", 
                        choices=["simplecnn", "resnet18", "resnet34", "resnet50", 
                                 "efficientnet_b0", "efficientnet_b1", "convnext_tiny", 
                                 "mobilenet_v3_small", "mobilenet_v3_large", "shufflenet_v2", "otrain"], 
                        help="Model architecture")
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir), help="Path to normalized snapshots")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train/val/test batch for debugging")
    parser.add_argument("--balance_data", action="store_true", help="Balance dataset by downsampling majority class (Real) to match Bogus count")
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = script_dir.parent / "output"
    os.makedirs(output_dir / "checkpoints" / args.version, exist_ok=True)
    os.makedirs(output_dir / "logs", exist_ok=True)
    os.makedirs(output_dir / "plots" / args.version, exist_ok=True)
    os.makedirs(output_dir / "results" / args.version, exist_ok=True)
    
    main(args)
