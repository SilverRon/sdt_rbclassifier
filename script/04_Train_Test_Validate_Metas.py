# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Library
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
import os
import sys
import glob
import json
import argparse
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.output_paths import OutputManager
from collections import Counter
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Optional Libraries (Graceful Failures)
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import optuna
except ImportError:
    optuna = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    torch = None
    pl = None

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Data Handler
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
class RBTabularDataModule:
    def __init__(self, data_dir, split_by='tile', valid_ratio=0.2, test_ratio=0.2, seed=42, balance=False, features=None, smoke_test=False):
        self.data_dir = data_dir
        self.split_by = split_by
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.balance = balance
        self.features = features
        self.smoke_test = smoke_test
        self.target_col = 'label_binary'
        
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.feature_names = []
        
        self.imputer = None
        self.scaler = None

    def prepare_data(self):
        # Check for parquet dependencies early
        try:
            import pyarrow
        except ImportError:
            try:
                import fastparquet
            except ImportError:
                logger.critical("No parquet engine found (pyarrow or fastparquet). please install one.")
                sys.exit(1)

        logger.info(f"Loading data from {self.data_dir}...")
        tiles = sorted([os.path.basename(p) for p in glob.glob(os.path.join(self.data_dir, "T*"))])
        if not tiles:
             raise ValueError(f"No tiles found in {self.data_dir}")
        
        if self.smoke_test:
            logger.info("SMOKE TEST MODE: Limiting to first 5 tiles to ensure valid splits.")
            tiles = tiles[:5]

        logger.info(f"Found {len(tiles)} tiles.")

        # Collect file paths
        tile_data_map = {}
        for tile in tiles:
            r_path = os.path.join(self.data_dir, tile, "real_ml.parquet")
            b_path = os.path.join(self.data_dir, tile, "bogus_ml.parquet")
            
            if os.path.exists(r_path) and os.path.exists(b_path):
                 tile_data_map[tile] = {'real': r_path, 'bogus': b_path}
        
        if self.split_by == 'tile':
             self._split_and_load_by_tile(tile_data_map)
        else:
             self._load_all_and_random_split(tile_data_map)
        
        # Check if any data was loaded
        if self.train_df is None or self.train_df.empty:
            logger.critical("No data was loaded. Check your data paths and parquet engines.")
            sys.exit(1)

        # Post-processing
        self._post_process_splits()
        
        # Log dataset sizes
        logger.info(f"Data Splits - Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")

    def _load_files(self, file_paths, label):
        dfs = []
        for p in file_paths:
            try:
                df = pd.read_parquet(p)
                if self.smoke_test:
                    df = df.head(500) # Reduced reduction for smoke test to ensure we have enough data
                df['tile'] = os.path.basename(os.path.dirname(p))
                df[self.target_col] = label
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _split_and_load_by_tile(self, tile_data_map):
        tiles = list(tile_data_map.keys())
        np.random.seed(self.seed)
        np.random.shuffle(tiles)
        
        # Ensure we have at least 1 tile for each split if we have enough tiles
        # Otherwise, prioritize Train > Val > Test
        if len(tiles) >= 3:
            n_test = max(1, int(len(tiles) * self.test_ratio))
            n_val = max(1, int(len(tiles) * self.valid_ratio))
        elif len(tiles) == 2:
            n_test = 0
            n_val = 1
        else: # len == 1
            n_test = 0
            n_val = 0
        
        test_tiles = tiles[:n_test]
        val_tiles = tiles[n_test:n_test+n_val]
        train_tiles = tiles[n_test+n_val:]
        
        splits = {'train': train_tiles, 'val': val_tiles, 'test': test_tiles}
        dfs = {}
        
        for split_name, split_tiles in splits.items():
            r_paths = [tile_data_map[t]['real'] for t in split_tiles]
            b_paths = [tile_data_map[t]['bogus'] for t in split_tiles]
            
            df_r = self._load_files(r_paths, 1) # Real = 1
            df_b = self._load_files(b_paths, 0) # Bogus = 0
            
            dfs[split_name] = pd.concat([df_r, df_b], ignore_index=True)

        self.train_df = dfs['train']
        self.val_df = dfs['val']
        self.test_df = dfs['test']

    def _load_all_and_random_split(self, tile_data_map):
        logger.info("Loading all data first for random split...")
        r_paths = [v['real'] for v in tile_data_map.values()]
        b_paths = [v['bogus'] for v in tile_data_map.values()]
        
        df_r = self._load_files(r_paths, 1)
        df_b = self._load_files(b_paths, 0)
        
        full_df = pd.concat([df_r, df_b], ignore_index=True)
        
        train_val, test = train_test_split(full_df, test_size=self.test_ratio, random_state=self.seed, stratify=full_df[self.target_col])
        train, val = train_test_split(train_val, test_size=self.valid_ratio / (1 - self.test_ratio), random_state=self.seed, stratify=train_val[self.target_col])
        
        self.train_df = train
        self.val_df = val
        self.test_df = test

    def _post_process_splits(self):
        # 1. Feature Selection
        exclude_cols = ['tile', 'label_binary', 'label', 'snapshot']
        
        if self.features is None:
             candidates = [c for c in self.train_df.columns if c not in exclude_cols]
             self.feature_names = [c for c in candidates if pd.api.types.is_numeric_dtype(self.train_df[c])]
        else:
             self.feature_names = self.features

        logger.info(f"Selected {len(self.feature_names)} features.")
        
        # 2. Balancing (Train only)
        if self.balance:
            logger.info("Balancing training data (undersampling majority class)...")
            counts = self.train_df[self.target_col].value_counts()
            min_count = counts.min()
            
            df_list = []
            for label in counts.index:
                sub = self.train_df[self.train_df[self.target_col] == label]
                if len(sub) > min_count:
                    sub = sub.sample(n=min_count, random_state=self.seed)
                df_list.append(sub)
            self.train_df = pd.concat(df_list, ignore_index=True).sample(frac=1, random_state=self.seed).reset_index(drop=True)
            logger.info(f"Balanced Train: {self.train_df[self.target_col].value_counts().to_dict()}")

        # 3. Handling NaN (Simple Imputation)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        X_train = self.train_df[self.feature_names].values
        self.imputer.fit(X_train)
        X_train = self.imputer.transform(X_train)
        self.scaler.fit(X_train) 
        
    def get_data(self, split='train', return_tensor=False):
        if split == 'train':
            df = self.train_df
        elif split == 'val':
            df = self.val_df
        elif split == 'test':
            df = self.test_df
        else:
            raise ValueError(f"Unknown split {split}")
            
        X = df[self.feature_names].values
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        y = df[self.target_col].values
        
        if return_tensor and torch is not None:
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        return X, y

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Models
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
class ModelWrapper:
    def __init__(self, model_name, device='cpu', n_jobs=-1, seed=42):
        self.model_name = model_name
        self.device = device
        self.n_jobs = n_jobs
        self.seed = seed
        self.model = None

    def build(self, params=None):
        if params is None:
            params = {}
            
        if self.model_name == 'rf':
            self.model = RandomForestClassifier(
                n_jobs=self.n_jobs, random_state=self.seed, **params
            )
        elif self.model_name == 'iso':
             self.model = IsolationForest(
                 n_jobs=self.n_jobs, random_state=self.seed, **params
             )
        elif self.model_name == 'lgbm':
            if lgb is None: raise ImportError("LightGBM not installed")
            if self.device == 'gpu':
                params['device'] = 'gpu'
                # Check if GPU is actually available for LGBM might need specific build
            self.model = lgb.LGBMClassifier(
                n_jobs=self.n_jobs, random_state=self.seed, **params
            )
        elif self.model_name == 'xgb':
            if xgb is None: raise ImportError("XGBoost not installed")
            tree_method = 'hist'
            xgb_device = 'cpu'
            if self.device == 'gpu':
                xgb_device = 'cuda'
            self.model = xgb.XGBClassifier(
                n_jobs=self.n_jobs, random_state=self.seed, tree_method=tree_method, device=xgb_device, **params
            )
        elif self.model_name == 'cat':
            if cb is None: raise ImportError("CatBoost not installed")
            task_type = 'GPU' if self.device == 'gpu' else 'CPU'
            self.model = cb.CatBoostClassifier(
                thread_count=self.n_jobs, random_seed=self.seed, task_type=task_type, verbose=0, **params
            )
        else:
            raise ValueError(f"Unknown model {self.model_name}")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.model_name == 'iso':
             self.model.fit(X_train)
        elif self.model_name in ['lgbm', 'xgb', 'cat']:
             eval_set = [(X_val, y_val)] if X_val is not None else None
             if self.model_name == 'lgbm':
                  self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=[
                      lgb.early_stopping(10, verbose=False)
                  ] if eval_set else None)
             elif self.model_name == 'xgb':
                  self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
             elif self.model_name == 'cat':
                  self.model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10)
        else:
             self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        if self.model_name == 'iso':
            scores = self.model.decision_function(X) 
            probs = 1 / (1 + np.exp(-scores))
            return np.vstack([1-probs, probs]).T
        else:
            return self.model.predict_proba(X)

    def predict(self, X):
        if self.model_name == 'iso':
            preds = self.model.predict(X) 
            return np.where(preds == 1, 1, 0)
        return self.model.predict(X)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# MLP (PyTorch Lightning)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
if pl is not None:
    class TabularMLP(pl.LightningModule):
        def __init__(self, input_dim, hidden_dims=[128, 64], lr=1e-3, pos_weight=None):
            super().__init__()
            self.save_hyperparameters()
            layers = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
            self.pos_weight = torch.tensor(pos_weight) if pos_weight else None

        def forward(self, x):
            return self.net(x).squeeze(-1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            if self.pos_weight is not None:
                 self.pos_weight = self.pos_weight.to(self.device)
            loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            if self.pos_weight is not None:
                 self.pos_weight = self.pos_weight.to(self.device)
            loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self.pos_weight)
            self.log('val_loss', loss, prog_bar=True)
            preds = torch.sigmoid(logits)
            acc = ((preds > 0.5) == y).float().mean()
            self.log('val_acc', acc, prog_bar=True)
            
        def predict_step(self, batch, batch_idx, dataloader_idx=0):
             x, y = batch
             logits = self(x)
             return torch.sigmoid(logits)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Tuning
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def objective(trial, model_name, dm, device='cpu'):
    params = {}
    if model_name == 'rf':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
    elif model_name == 'lgbm':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 0.5, log=True)
        params['num_leaves'] = trial.suggest_int('num_leaves', 20, 150)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
    elif model_name == 'xgb':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 0.5, log=True)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    elif model_name == 'cat':
        params['iterations'] = trial.suggest_int('iterations', 50, 500)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 0.5, log=True)
        params['depth'] = trial.suggest_int('depth', 3, 10)
    elif model_name == 'mlp': 
        # Tuning support for MLP
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        h_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
        params['hidden_dims'] = [h_dim, h_dim // 2]
    
    if model_name == 'mlp':
         # Avoid full tuning during simplified view, but here is placeholder logic
         return 0.5 
    
    wrapper = ModelWrapper(model_name, device=device)
    wrapper.build(params)
    
    X_train, y_train = dm.get_data('train')
    X_val, y_val = dm.get_data('val')
    
    wrapper.fit(X_train, y_train, X_val, y_val)
    
    probs = wrapper.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, probs)
    return score

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Pipeline Driver
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
class PipelineDriver:
    def __init__(self, args):
        self.args = args
        self.output_manager = OutputManager(pipeline='meta', run_name=args.version)
        self.output_dir = self.output_manager.run_dir
        
        # Save Args
        self.output_manager.write_args(vars(args))
            
    def run(self):
        logger.info(f"=== Experiment: {self.args.version} | Model: {self.args.model} | Device: {self.args.device} ===")
        
        # Data
        dm = RBTabularDataModule(
            data_dir=self.args.data_dir,
            split_by=self.args.split_by,
            seed=self.args.seed,
            balance=self.args.balance,
            smoke_test=self.args.smoke_test
        )
        dm.prepare_data()
        
        # Save features
        self.output_manager.write_features(dm.feature_names)

        if self.args.model == 'mlp':
            self.run_mlp(dm)
        else:
            self.run_sklearn(dm)
            
    def run_sklearn(self, dm):
        do_tune = (self.args.tune == "true") and not self.args.smoke_test
        best_params = {}
        
        if do_tune and optuna is not None:
            logger.info("Starting Optuna Tuning...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, self.args.model, dm, self.args.device), n_trials=self.args.n_trials)
            best_params = study.best_params
            logger.info(f"Best Params: {best_params}")
            self.output_manager.write_json(best_params, "best_params.json")
        
        elif self.args.smoke_test:
            # Force fast params
            if self.args.model in ['rf', 'lgbm', 'xgb']:
                best_params = {'n_estimators': 10, 'max_depth': 3}
            elif self.args.model == 'cat':
                best_params = {'iterations': 10, 'depth': 3}

        logger.info("Training Final Model...")
        wrapper = ModelWrapper(self.args.model, device=self.args.device, n_jobs=self.args.n_workers, seed=self.args.seed)
        wrapper.build(best_params)
        
        X_train, y_train = dm.get_data('train')
        X_val, y_val = dm.get_data('val')
        X_test, y_test = dm.get_data('test')
        
        wrapper.fit(X_train, y_train, X_val, y_val)
        
        with open(self.output_dir / "model.pkl", "wb") as f:
            pickle.dump(wrapper.model, f)
            
        logger.info("Evaluating...")
        y_test_pred = wrapper.predict(X_test)
        y_test_prob = wrapper.predict_proba(X_test)[:, 1]
        
        self.save_metrics(y_test, y_test_pred, y_test_prob)

    def run_mlp(self, dm):
        if pl is None:
            raise ImportError("PyTorch Lightning not installed")

        # Fix for "Too many open files" error
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        pl.seed_everything(self.args.seed)
        
        # Helper to get loaders
        def get_loader(split, shuffle=False):
            X, y = dm.get_data(split, return_tensor=True)
            if self.args.smoke_test:
                 X, y = X[:1000], y[:1000]
            return DataLoader(TensorDataset(X, y), batch_size=self.args.batch_size, shuffle=shuffle, num_workers=self.args.n_workers)

        train_loader = get_loader('train', shuffle=True)
        val_loader = get_loader('val')
        test_loader = get_loader('test')
        
        # Calculate pos_weight
        pos_weight = None
        if not self.args.balance:
            _, y_train = dm.get_data('train', return_tensor=True) # full data for stats
            if self.args.smoke_test: y_train = y_train[:1000]
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            if n_pos > 0:
                 pos_weight = (n_neg / n_pos).item()
        
        X_input_dim = dm.get_data('train', return_tensor=True)[0].shape[1]
        model = TabularMLP(input_dim=X_input_dim, lr=1e-3, pos_weight=pos_weight)
        
        # Callbacks
        checkpoint_cb = ModelCheckpoint(
            dirpath=self.output_manager.get_path('checkpoints'), filename='mlp-{epoch:02d}-{val_loss:.2f}', monitor='val_loss', mode='min', save_top_k=1
        )
        early_stop_cb = EarlyStopping(monitor='val_loss', patience=10 if not self.args.smoke_test else 2, mode='min')
        
        # Trainer
        accelerator = 'gpu' if self.args.device == 'gpu' and torch.cuda.is_available() else 'cpu'
        devices = 1
        max_epochs = self.args.max_epochs if not self.args.smoke_test else 2
        
        logger.info(f"MLP Training | Accelerator: {accelerator} | Devices: {devices}")
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=[checkpoint_cb, early_stop_cb],
            default_root_dir=self.output_manager.get_path('logs'),
            enable_progress_bar=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Test / Prediction SAFE MODE
        # We use trainer.predict to ensure lightning handles the device placement
        logger.info("Evaluating MLP using best checkpoint...")
        best_model_path = checkpoint_cb.best_model_path
        if not best_model_path:
             logger.warning("No best checkpoint found (maybe training failed or 1 epoch?), using last.")
             best_model_path = None # use current

        # Use trainer to predict. This matches devices automatically.
        preds_list = trainer.predict(model, dataloaders=test_loader, ckpt_path=best_model_path)
        y_test_prob = torch.cat(preds_list).cpu().numpy()
        
        # Get ground truth from test loader (order is preserved if not shuffled)
        y_test_true = []
        for batch in test_loader:
             _, y = batch
             y_test_true.append(y)
        y_test = torch.cat(y_test_true).numpy()
        
        y_test_pred = (y_test_prob > 0.5).astype(int)
        
        self.save_metrics(y_test, y_test_pred, y_test_prob)

    def save_metrics(self, y_true, y_pred, y_prob):
        metrics = {
            "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
            "f1_real": float(f1_score(y_true, y_pred, pos_label=1)),
            "f1_bogus": float(f1_score(y_true, y_pred, pos_label=0)),
            "auroc": float(roc_auc_score(y_true, y_prob)),
            "accuracy": float((y_true == y_pred).mean())
        }
        
        logger.info(f"Metrics: {metrics}")
        self.output_manager.write_metrics(metrics)
            
        print(classification_report(y_true, y_pred, target_names=['Bogus', 'Real']))
        
        # Plots
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bogus', 'Real'])
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix ({self.args.model})")
        plt.savefig(self.output_manager.get_path('plots', "confusion_matrix.png"))
        plt.close()
        
        logger.info("Done.")

def main():
    script_dir = Path(__file__).parent
    default_data_dir = script_dir.parent / "data" / "stacked_meta"
    
    parser = argparse.ArgumentParser(description="RB Tabular Classification Driver")
    parser.add_argument("--model", type=str, required=True, choices=['rf', 'lgbm', 'xgb', 'cat', 'iso', 'mlp'])
    parser.add_argument("--version", type=str, required=True, help="Experiment version identifier")
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir))
    parser.add_argument("--tune", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--split_by", type=str, default="tile", choices=["tile", "random"])
    parser.add_argument("--balance", action="store_true", help="Balance training data")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test with reduced data/params")
    
    args = parser.parse_args()
    PipelineDriver(args).run()

if __name__ == "__main__":
    main()
