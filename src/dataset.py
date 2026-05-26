
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import pytorch_lightning as pl
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from joblib import Parallel, delayed

def load_one_fits(path):
    try:
        with fits.open(path) as hdul:
            image = hdul[0].data.astype(np.float32)
            # Assumes image data is in the primary HDU or first extension
            # depending on file. Usually primary for these snapshots.
            
            # Use copy() to ensure it's in memory and file can be closed safely,
            # though context manager handles close.
            # Add channel dimension if missing: (1, height, width)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            return image
    except Exception as e:
        # print(f"Error loading {path}: {e}")
        # Return a zero tensor in case of error
        return np.zeros((1, 25, 25), dtype=np.float32)

class RBSnapshotDataset(Dataset):
    """
    Dataset for Real/Bogus snapshots.
    Reads FITS files.
    """
    def __init__(self, 
                 file_paths: Optional[List[str]] = None, 
                 labels: Optional[List[int]] = None, 
                 data: Optional[torch.Tensor] = None,
                 targets: Optional[torch.Tensor] = None,
                 transform=None):
        """
        Args:
            file_paths: List of paths to .fits files (optional if data provided)
            labels: List of integer labels (optional if targets provided)
            data: Pre-loaded tensor (N, C, H, W)
            targets: Pre-loaded targets (N,)
            transform: Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.data = data
        self.targets = targets
        self.transform = transform
        
        if self.data is None and self.file_paths is None:
            raise ValueError("Either data or file_paths must be provided.")

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.data is not None:
            image_tensor = self.data[idx]
            label_tensor = self.targets[idx]
        else:
            path = self.file_paths[idx]
            label = self.labels[idx]

            # Load FITS data
            image = load_one_fits(path)
            
            image_tensor = torch.from_numpy(image)
            label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_tensor

class RBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../data/norm_snapshot", batch_size: int = 128, num_workers: int = 8, balance_data: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_data = balance_data
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # Skip if already set up (prevents redundant loading when trainer calls setup again)
        if hasattr(self, 'train_dataset') and hasattr(self, 'val_dataset') and hasattr(self, 'test_dataset'):
            return
        
        # 1. glob all tiles in data_dir
        # Structure: data_dir/TXXXXX/sci/ and data_dir/TXXXXX/sub/
        # real = sci, bogus = sub
        
        # Structure: data_dir/TXXXXX/sci/ and data_dir/TXXXXX/sub/
        
        # Check if cache exists
        cache_dir = self.data_dir
        self.train_cache_path = os.path.join(cache_dir, "train_cache.pt")
        self.val_cache_path = os.path.join(cache_dir, "val_cache.pt")
        self.test_cache_path = os.path.join(cache_dir, "test_cache.pt")

        if os.path.exists(self.train_cache_path) and os.path.exists(self.val_cache_path) and os.path.exists(self.test_cache_path):
            print("Loading cached datasets from .pt files...")
            self.train_dataset = self._load_from_cache(self.train_cache_path)
            self.val_dataset = self._load_from_cache(self.val_cache_path)
            self.test_dataset = self._load_from_cache(self.test_cache_path)
        else:
            print("Cache not found. Walking directories and caching data (this will take a while for the first run)...")
            if self.balance_data:
                print("NOTE: Data balancing will be applied AFTER cache creation completes.")
                print("      You will see 'APPLYING DATA BALANCING' section when ready.\n")
            
            tile_paths = sorted(glob.glob(os.path.join(self.data_dir, "T*")))
            if not tile_paths:
                raise ValueError(f"No tiles found in {self.data_dir}")

            # Simple split strategy
            n_tiles = len(tile_paths)
            if n_tiles < 3:
                train_tiles = tile_paths
                val_tiles = tile_paths
                test_tiles = tile_paths
            else:
                test_tiles = [tile_paths[-1]]
                val_tiles = [tile_paths[-2]]
                train_tiles = tile_paths[:-2]
            
            print(f"Train Tiles : {len(train_tiles)}")
            print(f"Val Tiles   : {len(val_tiles)}")
            print(f"Test Tiles  : {len(test_tiles)}")

            self.train_dataset = self._create_and_cache_dataset(train_tiles, self.train_cache_path)
            self.val_dataset = self._create_and_cache_dataset(val_tiles, self.val_cache_path)
            self.test_dataset = self._create_and_cache_dataset(test_tiles, self.test_cache_path)
            
            print("\n✓ Cache creation completed!")
            if self.balance_data:
                print("  Now applying data balancing...")

        # Apply balancing if requested (only for train set usually, but user asked for general option)
        # We probably only want to balance the training set to avoid bias, but let's see.
        # User request: "label imbalance에 대한 ... 옵션"
        # Usually we only balance training data. Validation/Test should reflect real distribution.
        if self.balance_data:
            print("\n" + "="*60)
            print("APPLYING DATA BALANCING")
            print("="*60)
            self.train_dataset = self._balance_dataset(self.train_dataset)
            print("="*60 + "\n")

        # Calculate pos_weight
        # For cached dataset, we can get labels from targets
        # Assuming targets is a tensor
        if self.train_dataset.targets is not None:
             labels = self.train_dataset.targets
             n_pos = (labels == 1).sum().item()
             n_neg = (labels == 0).sum().item()
        else:
             labels = self.train_dataset.labels
             n_pos = sum(labels)
             n_neg = len(labels) - n_pos

        print(f"Final Training set: {n_pos} Real, {n_neg} Bogus. Total: {n_pos + n_neg}")
        if n_pos > 0:
            self.pos_weight = torch.tensor(n_neg / n_pos)
        else:
            self.pos_weight = torch.tensor(1.0)

    def _load_from_cache(self, cache_path):
        print(f"Loading {cache_path}...")
        data_dict = torch.load(cache_path)
        return RBSnapshotDataset(
            data=data_dict['data'],
            targets=data_dict['targets'],
            file_paths=data_dict.get('file_paths', None)
        )

    def _create_and_cache_dataset(self, tile_paths, cache_path):
        file_paths = []
        labels = []
        
        print("Collecting file paths...")
        for tile in tile_paths:
            sci_files = glob.glob(os.path.join(tile, "sci", "*.fits"))
            file_paths.extend(sci_files)
            labels.extend([1] * len(sci_files))
            
            sub_files = glob.glob(os.path.join(tile, "sub", "*.fits"))
            file_paths.extend(sub_files)
            labels.extend([0] * len(sub_files))
            
        print(f"Loading {len(file_paths)} images in parallel...")
        # Parallel load
        # Use joblib
        images = Parallel(n_jobs=self.num_workers * 2, backend="threading")(
            delayed(load_one_fits)(f) for f in tqdm(file_paths)
        )
        
        print("Stacking tensors...")
        # Stack into a single tensor
        data_tensor = torch.from_numpy(np.stack(images))
        targets_tensor = torch.tensor(labels, dtype=torch.float32)
        
        print(f"Saving cache to {cache_path}...")
        torch.save({
            'data': data_tensor, 
            'targets': targets_tensor,
            'file_paths': file_paths
        }, cache_path)
        
        return RBSnapshotDataset(data=data_tensor, targets=targets_tensor, file_paths=file_paths)

    def _balance_dataset(self, dataset):
        if dataset.targets is None:
            # Should not happen with cached logic, but fallback
            labels = np.array(dataset.labels)
            data = None # Cannot balance easily if not in memory
            print("Warning: Cannot balance dataset not loaded in memory.")
            return dataset
        
        targets = dataset.targets.numpy()
        
        # Indices
        real_indices = np.where(targets == 1)[0]
        bogus_indices = np.where(targets == 0)[0]
        
        n_real = len(real_indices)
        n_bogus = len(bogus_indices)
        
        print(f"  Original: {n_real} Real, {n_bogus} Bogus")

        if n_real > n_bogus:
            # Downsample Real to match Bogus count
            print(f"  Downsampling Real from {n_real} to {n_bogus} (using seed=42)")
            rng = np.random.RandomState(42)
            selected_real = rng.choice(real_indices, size=n_bogus, replace=False)
            selected_indices = np.concatenate([selected_real, bogus_indices])
        elif n_bogus > n_real:
             # Downsample Bogus (unlikely in this dataset)
            print(f"  Downsampling Bogus from {n_bogus} to {n_real} (using seed=42)")
            rng = np.random.RandomState(42)
            selected_bogus = rng.choice(bogus_indices, size=n_real, replace=False)
            selected_indices = np.concatenate([real_indices, selected_bogus])
        else:
            print(f"  Already balanced: {n_real} Real, {n_bogus} Bogus")
            return dataset
            
        # Sort indices to keep some order (optional but nice)
        selected_indices.sort()
        
        new_data = dataset.data[selected_indices]
        new_targets = dataset.targets[selected_indices]
        
        # Verify exact counts after balancing
        final_real = (new_targets == 1).sum().item()
        final_bogus = (new_targets == 0).sum().item()
        print(f"  Balanced result: {final_real} Real, {final_bogus} Bogus (Total: {len(new_targets)})")
        
        # Sanity check
        if final_real != final_bogus:
            print(f"  WARNING: Counts don't match! Real={final_real}, Bogus={final_bogus}")
        else:
            print(f"  ✓ Verification passed: Real count == Bogus count == {final_real}")
        
        return RBSnapshotDataset(data=new_data, targets=new_targets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
