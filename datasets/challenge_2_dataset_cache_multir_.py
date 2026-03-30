import json
import math
import os
import pickle
import random
from pathlib import Path

import lmdb
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset


CACHE_DIR = "./lmdb_cache_challenge2"
SFREQ     = 100

EXCLUDED_SUBJECTS = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH",
]


# ============================================================================
# Dataset Wrapper
# ============================================================================

class DatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int,
                 target_name: str = "externalizing", seed=None):
        self.dataset           = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name       = target_name
        self.rng               = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        target = float(self.dataset.description[self.target_name])

        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples
        offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        X = X[:, offset: offset + self.crop_size_samples]
        i_start += offset
        i_stop   = i_start + self.crop_size_samples

        infos = {
            "subject": self.dataset.description["subject"],
            "sex":     self.dataset.description["sex"],
            "age":     float(self.dataset.description["age"]),
            "task":    self.dataset.description["task"],
            "session": self.dataset.description.get("session") or "",
            "run":     self.dataset.description.get("run") or "",
        }
        return X, target, (i_window_in_trial, i_start, i_stop), infos


# ============================================================================
# LMDB
# ============================================================================

def _lmdb_path(r_path):
    path = Path(CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{Path(r_path).name.split('_')[0]}_externalizing_lmdb"


def _save_to_lmdb(windows_ds, lmdb_path):
    total    = sum(len(ds) for ds in windows_ds.datasets)
    map_size = max(total * 150 * 1024, 20 * 1024**3)
    env      = lmdb.open(str(lmdb_path), map_size=map_size, readonly=False,
                         lock=True, readahead=False, meminit=False)
    n, X_np = 0, None
    with env.begin(write=True) as txn:
        for ds in windows_ds.datasets:
            for i in range(len(ds)):
                try:
                    X, y, _, _ = ds[i]
                    X_np = np.array(X, dtype=np.float32)
                    txn.put(f'sample_{n}'.encode(),
                            pickle.dumps({'X': X_np, 'y': float(y)},
                                         protocol=pickle.HIGHEST_PROTOCOL))
                    n += 1
                except Exception as e:
                    print(f"Error saving sample {n}: {e}")
        txn.put(b'__metadata__', json.dumps({
            'n_samples':    n,
            'n_channels':   X_np.shape[0] if X_np is not None else 0,
            'n_timepoints': X_np.shape[1] if X_np is not None else 0,
        }).encode())
    env.close()
    print(f"Saved {n} windows to {lmdb_path}")


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = Path(lmdb_path)
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
        self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            meta = txn.get(b'__metadata__')
            self.metadata = json.loads(meta.decode()) if meta else {}
        self.n_samples = self.metadata.get('n_samples', 0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(f'sample_{idx}'.encode()))
        return torch.from_numpy(data['X']).float(), torch.tensor(data['y']).float()

    def close(self):
        if hasattr(self, 'env'): self.env.close()

    def __del__(self):
        self.close()


# ============================================================================
# Public API
# ============================================================================

def get_data_loader(r_path, batch_size=256, num_workers=12,
                    use_cache=True, force_reprocess=False):
    lmdb_path = _lmdb_path(r_path)

    if use_cache and lmdb_path.exists() and not force_reprocess:
        try:
            return DataLoader(LMDBDataset(lmdb_path), batch_size=batch_size,
                              shuffle=True, pin_memory=True, num_workers=num_workers)
        except Exception as e:
            print(f"Cache load failed ({e}), reprocessing...")

    root             = Path(r_path)
    bdf_files        = sorted(root.rglob("*contrastChangeDetection*_eeg.bdf"))
    assert bdf_files, f"No .bdf files found in {r_path}"

    participants_meta = (pd.read_csv(root / "participants.tsv", sep="\t")
                         .set_index("participant_id").to_dict(orient="index"))

    datasets = []
    for f in bdf_files:
        try:
            parts   = f.stem.split("_")
            subject = next(p for p in parts if p.startswith("sub-")).replace("sub-", "")
            task    = next(p for p in parts if p.startswith("task-")).replace("task-", "")
            run     = next((p.replace("run-", "") for p in parts if p.startswith("run-")), None)
            row     = participants_meta.get(f"sub-{subject}", {})
            ext     = row.get("externalizing")
            if ext is None or pd.isna(ext):
                continue
            raw = mne.io.read_raw_bdf(str(f), preload=True, verbose="ERROR")
            datasets.append(BaseDataset(raw=raw, description={
                "subject": subject, "run": run, "task": task,
                "age": row.get("age"), "sex": row.get("sex"),
                "gender": row.get("gender"), "externalizing": float(ext),
            }))
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    concat_ds = BaseConcatDataset([
        ds for ds in BaseConcatDataset(datasets).datasets
        if ds.description["subject"] not in EXCLUDED_SUBJECTS
        and ds.raw.n_times >= 4 * SFREQ
        and len(ds.raw.ch_names) == 129
        and not math.isnan(ds.description["externalizing"])
    ])

    windows_ds = create_fixed_length_windows(
        concat_ds,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )
    windows_ds = BaseConcatDataset(
        [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
    )

    if use_cache:
        _save_to_lmdb(windows_ds, lmdb_path)

    return DataLoader(LMDBDataset(lmdb_path), batch_size=batch_size,
                      shuffle=True, pin_memory=True, num_workers=num_workers)


def get_combined_dataloader(r_paths, batch_size=256, num_workers=12, use_cache=True):
    all_datasets = []
    for r_path in r_paths:
        lmdb_path = _lmdb_path(r_path)
        if use_cache and lmdb_path.exists():
            try:
                all_datasets.append(LMDBDataset(lmdb_path))
                continue
            except Exception as e:
                print(f"Cache load failed: {e}")
        get_data_loader(r_path, batch_size=batch_size, num_workers=num_workers, use_cache=True)
        all_datasets.append(LMDBDataset(lmdb_path))

    loader = DataLoader(ConcatDataset(all_datasets), batch_size=batch_size, shuffle=True,
                        pin_memory=True, num_workers=num_workers,
                        persistent_workers=num_workers > 0)
    return loader, all_datasets


def close_datasets(datasets):
    for ds in datasets:
        if hasattr(ds, 'close'): ds.close()


def clear_cache(specific_r=None):
    import shutil
    cache = Path(CACHE_DIR)
    if not cache.exists(): return
    targets = ([d for d in cache.iterdir() if d.is_dir() and d.name.startswith(specific_r)]
               if specific_r else [cache])
    for t in targets:
        shutil.rmtree(t); print(f"Removed: {t}")


def list_caches():
    cache = Path(CACHE_DIR)
    if not cache.exists(): return
    print(f"\n{'Directory':<30} {'MB':>10} {'Samples':>10}")
    print("-" * 55)
    for d in sorted(d for d in cache.iterdir() if d.is_dir()):
        size_mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1024**2
        try:
            env = lmdb.open(str(d), readonly=True, lock=False)
            with env.begin() as txn:
                meta = txn.get(b'__metadata__')
                n = json.loads(meta.decode()).get('n_samples', 0) if meta else 0
            env.close()
            print(f"{d.name:<30} {size_mb:>10.1f} {n:>10,}")
        except Exception:
            print(f"{d.name:<30} {size_mb:>10.1f} {'ERROR':>10}")


if __name__ == "__main__":
    list_caches()
