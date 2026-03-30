import math
import os
import random
from pathlib import Path

import mne
import pandas as pd
import torch
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.preprocessing import create_fixed_length_windows
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

DATA_DIR      = Path("/home/patrick5/llm_project/data/HBN/finetune/BIDS/R5_100_bdf_bids")
SFREQ         = 100
SEED          = 2025
EXCLUDED_SUBS = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
                 "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]


class DatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int, seed=None):
        self.dataset           = dataset
        self.crop_size_samples = crop_size_samples
        self.rng               = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        p_factor = float(self.dataset.description["p_factor"])

        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples
        offset  = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start += offset
        i_stop   = i_start + self.crop_size_samples
        X        = X[:, offset: offset + self.crop_size_samples]

        return X, p_factor

    @property
    def description(self):
        return self.dataset.description


def get_data_loader():
    bdf_files        = sorted(DATA_DIR.rglob("*contrastChangeDetection*_eeg.bdf"))
    assert bdf_files, "No .bdf files found!"

    participants_meta = (pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")
                         .set_index("participant_id").to_dict(orient="index"))

    datasets = []
    for f in bdf_files:
        try:
            parts    = f.stem.split("_")
            subject  = next(p for p in parts if p.startswith("sub-")).replace("sub-", "")
            task     = next(p for p in parts if p.startswith("task-")).replace("task-", "")
            run      = next((p.replace("run-", "") for p in parts if p.startswith("run-")), None)
            row      = participants_meta.get(f"sub-{subject}", {})
            p_factor = row.get("p_factor")
            if p_factor is None or pd.isna(p_factor):
                continue
            raw = mne.io.read_raw_bdf(str(f), preload=True, verbose="ERROR")
            datasets.append(BaseDataset(raw=raw, description={
                "subject": subject, "run": run, "task": task,
                "age": row.get("age"), "sex": row.get("sex"),
                "gender": row.get("gender"), "p_factor": float(p_factor),
            }))
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    all_datasets = BaseConcatDataset([
        ds for ds in BaseConcatDataset(datasets).datasets
        if ds.description["subject"] not in EXCLUDED_SUBS
        and ds.raw.n_times >= 4 * SFREQ
        and len(ds.raw.ch_names) == 129
        and not math.isnan(ds.description["p_factor"])
    ])
    print(f"Total recordings after filtering: {len(all_datasets.datasets)}")

    windows_ds = create_fixed_length_windows(
        all_datasets,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )
    windows_ds = BaseConcatDataset(
        [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
    )

    subjects    = windows_ds.description["subject"].unique()
    train_subj, val_test = train_test_split(subjects, test_size=0.2,
                                            random_state=check_random_state(SEED), shuffle=True)
    val_subj, test_subj  = train_test_split(val_test, test_size=0.5,
                                            random_state=check_random_state(SEED + 1), shuffle=True)
    assert (set(train_subj) | set(val_subj) | set(test_subj)) == set(subjects)

    splits    = windows_ds.split("subject")
    train_set = BaseConcatDataset([splits[s] for s in train_subj if s in splits])
    val_set   = BaseConcatDataset([splits[s] for s in val_subj   if s in splits])
    test_set  = BaseConcatDataset([splits[s] for s in test_subj  if s in splits])

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    labels    = [ds[i][1] for ds in train_set.datasets for i in range(len(ds))]
    all_y     = torch.tensor(labels)
    y_mean, y_std = all_y.mean(), all_y.std()

    loader_kwargs = dict(batch_size=64, pin_memory=True, num_workers=12)
    return {
        "train": DataLoader(train_set, shuffle=True,  **loader_kwargs),
        "val":   DataLoader(val_set,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_set,  shuffle=False, **loader_kwargs),
        "norm":  (y_mean, y_std),
    }


if __name__ == '__main__':
    get_data_loader()
