import math
import os
import random
from pathlib import Path

import mne
import pandas as pd
import torch
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.models import EEGNeX
from braindecode.preprocessing import create_fixed_length_windows
from torch import optim
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% Constants
DATA_DIR  = Path("/homepatrick/llm_project/data/HBN/finetune/BIDS/R5_100_bdf_bids")
SAVE_PATH = Path("/homepatrick/llm_project/baseline_ft_ckpt/weights_challenge_2.pt")
SFREQ     = 100

EXCLUDED_SUBS = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
                 "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
SEX_MAP = {"M": 0, "F": 1}

# %% Load data
bdf_files        = sorted(DATA_DIR.rglob("*contrastChangeDetection*_eeg.bdf"))
assert bdf_files, "No .bdf files found!"

participants_meta = (pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")
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
            "age": row.get("age"), "sex": SEX_MAP.get(row.get("sex")),
            "externalizing": float(ext),
        }))
    except Exception as e:
        print(f"Error loading {f.name}: {e}")

all_datasets = BaseConcatDataset([
    ds for ds in BaseConcatDataset(datasets).datasets
    if ds.description["subject"] not in EXCLUDED_SUBS
    and ds.raw.n_times >= 4 * SFREQ
    and len(ds.raw.ch_names) == 129
    and not math.isnan(ds.description["externalizing"])
])
print(f"Total recordings after filtering: {len(all_datasets.datasets)}")

# %% Dataset wrapper
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
        X      = X[:, offset: offset + self.crop_size_samples]
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

# %% Windows
windows_ds = create_fixed_length_windows(
    all_datasets,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)
windows_ds = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
)

# %% Model
model     = EEGNeX(n_chans=129, n_outputs=1, n_times=2 * SFREQ).to(device)
optimizer = optim.Adamax(model.parameters(), lr=0.002)
print(model)

# %% Train
loader   = DataLoader(windows_ds, batch_size=128, shuffle=True, num_workers=1)
n_epochs = 10

for epoch in range(n_epochs):
    for step, (X, y, _, _) in enumerate(loader):
        optimizer.zero_grad()
        X    = X.to(torch.float32).to(device)
        y    = y.to(torch.float32).to(device).unsqueeze(1)
        loss = l1_loss(model(X), y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} step {step}: loss={loss.item():.4f}")

# %% Save
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved: {SAVE_PATH}")
