# %% Imports
import copy
import random
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
import torch
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.models import EEGNeX
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from mne_bids import get_bids_path_from_fname
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% Constants
DATA_DIR       = Path("/home/patrick/llm_project/data/HBN/finetune/BIDS/R5_100_bdf_bids")
SAVE_PATH      = Path("/home/patrick/llm_project/baseline_ft_ckpt/weights_challenge_1.pt")
EPOCH_LEN      = 2.0
SFREQ          = 100
SHIFT          = 0.5
ANCHOR         = "stimulus_anchor"
EXCLUDED_SUBS  = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
                   "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
SEED           = 2025

# %% Load raw data
bdf_files = sorted(DATA_DIR.rglob("*contrastChangeDetection*_eeg.bdf"))
assert bdf_files, "No .bdf files found!"

datasets = []
for f in bdf_files:
    try:
        parts   = f.stem.split("_")
        subject = next(p for p in parts if p.startswith("sub-")).replace("sub-", "")
        task    = next(p for p in parts if p.startswith("task-")).replace("task-", "")
        run     = next((p.replace("run-", "") for p in parts if p.startswith("run-")), None)
        raw     = mne.io.read_raw_bdf(str(f), preload=True, verbose="ERROR")
        datasets.append(BaseDataset(raw=raw, description={
            "subject": subject, "task": task, "run": int(run) if run else None}))
    except Exception as e:
        print(f"Error loading {f.name}: {e}")

dataset_ccd = BaseConcatDataset(datasets)
print(f"Total recordings loaded: {len(dataset_ccd.datasets)}")

# %% Trial table helpers
def _to_float_or_none(x):
    return None if pd.isna(x) else float(x)

def _to_int_or_none(x):
    if pd.isna(x): return None
    if isinstance(x, (bool, np.bool_)): return int(bool(x))
    if isinstance(x, (int, np.integer)): return int(x)
    try: return int(x)
    except Exception: return None

def _to_str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def build_trial_table(events_df: pd.DataFrame) -> pd.DataFrame:
    events_df = (events_df.copy()
                 .assign(onset=lambda d: pd.to_numeric(d["onset"], errors="raise"))
                 .sort_values("onset", kind="mergesort").reset_index(drop=True))

    trials    = events_df[events_df["value"].eq("contrastTrial_start")].copy().reset_index(drop=True)
    stimuli   = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

    trials["next_onset"] = trials["onset"].shift(-1)
    trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

    rows = []
    for _, tr in trials.iterrows():
        start, end  = float(tr["onset"]), float(tr["next_onset"])
        stim_block  = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset  = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])
        anchor      = stim_onset if not np.isnan(stim_onset) else start
        resp_block  = responses[(responses["onset"] >= anchor) & (responses["onset"] < end)]

        if resp_block.empty:
            resp_onset, resp_type, feedback = np.nan, None, None
        else:
            r = resp_block.iloc[0]
            resp_onset, resp_type, feedback = float(r["onset"]), r["value"], r["feedback"]

        correct = None
        if isinstance(feedback, str):
            correct = feedback == "smiley_face"

        rows.append({
            "trial_start_onset":  start,
            "trial_stop_onset":   end,
            "stimulus_onset":     stim_onset,
            "response_onset":     resp_onset,
            "rt_from_stimulus":   (resp_onset - stim_onset) if not (np.isnan(stim_onset) or np.isnan(resp_onset)) else np.nan,
            "rt_from_trialstart": (resp_onset - start)      if not np.isnan(resp_onset) else np.nan,
            "response_type":      resp_type,
            "correct":            correct,
        })
    return pd.DataFrame(rows)


def annotate_trials_with_target(raw, target_field="rt_from_stimulus", epoch_length=2.0,
                                require_stimulus=True, require_response=True):
    bids_path   = get_bids_path_from_fname(raw.filenames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath
    trials      = build_trial_table(pd.read_csv(events_file, sep="\t"))

    if require_stimulus: trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response:  trials = trials[trials["response_onset"].notna()].copy()
    if target_field not in trials.columns:
        raise KeyError(f"{target_field} not in trial table.")

    extras = [{
        "target":             _to_float_or_none(row[target_field]),
        "rt_from_stimulus":   _to_float_or_none(row["rt_from_stimulus"]),
        "rt_from_trialstart": _to_float_or_none(row["rt_from_trialstart"]),
        "stimulus_onset":     _to_float_or_none(row["stimulus_onset"]),
        "response_onset":     _to_float_or_none(row["response_onset"]),
        "correct":            _to_int_or_none(row["correct"]),
        "response_type":      _to_str_or_none(row["response_type"]),
    } for _, row in trials.iterrows()]

    raw.set_annotations(mne.Annotations(
        onset=trials["trial_start_onset"].to_numpy(float),
        duration=np.full(len(trials), float(epoch_length)),
        description=["contrast_trial_start"] * len(trials),
        orig_time=raw.info["meas_date"], extras=extras,
    ), verbose=False)
    return raw


def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    ann  = raw.annotations
    mask = ann.description == "contrast_trial_start"
    if not np.any(mask): return raw

    stim_onsets, resp_onsets = [], []
    stim_extras, resp_extras = [], []

    def _valid(v): return v is not None and not (isinstance(v, float) and np.isnan(v))

    for idx in np.where(mask)[0]:
        ex     = ann.extras[idx] if ann.extras is not None else {}
        t0     = float(ann.onset[idx])
        stim_t = ex.get("stimulus_onset")
        resp_t = ex.get("response_onset")

        if not _valid(stim_t):
            rtt, rts = ex.get("rt_from_trialstart"), ex.get("rt_from_stimulus")
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)
        if not _valid(resp_t):
            rtt = ex.get("rt_from_trialstart")
            if rtt is not None: resp_t = t0 + float(rtt)

        if _valid(stim_t): stim_onsets.append(float(stim_t)); stim_extras.append(dict(ex, anchor="stimulus"))
        if _valid(resp_t):  resp_onsets.append(float(resp_t)); resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        raw.set_annotations(ann + mne.Annotations(
            onset=new_onsets, duration=np.zeros_like(new_onsets),
            description=[stim_desc]*len(stim_onsets) + [resp_desc]*len(resp_onsets),
            orig_time=raw.info["meas_date"], extras=stim_extras + resp_extras,
        ), verbose=False)
    return raw


def add_extras_columns(windows_ds, original_ds, desc="contrast_trial_start",
                       keys=("target","rt_from_stimulus","rt_from_trialstart",
                             "stimulus_onset","response_onset","correct","response_type")):
    float_cols = {"target","rt_from_stimulus","rt_from_trialstart","stimulus_onset","response_onset"}
    for win_ds, base_ds in zip(windows_ds.datasets, original_ds.datasets):
        ann = base_ds.raw.annotations
        idx = np.where(ann.description == desc)[0]
        if not idx.size: continue
        per_trial = [{k: (ann.extras[i].get(k) if ann.extras else None) for k in keys} for i in idx]
        md        = win_ds.metadata.copy()
        trial_ids = (md["i_window_in_trial"].to_numpy() == 0).cumsum() - 1
        for k in keys:
            vals = [per_trial[t][k] if t < len(per_trial) else None for t in trial_ids]
            if k == "correct":
                md[k] = pd.Series([None if v is None else int(bool(v)) for v in vals], index=md.index, dtype="Int64")
            elif k in float_cols:
                md[k] = pd.Series([np.nan if v is None else float(v) for v in vals], index=md.index, dtype="Float64")
            else:
                md[k] = pd.Series(vals, index=md.index, dtype="string")
        win_ds.metadata = md.reset_index(drop=True)
    return windows_ds


def keep_only_recordings_with(desc, concat_ds):
    return BaseConcatDataset([
        ds for ds in concat_ds.datasets
        if np.any(ds.raw.annotations.description == desc)
    ])

# %% Preprocess and window
preprocess(dataset_ccd, [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor(lambda x: x / 100),
    Preprocessor(annotate_trials_with_target, target_field="rt_from_stimulus",
                 epoch_length=EPOCH_LEN, require_stimulus=True,
                 require_response=True, apply_on_array=False),
    Preprocessor(add_aux_anchors, apply_on_array=False),
], n_jobs=1)

dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
single_windows = create_windows_from_events(
    dataset, mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT * SFREQ),
    trial_stop_offset_samples=int((SHIFT + EPOCH_LEN) * SFREQ),
    window_size_samples=int(EPOCH_LEN * SFREQ),
    window_stride_samples=SFREQ, preload=True,
)
single_windows = add_extras_columns(single_windows, dataset, desc=ANCHOR)

# %% Train/val/test split
meta     = single_windows.get_metadata()
subjects = [s for s in meta["subject"].unique() if s not in EXCLUDED_SUBS]

train_subj, val_test = train_test_split(subjects, test_size=0.2,
                                        random_state=check_random_state(SEED), shuffle=True)
val_subj, test_subj  = train_test_split(val_test, test_size=0.5,
                                        random_state=check_random_state(SEED + 1), shuffle=True)

splits     = single_windows.split("subject")
train_set  = BaseConcatDataset([splits[s] for s in train_subj if s in splits])
valid_set  = BaseConcatDataset([splits[s] for s in val_subj  if s in splits])
test_set   = BaseConcatDataset([splits[s] for s in test_subj if s in splits])

print(f"Train: {len(train_set)}  Val: {len(valid_set)}  Test: {len(test_set)}")

# %% Dataloaders
train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=1)
valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, num_workers=1)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=1)

# %% Model
model = EEGNeX(n_chans=129, n_outputs=1, n_times=200, sfreq=100).to(device)
print(model)

# %% Training functions
def train_one_epoch(loader, model, loss_fn, optimizer, scheduler, epoch):
    model.train()
    total_loss, sq_err, n = 0.0, 0.0, 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        sq_err     += torch.sum((pred.detach().view(-1) - y.view(-1)) ** 2).item()
        n          += y.numel()
    if scheduler: scheduler.step()
    return total_loss / len(loader), (sq_err / max(n, 1)) ** 0.5


@torch.no_grad()
def evaluate(loader, model, loss_fn):
    model.eval()
    total_loss, sq_err, n = 0.0, 0.0, 0
    all_y = []
    for batch in loader:
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        pred = model(X)
        total_loss += loss_fn(pred, y).item()
        sq_err     += torch.sum((pred.view(-1) - y.view(-1)) ** 2).item()
        n          += y.numel()
        all_y.append(y.view(-1))
    rmse = (sq_err / max(n, 1)) ** 0.5
    norm = rmse / torch.std(torch.cat(all_y))
    print(f"RMSE: {rmse:.6f}  Normalised RMSE: {norm:.6f}  Loss: {total_loss/len(loader):.6f}")
    return total_loss / len(loader), rmse

# %% Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=99)
loss_fn   = torch.nn.MSELoss()

best_rmse, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
PATIENCE, MIN_DELTA, N_EPOCHS = 5, 1e-4, 100

for epoch in range(1, N_EPOCHS + 1):
    train_loss, train_rmse = train_one_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch)
    print(f"Epoch {epoch} — train loss: {train_loss:.6f}  train RMSE: {train_rmse:.6f}")
    val_loss, val_rmse = evaluate(valid_loader, model, loss_fn)

    if val_rmse < best_rmse - MIN_DELTA:
        best_rmse, best_state, best_epoch, no_improve = val_rmse, copy.deepcopy(model.state_dict()), epoch, 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
            break

if best_state: model.load_state_dict(best_state)

# %% Save
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved: {SAVE_PATH}")
