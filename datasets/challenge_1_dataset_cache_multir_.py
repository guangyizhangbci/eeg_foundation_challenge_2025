from pathlib import Path
import json
import os
import pickle

import lmdb
import mne
import numpy as np
import pandas as pd
import torch
from mne_bids import get_bids_path_from_fname
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events


CACHE_DIR  = "./lmdb_cache_challenge1"
EPOCH_LEN  = 2.0
SFREQ      = 100
SHIFT      = 0.5
ANCHOR     = "stimulus_anchor"


# ============================================================================
# Trial Table Helpers
# ============================================================================

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
        start, end = float(tr["onset"]), float(tr["next_onset"])
        stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])

        anchor = stim_onset if not np.isnan(stim_onset) else start
        resp_block = responses[(responses["onset"] >= anchor) & (responses["onset"] < end)]

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
    bids_path  = get_bids_path_from_fname(raw.filenames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath
    events_df   = pd.read_csv(events_file, sep="\t")
    trials      = build_trial_table(events_df)

    if require_stimulus: trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response:  trials = trials[trials["response_onset"].notna()].copy()
    if target_field not in trials.columns:
        raise KeyError(f"{target_field} not in computed trial table.")

    extras = []
    for i, row in trials.iterrows():
        extras.append({
            "target":             _to_float_or_none(row[target_field]),
            "rt_from_stimulus":   _to_float_or_none(row["rt_from_stimulus"]),
            "rt_from_trialstart": _to_float_or_none(row["rt_from_trialstart"]),
            "stimulus_onset":     _to_float_or_none(row["stimulus_onset"]),
            "response_onset":     _to_float_or_none(row["response_onset"]),
            "correct":            _to_int_or_none(row["correct"]),
            "response_type":      _to_str_or_none(row["response_type"]),
        })

    raw.set_annotations(mne.Annotations(
        onset=trials["trial_start_onset"].to_numpy(float),
        duration=np.full(len(trials), float(epoch_length)),
        description=["contrast_trial_start"] * len(trials),
        orig_time=raw.info["meas_date"],
        extras=extras,
    ), verbose=False)
    return raw


def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    ann  = raw.annotations
    mask = ann.description == "contrast_trial_start"
    if not np.any(mask):
        return raw

    stim_onsets, resp_onsets   = [], []
    stim_extras, resp_extras   = [], []

    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        t0 = float(ann.onset[idx])
        stim_t = ex.get("stimulus_onset")
        resp_t = ex.get("response_onset")

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt, rts = ex.get("rt_from_trialstart"), ex.get("rt_from_stimulus")
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex.get("rt_from_trialstart")
            if rtt is not None:
                resp_t = t0 + float(rtt)

        def _valid(v): return v is not None and not (isinstance(v, float) and np.isnan(v))
        if _valid(stim_t):
            stim_onsets.append(float(stim_t)); stim_extras.append(dict(ex, anchor="stimulus"))
        if _valid(resp_t):
            resp_onsets.append(float(resp_t)); resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        raw.set_annotations(ann + mne.Annotations(
            onset=new_onsets,
            duration=np.zeros_like(new_onsets),
            description=[stim_desc]*len(stim_onsets) + [resp_desc]*len(resp_onsets),
            orig_time=raw.info["meas_date"],
            extras=stim_extras + resp_extras,
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
                md[k] = pd.Series([None if v is None else int(bool(v)) for v in vals],
                                   index=md.index, dtype="Int64")
            elif k in float_cols:
                md[k] = pd.Series([np.nan if v is None else float(v) for v in vals],
                                   index=md.index, dtype="Float64")
            else:
                md[k] = pd.Series(vals, index=md.index, dtype="string")

        win_ds.metadata = md.reset_index(drop=True)

    return windows_ds


def keep_only_recordings_with(desc, concat_ds):
    return BaseConcatDataset([
        ds for ds in concat_ds.datasets
        if np.any(ds.raw.annotations.description == desc)
    ])


# ============================================================================
# LMDB
# ============================================================================

def _lmdb_path(r_path):
    path = Path(CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{Path(r_path).name.split('_')[0]}_ccd_lmdb"


def _save_to_lmdb(single_windows, lmdb_path):
    total   = sum(len(ds) for ds in single_windows.datasets)
    map_size = max(total * 150 * 1024, 20 * 1024**3)
    env     = lmdb.open(str(lmdb_path), map_size=map_size, readonly=False,
                        lock=True, readahead=False, meminit=False)
    n, X_np = 0, None
    with env.begin(write=True) as txn:
        for ds in single_windows.datasets:
            for i in range(len(ds)):
                try:
                    X, y, _ = ds[i]
                    X_np    = np.array(X, dtype=np.float32)
                    y_val   = float(y) if not isinstance(y, (list, tuple, np.ndarray)) else float(y[0])
                    txn.put(f'sample_{n}'.encode(), pickle.dumps({'X': X_np, 'y': y_val},
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
            self.metadata  = json.loads(meta.decode()) if meta else {}
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

def get_data_loader(r_path, batch_size=256, num_workers=8, use_cache=True, force_reprocess=False):
    lmdb_path = _lmdb_path(r_path)

    if use_cache and lmdb_path.exists() and not force_reprocess:
        try:
            return DataLoader(LMDBDataset(lmdb_path), batch_size=batch_size,
                              shuffle=True, pin_memory=True, num_workers=num_workers)
        except Exception as e:
            print(f"Cache load failed ({e}), reprocessing...")

    root      = Path(r_path)
    bdf_files = sorted(root.rglob("*contrastChangeDetection*_eeg.bdf"))
    assert bdf_files, f"No .bdf files found in {r_path}"

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

    concat_ds = BaseConcatDataset(datasets)
    preprocess(concat_ds, [
        Preprocessor(annotate_trials_with_target, target_field="rt_from_stimulus",
                     epoch_length=EPOCH_LEN, require_stimulus=True,
                     require_response=True, apply_on_array=False),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ], n_jobs=1)

    dataset = keep_only_recordings_with(ANCHOR, concat_ds)
    windows = create_windows_from_events(
        dataset, mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT * SFREQ),
        trial_stop_offset_samples=int((SHIFT + EPOCH_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN * SFREQ),
        window_stride_samples=SFREQ, preload=True,
    )
    windows = add_extras_columns(windows, dataset, desc=ANCHOR)

    if use_cache:
        _save_to_lmdb(windows, lmdb_path)

    return DataLoader(LMDBDataset(lmdb_path), batch_size=batch_size,
                      shuffle=True, pin_memory=True, num_workers=num_workers)


def get_combined_dataloader(r_paths, batch_size=256, num_workers=8, use_cache=True):
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
        shutil.rmtree(t)
        print(f"Removed: {t}")


def list_caches():
    cache = Path(CACHE_DIR)
    if not cache.exists(): return
    lmdb_dirs = sorted(d for d in cache.iterdir() if d.is_dir())
    print(f"\n{'Directory':<30} {'MB':>10} {'Samples':>10}")
    print("-" * 55)
    for d in lmdb_dirs:
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
