import pickle
import lmdb
import torch
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    def __init__(self, dataset_dir, patch_size=100):
        super().__init__()
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with self.db.begin(write=False) as txn:
            signal = pickle.loads(txn.get(self.keys[idx].encode()))
        c, t = signal.shape
        assert t % self.patch_size == 0, f"Time length {t} not divisible by patch size {self.patch_size}"
        return torch.tensor(signal, dtype=torch.float32).view(c, t // self.patch_size, self.patch_size)
