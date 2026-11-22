import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryGraphCache:
    """Read-only in-memory cache of preprocessed MD trajectory graphs."""

    def __init__(self, cache_path: str):
        with np.load(cache_path) as d:
            self.atom_types = torch.from_numpy(d["atom_types"].astype(np.int64))
            self.target_index = torch.from_numpy(d["target_index"].astype(np.int64))
            self.nbr_lists = torch.from_numpy(d["nbr_lists"].astype(np.int64))
            self.nbr_dists = torch.from_numpy(d["nbr_dists"].astype(np.float32))
            self.nbr_vecs = (
                torch.from_numpy(d["nbr_vecs"].astype(np.float32))
                if "nbr_vecs" in d.files
                else None
            )
            self.traj_coords = (
                torch.from_numpy(d["traj_coords"].astype(np.float32))
                if "traj_coords" in d.files
                else None
            )
        self.num_frames, self.num_atoms, self.num_nbrs = self.nbr_lists.shape
        self.num_targets = self.target_index.shape[0]


class LaggedFramePairDataset(Dataset):
    """Chronological frame-pair dataset (t, t + tau) within a fixed frame slice [start, end)."""

    def __init__(self, cache: TrajectoryGraphCache, start_frame: int, end_frame: int, tau: int):
        assert 0 <= start_frame < end_frame <= cache.num_frames
        assert end_frame - start_frame > tau, "Slice shorter than lag time tau"
        self.cache = cache
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.tau = tau
        self.length = (end_frame - start_frame) - tau

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        t0 = self.start_frame + idx
        t1 = t0 + self.tau
        return t0, t1
