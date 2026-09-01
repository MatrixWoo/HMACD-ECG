"""
PTB-XL+ feature dataset: Human Concept Bank U = [u_1, ..., u_M].

Merges 12SL, Uni-G, and ECGDeli features into one feature matrix,
aligned with PTB-XL records. Same train/val/test split as PTBXLDataset.

Usage:
    ds = PTBXLPlusDataset(data_path, split="train", fold=10)
    u = ds[0]  # Tensor [M] — human ECG features for sample 0
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PTBXLPlusDataset(Dataset):
    """
    PTB-XL+ Human Concept Bank dataset.

    Returns human-engineered ECG features aligned with PTB-XL records.
    Features come from three algorithms: 12SL, Uni-G, ECGDeli.

    Args:
        data_path:       path to ptb-xl/1.0.3/ (contains ptbxl_database.csv)
        plus_path:       path to ptb-xl-plus features directory
        split:           "train" | "val" | "test"
        fold:            test fold (default 10), val = fold-1
        nan_thresh:      drop features with > this fraction NaN (default 0.5)
        cache_path:      where to save preprocessed feature matrix

    Returns per sample:
        u: torch.Tensor [M]  — human concept feature vector (float32)
    """

    def __init__(
        self,
        data_path,
        plus_path=None,
        split="train",
        fold=10,
        nan_thresh=0.5,
        cache_path=None,
    ):
        self.data_path = data_path
        self.split = split
        self.fold = fold

        if plus_path is None:
            plus_path = os.path.join(
                os.path.dirname(data_path.rstrip("/")),
                "ptb-xl-plus/1.0.0/features",
            )
        self.plus_path = plus_path

        if cache_path is None:
            cache_path = os.path.join(data_path, "ptbxl_plus_features.npz")
        self.cache_path = cache_path

        # ---- load or build feature matrix ----
        U, ecg_ids, feature_names = self._load_or_build_features(
            data_path, plus_path, nan_thresh, cache_path
        )
        self.U = U  # np.ndarray [N_total, M]
        self.ecg_ids = ecg_ids  # np.ndarray [N_total]
        self.feature_names = feature_names  # list of str

        # ---- apply same stratified split as PTBXLDataset ----
        db = pd.read_csv(os.path.join(data_path, "ptbxl_database.csv"))
        val_fold = fold - 1

        # build index mask matching ecg_id order
        indices = []
        for row_idx, row in db.iterrows():
            if split == "train":
                if row["strat_fold"] != fold and row["strat_fold"] != val_fold:
                    indices.append(row_idx)
            elif split == "val":
                if row["strat_fold"] == val_fold:
                    indices.append(row_idx)
            elif split == "test":
                if row["strat_fold"] == fold:
                    indices.append(row_idx)

        # filter to only indices present in feature matrix
        valid_ids = set(self.ecg_ids)
        self.indices = [
            i for i in indices
            if db.iloc[i]["ecg_id"] in valid_ids
        ]

        # ---- precompute ecg_id → position mapping for fast lookup ----
        self._id_to_pos = {int(eid): p for p, eid in enumerate(self.ecg_ids)}
        self._db = db  # keep reference for __getitem__

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        ecg_id = self._db.iloc[row_idx]["ecg_id"]
        pos = self._id_to_pos[int(ecg_id)]
        u = torch.from_numpy(self.U[pos].astype(np.float32))
        return u

    @property
    def M(self):
        """Number of human concept features."""
        return self.U.shape[1]

    # ------------------------------------------------------------------
    #  Feature loading and preprocessing (internal)
    # ------------------------------------------------------------------

    def _load_or_build_features(self, data_path, plus_path, nan_thresh, cache_path):
        """Load cached features or build from scratch."""
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            return data["U"], data["ecg_ids"], list(data["feature_names"])

        print("Building PTB-XL+ feature matrix (one-time, cached)...")
        U, ecg_ids, feature_names = self._build_feature_matrix(
            data_path, plus_path, nan_thresh
        )
        np.savez(
            cache_path,
            U=U,
            ecg_ids=ecg_ids,
            feature_names=np.array(feature_names),
        )
        print(f"  Saved cached features to {cache_path}")
        return U, ecg_ids, feature_names

    def _build_feature_matrix(self, data_path, plus_path, nan_thresh):
        """Load and merge all three feature sets."""

        # ---- 1. load all three feature sets ----
        sl12 = pd.read_csv(os.path.join(plus_path, "12sl_features.csv"))
        unig = pd.read_csv(os.path.join(plus_path, "unig_features.csv"))
        ecgdeli = pd.read_csv(os.path.join(plus_path, "ecgdeli_features.csv"))

        # ---- 3. merge all on ecg_id ----
        U_df = sl12.merge(unig, on="ecg_id", how="inner", suffixes=("", "_unig"))
        U_df = U_df.merge(ecgdeli, on="ecg_id", how="inner", suffixes=("", "_ecgdeli"))

        print(f"  Merged feature matrix: {U_df.shape[0]} rows × {U_df.shape[1]} cols")

        # ---- 6. separate ecg_id, drop non-numeric columns ----
        ecg_ids = U_df["ecg_id"].values.astype(np.int64)
        U_df = U_df.drop(columns=["ecg_id"])

        # drop any remaining non-numeric columns
        non_num = U_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_num:
            print(f"  Dropping non-numeric columns: {non_num}")
            U_df = U_df.drop(columns=non_num)

        # ---- 7. drop features with too many NaN ----
        nan_frac = U_df.isna().mean()
        drop_cols = nan_frac[nan_frac > nan_thresh].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} features with >{nan_thresh:.0%} NaN")
            U_df = U_df.drop(columns=drop_cols)

        # ---- 8. impute remaining NaN with column median ----
        nan_after = U_df.isna().sum().sum()
        if nan_after > 0:
            print(f"  Imputing {nan_after} remaining NaN values with column median")
            U_df = U_df.fillna(U_df.median())

        # ---- 9. keep feature names ----
        feature_names = U_df.columns.tolist()

        # ---- 10. to numpy ----
        U = U_df.values.astype(np.float32)

        return U, ecg_ids, feature_names


if __name__ == "__main__":
    DATA = "/home/wuzuoxu/Data/ECG/1.0.3/"
    PLUS = "/home/wuzuoxu/Data/ECG/ptb-xl-plus/1.0.0/features/"

    print("=" * 60)
    print("Building PTB-XL+ Human Concept Bank (first run caches to .npz)")
    print("=" * 60)

    ds = PTBXLPlusDataset(DATA, plus_path=PLUS, split="train", fold=10)
    print(f"\nTrain samples: {len(ds)}")
    print(f"Feature dimension M: {ds.M}")
    u = ds[0]
    print(f"Sample feature vector: shape={u.shape}, dtype={u.dtype}")
    print(f"  min={u.min():.3f}, max={u.max():.3f}, mean={u.mean():.3f}")

    # val / test
    ds_v = PTBXLPlusDataset(DATA, plus_path=PLUS, split="val", fold=10)
    ds_t = PTBXLPlusDataset(DATA, plus_path=PLUS, split="test", fold=10)
    print(f"Val samples: {len(ds_v)}")
    print(f"Test samples: {len(ds_t)}")
    print(f"\nFeature names (first 20): {ds.feature_names[:20]}")
