import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SUPERCLASS_LIST = ["NORM", "MI", "STTC", "CD", "HYP"]


class PTBXLDataset(Dataset):
    def __init__(self, data_path, split="train", fold=10):
        """
        Args:
            data_path: path to ptb-xl/1.0.3/
            split: "train" | "val" | "test"
            fold: test fold (default 10), val fold is fold-1
                  PTB-XL strat_fold is 1-10.
        """
        self.data_path = data_path
        self.split = split

        # 1. 读 SCP statements → 建 code → diagnostic_class (superclass) 映射
        stmt_df = pd.read_csv(
            os.path.join(data_path, "scp_statements.csv"), index_col=0
        )
        # diagnostic_class 就是 superclass: NORM / MI / STTC / CD / HYP
        self.scp_to_superclass = {}
        for code, row in stmt_df.iterrows():
            dc = row["diagnostic_class"]
            if isinstance(dc, str) and dc in SUPERCLASS_LIST:
                self.scp_to_superclass[code] = dc

        # 2. 读主 CSV
        df = pd.read_csv(os.path.join(data_path, "ptbxl_database.csv"))

        # 3. 按 strat_fold 划分 (fold 是 1-10)
        val_fold = fold - 1  # e.g. fold=10 → val on 9

        if split == "train":
            df = df[(df["strat_fold"] != fold) & (df["strat_fold"] != val_fold)]
        elif split == "val":
            df = df[df["strat_fold"] == val_fold]
        elif split == "test":
            df = df[df["strat_fold"] == fold]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.df = df.reset_index(drop=True)

        # filter out records with missing .dat files
        self.df = self.df[
            self.df.apply(
                lambda row: os.path.isfile(
                    os.path.join(data_path, row["filename_lr"] + ".dat")
                ),
                axis=1,
            )
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. 读 ECG 信号 (.dat 是 time-major 交错存储 → reshape(-1,12).T)
        file_path = os.path.join(self.data_path, row["filename_lr"] + ".dat")
        raw = np.fromfile(file_path, dtype=np.int16)
        x = torch.from_numpy(raw.reshape(-1, 12).T.astype(np.float32))

        # 2. 解析 label: scp_codes → SCP codes → diagnostic_class (superclass)
        scp_dict = ast.literal_eval(row["scp_codes"])  # e.g. {"IMI": 70.0, "STTC": 30.0}
        superclass_set = set()
        for code in scp_dict:
            if code in self.scp_to_superclass:
                superclass_set.add(self.scp_to_superclass[code])

        y = torch.zeros(len(SUPERCLASS_LIST), dtype=torch.float32)
        for cls_name in superclass_set:
            y[SUPERCLASS_LIST.index(cls_name)] = 1.0

        return x, y


if __name__ == "__main__":
    ds = PTBXLDataset("/home/wuzuoxu/Data/ECG/1.0.3/", split="train")
    print(f"size of dataset: {len(ds)}")
    x, y = ds[0]
    print(f"x shape: {x.shape}")   # torch.Size([12, 1000])
    print(f"x dtype: {x.dtype}")   # torch.float32
    print(f"y shape: {y.shape}")   # torch.Size([5])
    print(f"y: {y}")
