import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class sinusoid_dataset(Dataset):
    def __init__(self, data_path, idx):
        self.all_data = pd.read_csv(data_path)

        self.all_data = self.all_data.loc[self.all_data['sample_idx'].isin(idx)].copy()
        unique_idx = self.all_data['sample_idx'].unique()
        map_dict = dict(zip(unique_idx, np.arange(len(unique_idx))))

        self.all_data['sample_idx'] = self.all_data['sample_idx'].map(map_dict)
        self.all_data.set_index('sample_idx', inplace=True)

        self.length = len(unique_idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        traj = self.all_data.loc[idx]
        return {"idx": idx, "traj": traj}


def sinusoid_collate(batch):
    t = torch.tensor(batch[0]["traj"]["time"].values)
    traj_columns = [False, True]
    trajs = [torch.from_numpy(b["traj"].iloc[:, traj_columns].values) for b in batch]
    trajs = torch.stack(trajs, dim=0)

    res = dict()
    res["time"] = t
    res["traj"] = trajs
    return res