import numpy as np
import torch
import pandas as pd
from math import pi
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


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(b * x + c) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.
    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.
    periodicity_range: tuple of float
        Defines the range from which the periodicity (i.e. b) of the sine function
        is sampled.
    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.
    num_samples : int
        Number of samples of the function contained in dataset.
    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """

    def __init__(self, amplitude_range=(0.4, 1.6), periodicity_range=(1., pi + (pi - 1.0) * 0.5),
                 shift_range=(0, pi), num_samples=3000, num_points=100):
        self.amplitude_range = amplitude_range
        self.periodicity_range = periodicity_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = periodicity_range
        c_min, c_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min

            # For VERSION 3
            # Sample random periodicity
            b = (b_max - b_min) * np.random.rand() + b_min
            # Sample random shift
            c = (c_max - c_min) * np.random.rand() + c_min
            # Shape (num_points, x_dim)
            x = torch.linspace(0, 9.9, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)

            # For VERSION 1
            # y = a * torch.sin(x + c)

            # For VERSION 2
            # y = a * torch.sin(x * pi + c)

            # For VERSION3
            y = a * torch.sin(x * b + c)

            # For noisy version
            noise = torch.randn(y.shape) * 0.01
            y += noise

            traj_idx = torch.ones_like(x) * i

            self.data.append((traj_idx, x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples