import numpy as np
import pandas as pd
import torch

class SteelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, task):
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:]

        self.xs = data[:, 3:].astype(np.float32)
        self.ys = data[:, task:task+1].astype(np.float32)
        self.x_num = self.xs.shape[1]

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]