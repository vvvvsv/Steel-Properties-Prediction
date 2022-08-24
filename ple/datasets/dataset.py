import numpy as np
import pandas as pd
import torch

class SbgcDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:]

        self.categorical_data = data[:, 3:5].astype(np.int)
        self.numerical_data = data[:, 5:].astype(np.float32)
        self.labels = data[:, :3].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]