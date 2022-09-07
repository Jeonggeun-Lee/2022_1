from torch.utils.data import Dataset
import torch
from config import config

class DS(Dataset):
    def __init__(self, x_path, y_path):
        self.x = torch.load(x_path, map_location=config['device'])
        self.y = torch.load(y_path, map_location=config['device'])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
