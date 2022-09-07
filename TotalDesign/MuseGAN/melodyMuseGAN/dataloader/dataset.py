from torch.utils.data import Dataset
import pickle

class Dataset(Dataset):
  def __init__(self, chord_path, melody_path):
    with open(chord_path, "rb") as f:
      self.chord_data = pickle.load(f)
    with open(melody_path, "rb") as f:
      self.melody_data = pickle.load(f)
  
  def __len__(self):
    return len(self.chord_data)

  def __getitem__(self, idx):
    return self.chord_data[idx], self.melody_data[idx]
