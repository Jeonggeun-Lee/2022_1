import torch

common_config = {
  'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'batch_size': 16,
  'num_tracks': 4,
  'num_measures': 8,
  'num_steps_per_measure': 16,
  'num_steps_per_note': 4,
  'num_steps': 128,
  'num_pitch_bins': 128,
}