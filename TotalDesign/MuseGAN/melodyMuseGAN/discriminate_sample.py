from MuseGAN.melodyMuseGAN.model.discriminator import Discriminator
import pickle
import numpy as np
import torch

real_melody_path = './data/melody_train_data.pkl'
real_chord_path = './data/chord_train_data.pkl'
fake_score_path = './result/generated_score.npy'
disc_param_path = './parameters/d_440.prm'

with open(real_melody_path, 'rb') as f:
    real_melody = pickle.load(f)[:10]
with open(real_chord_path, 'rb') as f:
    real_chord = pickle.load(f)[:10]

real_score = np.concatenate([real_melody, real_chord], axis=-1)
fake_score = np.load(fake_score_path)

real_score = torch.tensor(real_score).to(device='cuda:0')
fake_score = torch.tensor(fake_score).to(device='cuda:0')

fake_score = torch.unsqueeze(fake_score, dim=0)
real_score = torch.reshape(real_score, (10, 8, 16, 52))
fake_score = torch.reshape(fake_score, (1, 8, 16, 52))

disc = Discriminator()
disc.to(device='cuda:0')
disc.load_state_dict(torch.load(disc_param_path))

real_disc_result = disc(real_score)
fake_disc_result = disc(fake_score)

print('real discrimination result')
print(real_disc_result)
print('fake discrimination result')
print(fake_disc_result)
