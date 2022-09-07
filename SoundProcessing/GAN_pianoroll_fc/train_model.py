# train_model.py

from model.discriminator import Discriminator
from model.generator import Generator
from trainer.trainer import Trainer
from config.common_config import common_config
from config.trainer_config import trainer_config
import torch.optim as optim

num_tracks = common_config['num_tracks']
num_measures = common_config['num_measures']
num_pitch_bins = common_config['num_pitch_bins']


def train_model():

  g = Generator()
  d = Discriminator()
  g_optimizer = optim.Adam(g.parameters(), lr=trainer_config['g_lr'])
  d_optimizer = optim.Adam(d.parameters(), lr=trainer_config['d_lr'])
  data_path = trainer_config['data_path']

  trainer = Trainer(g, d, g_optimizer, d_optimizer, data_path)

  trainer.train()

if __name__ == '__main__':
  train_model()
