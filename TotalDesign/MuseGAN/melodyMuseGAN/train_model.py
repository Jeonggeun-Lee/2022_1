from MuseGAN.melodyMuseGAN.model.generator import Generator
from MuseGAN.melodyMuseGAN.model.discriminator import Discriminator
from MuseGAN.melodyMuseGAN.trainer.trainer import Trainer
from MuseGAN.melodyMuseGAN.config.trainer_config import trainer_config
from MuseGAN.melodyMuseGAN.config.common_config import common_config
import torch.optim as optim
import torch

TRAINED_EPOCHS = 200

GEN_PARAM_PATH = 'C:/Users/Administrator/PycharmProjects/TotalDesign/MuseGAN/DG_GAN/parameters/g_{:03d}.prm'.format(TRAINED_EPOCHS)
DISC_PARAM_PATH = 'C:/Users/Administrator/PycharmProjects/TotalDesign/MuseGAN/DG_GAN/parameters/d_{:03d}.prm'.format(TRAINED_EPOCHS)

def train_model():
  g = Generator()
  d = Discriminator()
  #이정근 추가
  g.to(common_config['device'])
  d.to(common_config['device'])

  if TRAINED_EPOCHS != 0:
    g.state_dict(torch.load(GEN_PARAM_PATH))
    d.state_dict(torch.load(DISC_PARAM_PATH))

  g_optimizer = optim.Adam(g.parameters(), lr=trainer_config['g_lr'])
  d_optimizer = optim.Adam(d.parameters(), lr=trainer_config['d_lr'])

  chord_data_path = trainer_config['data_path']['chord']
  melody_data_path = trainer_config['data_path']['np_array']
  trainer = Trainer(g, d, g_optimizer, d_optimizer, chord_data_path, melody_data_path, trained_epochs=TRAINED_EPOCHS)

  trainer.train()

if __name__ == '__main__':
  train_model() # trained 1000