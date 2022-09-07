import torch
import torch.nn as nn
from MuseGAN.melodyMuseGAN.config.model_config import model_config
from MuseGAN.melodyMuseGAN.config.common_config import common_config

discriminator_config = model_config['discriminator']

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.config: dict = model_config
    self.device = common_config['device']
    self.fc1 = nn.Linear(**discriminator_config['fc1'])
    # self.batch_norm1 = nn.BatchNorm1d(model_config['fc1']['out_features'])
    self.fc2 = nn.Linear(**discriminator_config['fc2'])
    self.leaky_relu = nn.LeakyReLU(0.2)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    """
    입력 타입: x.shape == (Batch, 8, 16, 52)
    """

    x = x.type(torch.float32)
    x = x.to(self.device)

    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    x = self.fc1(x)
    x = self.leaky_relu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    output = torch.squeeze(x)
    # output.shape == (batch,)

    return output