import torch
import torch.nn as nn
from config.model_config import model_config
from config.common_config import common_config

dis_config = model_config['discriminator']
num_tracks = common_config['num_tracks']
num_steps = common_config['num_steps']
num_measures = common_config['num_measures']
num_pitch_bins = common_config['num_pitch_bins']


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.config: dict = dis_config
    self.device = common_config['device']
    self.rnn = nn.LSTM(**dis_config['rnn'])
    self.bn = nn.BatchNorm1d(num_steps * num_tracks * num_pitch_bins)
    self.fc = nn.Linear(**dis_config['fc'])
    self.leaky_relu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    """
    입력 타입: x.shape == (Batch,)
    """
    x = x.type(torch.float)
    x = x.to(self.device)
    x = torch.transpose(x, 1, 2)
    x = torch.reshape(x, (-1, num_steps, num_tracks*num_pitch_bins))
    x, (_, _) = self.rnn(x)
    x = torch.reshape(x, (-1, num_steps * num_tracks * num_pitch_bins))
    x = self.bn(x)
    x = torch.reshape(x, (-1, num_steps, num_tracks * num_pitch_bins))
    x = torch.transpose(x, 0, 1)
    x = x[-1]
    # x = self.bn(x)
    x = self.leaky_relu(x)
    x = torch.squeeze(x, dim=0)
    output = self.fc(x)
    output = self.sigmoid(output)
    return output
