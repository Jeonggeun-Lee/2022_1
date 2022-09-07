# trainer.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.DS import DS
from config.common_config import common_config
from config.trainer_config import trainer_config

from statistics import mean
import numpy as np

class Trainer:
  def __init__(self, g, d, g_optimizer, d_optimizer, data_path):
    self.g = g
    self.d = d
    self.g_optimizer = g_optimizer
    self.d_optimizer = d_optimizer

    self.device = common_config['device']
    self.batch_size = common_config['batch_size']
    self.dataloader = DataLoader(DS(data_path), batch_size=common_config['batch_size'], shuffle=True)
    self.criterion = torch.nn.BCELoss()

    self.g.to(self.device)
    self.d.to(self.device)

    self.d_losses = []
    self.g_losses = []



  def train(self, epochs=trainer_config['epoch']):
    loss_log_g = []
    loss_log_d = []
    for epoch in range(epochs):
      self.g.train()
      self.d.train()
      for real_score in tqdm(self.dataloader):
        real_score.to(self.device)

        fake_score = self.g()
        fake_score_tensor = fake_score.detach()
        out = self.d(fake_score)

        loss_g = self.criterion(out, torch.ones_like(out))
        self.g_losses.append(loss_g.item())

        self.d.zero_grad(), self.g.zero_grad()
        loss_g.backward()

        self.g_optimizer.step()

        real_out = self.d(real_score)
        loss_d_real = self.criterion(real_out, torch.ones_like(real_out))

        fake_score = fake_score_tensor

        fake_out = self.d(fake_score)
        loss_d_fake = self.criterion(fake_out, torch.zeros_like(fake_out))

        loss_d = loss_d_real + loss_d_fake
        self.d_losses.append(loss_d.item())

        self.d.zero_grad(), self.g.zero_grad()
        loss_d.backward()
        if self.g_losses[-1] < 88/75 * self.d_losses[-1]:
          self.d_optimizer.step()


      print()
      print(f"EPOCH: {epoch+1}/{epochs} | g_loss: {self.g_losses[-1]:.4f} | d_loss: {self.d_losses[-1]:.4f}")
      loss_log_g.append(mean(self.g_losses))
      loss_log_d.append(mean(self.d_losses))
      loss_log_g_arr = np.array(loss_log_g)
      loss_log_d_arr = np.array(loss_log_d)
      np.save('./loss_log/loss_log_g', loss_log_g_arr)
      np.save('./loss_log/loss_log_d', loss_log_d_arr)
      if (epoch+1)%10 == 0:
        torch.save(self.g.state_dict(), './parameters/g_{:03d}.prm'.format(epoch+1), pickle_protocol=4)
        torch.save(self.d.state_dict(), './parameters/d_{:03d}.prm'.format(epoch+1), pickle_protocol=4)