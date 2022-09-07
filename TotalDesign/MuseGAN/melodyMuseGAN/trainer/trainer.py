import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from MuseGAN.melodyMuseGAN.dataloader.dataset import Dataset
from MuseGAN.melodyMuseGAN.config.common_config import common_config
from MuseGAN.melodyMuseGAN.config.model_config import model_config
from MuseGAN.melodyMuseGAN.config.trainer_config import trainer_config

import numpy as np
import torch.nn as nn

class Trainer:
  def __init__(self, g, d, g_optimizer, d_optimizer, chord_data_path, melody_data_path, trained_epochs):

    self.g = g
    self.d = d
    self.g_optimizer = g_optimizer
    self.d_optimizer = d_optimizer

    self.device = common_config['device']
    self.batch_size = trainer_config['batch_size']
    self.dataloader = DataLoader(Dataset(chord_data_path, melody_data_path),
                                 batch_size=trainer_config['batch_size'], shuffle=True)
    # self.test_dataloader = DataLoader(ChordWithMelodyDataset(chord_data_path, melody_data_path),
    #                              batch_size=melody_trainer_config['batch_size'], shuffle=True)
    self.criterion = torch.nn.BCELoss()

    self.d_losses = []
    self.g_losses = []
    self.zeros = torch.zeros(self.batch_size).to(self.device)
    self.ones = torch.ones(self.batch_size).to(self.device)
    self.trained_epochs = trained_epochs

  def train(self, epochs=trainer_config['epoch']):

    loss_log_g = []
    loss_log_d = []
    for epoch in range(epochs):
      #이정근 변경
      self.g.train()
      self.d.train()

      for c, m in tqdm(self.dataloader):
        #####################
        c.requires_grad = True
        m.requires_grad = True

        batch_len = len(m)
        c = c.to(common_config['device'])
        m = m.to(common_config['device'])
        fake_score = self.g(c)
        fake_score_tensor = fake_score.detach()

        # Chord Tone Ratio(fake) - start
        score = fake_score.reshape((fake_score.shape[0], fake_score.shape[1]*fake_score.shape[2], fake_score.shape[3]))
        score_split = torch.split(score,
                                  (model_config['pitch_bins'], model_config['onset_bins'], model_config['chord_bins']),
                                  dim=-1)
        pitch = score_split[0]
        onset = score_split[1]
        chord = score_split[2]

        pad_pitch = nn.ConstantPad2d((47, 128 - (47 + 37), 0, 0), 0)
        pitch = pad_pitch(pitch)

        chord = torch.tile(chord, (1, 1, 11))
        chord = chord[:, :, :128]

        onset_split = torch.split(onset, (1, 1, 1), dim=-1)

        rest = onset_split[0]
        rest = torch.tile(rest, (1, 1, 128))
        one_tensor = torch.ones_like(rest)
        rest_invert = one_tensor - rest
        pitch_no_rest = torch.logical_and(pitch, rest_invert)

        hold = onset_split[2]
        hold = torch.tile(hold, (1, 1, 128))
        hold_invert = one_tensor - hold
        pitch_only_onset = torch.logical_and(pitch_no_rest, hold_invert)

        zero_tensor = torch.zeros_like(pitch)
        hold_integrated = torch.zeros_like(pitch)

        pitch_hold = pitch
        for i in range(1, 16, 1):
          pitch_shift = torch.concat([zero_tensor[:, :i, :], pitch_hold[:, :-i, :]], dim=1)
          pitch_hold = torch.logical_and(pitch_shift, hold)
          hold_integrated = torch.logical_or(hold_integrated, pitch_hold)

        pitch = torch.logical_or(pitch_only_onset, hold_integrated)

        melody_sum = torch.sum(pitch, dim=-1)
        melody_sum = torch.sum(melody_sum, dim=-1)

        match = torch.logical_and(pitch, chord)
        match_sum = torch.sum(match, dim=-1)
        match_sum = torch.sum(match_sum, dim=-1)

        ctr_fake = match_sum / melody_sum
        ctr_fake = torch.mean(ctr_fake)
        ctr_fake_tensor = ctr_fake.detach()
        # chord tone ratio(fake) - end

        real_score = torch.concat([m, c], dim=-1)  # 버그 위험
        real_score = real_score.reshape(-1, model_config['num_bars'], model_config['steps_per_bar'],
                                        model_config['bits_per_step'])

        # Chord Tone Ratio(real) - start
        score = real_score.reshape(
          (real_score.shape[0], real_score.shape[1] * real_score.shape[2], real_score.shape[3]))
        score_split = torch.split(score,
                                  (model_config['pitch_bins'], model_config['onset_bins'], model_config['chord_bins']),
                                  dim=-1)
        pitch = score_split[0]
        onset = score_split[1]
        chord = score_split[2]

        pad_pitch = nn.ConstantPad2d((47, 128 - (47 + 37), 0, 0), 0)
        pitch = pad_pitch(pitch)

        chord = torch.tile(chord, (1, 1, 11))
        chord = chord[:, :, :128]

        onset_split = torch.split(onset, (1, 1, 1), dim=-1)

        rest = onset_split[0]
        rest = torch.tile(rest, (1, 1, 128))
        one_tensor = torch.ones_like(rest)
        rest_invert = one_tensor - rest
        pitch_no_rest = torch.logical_and(pitch, rest_invert)

        hold = onset_split[2]
        hold = torch.tile(hold, (1, 1, 128))
        hold_invert = one_tensor - hold
        pitch_only_onset = torch.logical_and(pitch_no_rest, hold_invert)

        zero_tensor = torch.zeros_like(pitch)
        hold_integrated = torch.zeros_like(pitch)

        pitch_hold = pitch
        for i in range(1, 16, 1):
          pitch_shift = torch.concat([zero_tensor[:, :i, :], pitch_hold[:, :-i, :]], dim=1)
          pitch_hold = torch.logical_and(pitch_shift, hold)
          hold_integrated = torch.logical_or(hold_integrated, pitch_hold)

        pitch = torch.logical_or(pitch_only_onset, hold_integrated)

        melody_sum = torch.sum(pitch, dim=-1)
        melody_sum = torch.sum(melody_sum, dim=-1)

        match = torch.logical_and(pitch, chord)
        match_sum = torch.sum(match, dim=-1)
        match_sum = torch.sum(match_sum, dim=-1)

        ctr_real = match_sum / melody_sum
        ctr_real = torch.mean(ctr_real)
        # chord tone ratio(real) - end


        out = self.d(fake_score)

        loss_g = self.criterion(out, self.ones[:batch_len]) + trainer_config['ctr_coef']*torch.abs(ctr_real-ctr_fake)
        self.g_losses.append(loss_g.item())

        self.d.zero_grad(), self.g.zero_grad()
        loss_g.backward()
        self.g_optimizer.step()

        real_out = self.d(real_score)
        loss_d_real = self.criterion(real_out, self.ones[: batch_len])

        fake_score = fake_score_tensor

        fake_out = self.d(fake_score)
        loss_d_fake = self.criterion(fake_out, self.zeros[:batch_len]) + trainer_config['ctr_coef']*torch.abs(ctr_real-ctr_fake_tensor)

        loss_d = loss_d_real + loss_d_fake
        self.d_losses.append(loss_d.item())

        self.d.zero_grad(), self.g.zero_grad()
        if len(loss_log_g) > 0:
          if loss_log_g[-1] < loss_log_d[-1]:
             loss_d.backward()
             self.d_optimizer.step()
        else:
          loss_d.backward()
          self.d_optimizer.step()

      print()
      print('epoch:', self.trained_epochs+epoch+1, '/', self.trained_epochs+epochs, 'loss_g:', self.g_losses[-1], '/ loss_d:', self.d_losses[-1])
      loss_log_g.append(self.g_losses[-1])
      loss_log_d.append(self.d_losses[-1])
      loss_log_g_arr = np.array(loss_log_g)
      loss_log_d_arr = np.array(loss_log_d)
      np.save('./loss_log/loss_log_g', loss_log_g_arr)
      np.save('./loss_log/loss_log_d', loss_log_d_arr)
      if (self.trained_epochs+epoch+1) % 10 == 0:
        torch.save(
          self.g.state_dict(),
          trainer_config['parameter_path'] + '/g_{:03d}.prm'.format(self.trained_epochs+epoch+1),
          pickle_protocol=4)
        torch.save(
          self.d.state_dict(),
          trainer_config['parameter_path'] + '/d_{:03d}.prm'.format(self.trained_epochs+epoch+1),
          pickle_protocol=4)