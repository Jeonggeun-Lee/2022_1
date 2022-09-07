import torch
import torch.nn as nn

from config.model_config import model_config
from config.common_config import common_config

device = common_config['device']
num_steps = common_config['num_steps']
gen_config = model_config['generator']
num_tracks = common_config['num_tracks']
num_measures = common_config['num_measures']
num_pitch_bins = common_config['num_pitch_bins']
num_steps_per_measure = common_config['num_steps_per_measure']

class Generator(nn.Module):
  def __init__(self) -> object:
    super().__init__()
    self.config: dict = gen_config
    self.device = common_config['device']
    self.fc_track_generic = nn.Sequential(
      nn.Linear(**gen_config['fc_tr_g0']), nn.LeakyReLU(),
      nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins // 16),
      nn.Linear(**gen_config['fc_tr_g1']),
      #nn.BatchNorm1d(num_measures * num_steps_per_measure * num_pitch_bins // 4)
    ).to(device=device)
    if self.eval:
      self.fc_track_generic.eval()

    self.fc_track = [
      nn.Sequential(
        nn.Linear(**gen_config['fc_tr0']), nn.LeakyReLU(),
        nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins // 16),
        nn.Linear(**gen_config['fc_tr1']),
        #nn.BatchNorm1d(num_measures * num_steps_per_measure * num_pitch_bins // 4)
      ).to(device=device),
      nn.Sequential(
        nn.Linear(**gen_config['fc_tr0']), nn.LeakyReLU(),
        nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins // 16),
        nn.Linear(**gen_config['fc_tr1']),
        #nn.BatchNorm1d(num_measures * num_steps_per_measure * num_pitch_bins // 4)
      ).to(device=device),
      nn.Sequential(
        nn.Linear(**gen_config['fc_tr0']), nn.LeakyReLU(),
        nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins // 16),
        nn.Linear(**gen_config['fc_tr1']),
        #nn.BatchNorm1d(num_measures * num_steps_per_measure * num_pitch_bins // 4)
      ).to(device=device),
      nn.Sequential(
        nn.Linear(**gen_config['fc_tr0']), nn.LeakyReLU(),
        nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins // 16),
        nn.Linear(**gen_config['fc_tr1']),
        #nn.BatchNorm1d(num_measures * num_steps_per_measure * num_pitch_bins // 4)
      ).to(device=device)
    ]
    if self.eval:
      for seq in self.fc_track:
        seq.eval()


    self.fc_list = [
      nn.Sequential(
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc'])
      ),
      nn.Sequential(
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc'])
      ),
      nn.Sequential(
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc'])
      ),
      nn.Sequential(
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc']), nn.LeakyReLU(), nn.BatchNorm1d(num_steps_per_measure * num_pitch_bins),
        nn.Linear(**gen_config['fc'])
      )
    ]

    if self.eval:
      for seq in self.fc_list:
        seq.eval()


    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()


  def forward(self, batch_size=common_config['batch_size']):
    """
    입력 타입: x.shape ==
    """
    z0 = torch.randn((batch_size, num_steps_per_measure * num_pitch_bins // 256), dtype=torch.float).to(self.device)
    z1 = torch.randn((batch_size, 1, num_steps_per_measure * num_pitch_bins // 4), dtype=torch.float).to(self.device)
    zt = [
      [torch.randn((batch_size, num_steps // num_measures * num_pitch_bins // 256), dtype=torch.float).to(self.device),
       torch.randn((batch_size, num_steps // num_measures * num_pitch_bins // 256), dtype=torch.float).to(self.device),
       torch.randn((batch_size, num_steps // num_measures * num_pitch_bins // 256), dtype=torch.float).to(self.device),
       torch.randn((batch_size, num_steps // num_measures * num_pitch_bins // 256), dtype=torch.float).to(self.device)
       ],
      [torch.randn((batch_size, 1, num_steps // num_measures * num_pitch_bins // 4), dtype=torch.float).to(self.device),
       torch.randn((batch_size, 1, num_steps // num_measures * num_pitch_bins // 4), dtype=torch.float).to(self.device),
       torch.randn((batch_size, 1, num_steps // num_measures * num_pitch_bins // 4), dtype=torch.float).to(self.device),
       torch.randn((batch_size, 1, num_steps // num_measures * num_pitch_bins // 4), dtype=torch.float).to(self.device)
       ]
      ]
    z0.requires_grad = True
    z1.requires_grad = True

    for i in range(len(zt)):
      for j in range(len(zt[i])):
        zt[i][j].requires_grad = True


    z0 = self.fc_track_generic(z0) # z0 shape: (batach_size, num_measures * num_steps_per_measure * num_pitch_bins // 4)
    z0 = self.relu(z0)
    z0 = torch.reshape(z0, (-1, num_measures, num_steps_per_measure * num_pitch_bins//4))
    z0_split = torch.split(z0, 1, dim=1)
    z01_list = []
    for i in range(num_measures):
      z01_list.append(torch.concat([z0_split[i], z1], dim=-1))
    z01 = torch.concat(z01_list, dim=1)  # (-1, num_measures, num_steps // num_measures * num_pitch_bins//2)
    z01 = torch.unsqueeze(z01, dim=0) # (1, -1, num_measures, num_steps // num_measures * num_pitch_bins//2)
    zt_con_list = []
    for i in range(num_tracks):
      zt[0][i] = self.fc_track[i](zt[0][i])
      zt[0][i] = self.relu(zt[0][i])
      zt[0][i] = torch.reshape(zt[0][i], (-1, num_measures, num_steps_per_measure * num_pitch_bins//4))
      zt0_split = torch.split(zt[0][i], 1, dim=1)
      zt01_list = []
      for j in range(num_measures):
        zt01_list.append(torch.concat([zt0_split[j], zt[1][i]], dim=-1))
      zt01 = torch.concat(zt01_list, dim=1)
      zt_con_list.append(zt01)
    zt_con = torch.stack(zt_con_list, dim=1)  # (-1, num_tracks, num_measures, num_steps_per_measure * num_pitch_bins//2)
    zt_con = torch.transpose(zt_con, 0, 1)  # (num_tracks, -1, num_measures, num_steps_per_measure * num_pitch_bins//2)
    zt_con_spit = torch.split(zt_con, 1, dim=0)

    z_all_list = []
    for i in range(num_tracks):
      z_all_list.append(torch.concat([z01, zt_con_spit[i]], dim=-1))
    z_all = torch.concat(z_all_list, dim=0)
    z_all = torch.transpose(z_all, 0, 1) # (-1, num_tracks, num_measures, num_steps_per_measure * num_pitch_bins)

    z_all_split = torch.split(z_all, 1, dim=1)
    z_all_list = []
    for i in range(len(z_all_split)):
      #z_ss = torch.split(z_all_split[i], num_steps_per_measure, dim=2)
      z_ss = torch.split(z_all_split[i], 1, dim=2)
      z_ss = list(z_ss)
      z_ss_con = []
      for j in range(len(z_ss)):
        z_ss[j] = torch.reshape(z_ss[j], shape=(-1, num_steps_per_measure*num_pitch_bins))
        self.fc_list[i].to(device=device)
        z_ss[j] = self.fc_list[i](z_ss[j])
        z_ss[j] = torch.reshape(z_ss[j], shape=(-1, 1, 1, num_steps_per_measure * num_pitch_bins))
        z_ss_con.append(z_ss[j])
      z_ss_con = torch.concat(z_ss_con, dim=2)
      z_all_list.append(z_ss_con)
    z_all = torch.concat(z_all_list, dim=1)
    z_all = torch.reshape(z_all, shape=(-1, num_tracks, num_steps, num_pitch_bins))
    z_all = self.sigmoid(z_all)
    return z_all
