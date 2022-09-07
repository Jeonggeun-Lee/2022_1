import torch
import torch.nn as nn

from MuseGAN.melodyMuseGAN.config.model_config import model_config
from MuseGAN.melodyMuseGAN.config.common_config import common_config
import torch.nn.functional as F

generator_config = model_config['generator']

class Generator(nn.Module):
    def __init__(self) -> object:
        super().__init__()
        self.config: dict = generator_config
        self.device = common_config['device']
        self.fc_zt1 = nn.Linear(**generator_config['fc_zt1'])
        self.batch_norm1 = nn.BatchNorm1d(generator_config['fc_zt1']['out_features'])
        self.fc_zt2 = nn.Linear(**generator_config['fc_zt2'])
        self.batch_norm2 = nn.BatchNorm1d(generator_config['fc_zt2']['out_features'])
        self.fc_zt3 = nn.Linear(**generator_config['fc_zt3'])
        self.batch_norm3 = nn.BatchNorm1d(generator_config['fc_zt3']['out_features'])
        self.fc_zt4 = nn.Linear(**generator_config['fc_zt4_more'])
        self.batch_norm4 = nn.BatchNorm1d(generator_config['fc_zt4_more']['out_features'])
        self.fc_zt5 = nn.Linear(**generator_config['fc_zt4_more'])
        self.batch_norm5 = nn.BatchNorm1d(generator_config['fc_zt4_more']['out_features'])
        self.fc_zt6 = nn.Linear(**generator_config['fc_zt4_more'])
        self.batch_norm6 = nn.BatchNorm1d(generator_config['fc_zt4_more']['out_features'])
        self.fc_zt7 = nn.Linear(**generator_config['fc_zt4_more'])
        self.batch_norm7 = nn.BatchNorm1d(generator_config['fc_zt4_more']['out_features'])
        self.fc_zt8 = nn.Linear(**generator_config['fc_zt4_more'])
        self.batch_norm8 = nn.BatchNorm1d(generator_config['fc_zt4_more']['out_features'])
        # self.fc_zt9 = nn.Linear(**model_config['fc_zt4_more'])
        # self.batch_norm9 = nn.BatchNorm1d(model_config['fc_zt4_more']['out_features'])
        # self.fc_zt10 = nn.Linear(**model_config['fc_zt4_more'])
        # self.batch_norm10 = nn.BatchNorm1d(model_config['fc_zt4_more']['out_features'])

        self.fc_bar1 = nn.Linear(**generator_config['fc_bar1'])
        self.batch_norm1_ = nn.BatchNorm1d(generator_config['fc_bar1']['out_features'])
        self.fc_bar2 = nn.Linear(**generator_config['fc_bar2'])
        self.batch_norm2_ = nn.BatchNorm1d(generator_config['fc_bar2']['out_features'])
        self.fc_bar3 = nn.Linear(**generator_config['fc_bar3_more'])
        self.batch_norm3_ = nn.BatchNorm1d(generator_config['fc_bar3_more']['out_features'])
        self.fc_bar4 = nn.Linear(**generator_config['fc_bar3_more'])
        self.batch_norm4_ = nn.BatchNorm1d(generator_config['fc_bar3_more']['out_features'])
        self.fc_bar5 = nn.Linear(**generator_config['fc_bar3_more'])
        self.batch_norm5_ = nn.BatchNorm1d(generator_config['fc_bar3_more']['out_features'])
        self.fc_bar6 = nn.Linear(**generator_config['fc_bar3_more'])
        self.batch_norm6_ = nn.BatchNorm1d(generator_config['fc_bar3_more']['out_features'])
        self.fc_bar7 = nn.Linear(**generator_config['fc_bar3_more'])
        self.batch_norm7_ = nn.BatchNorm1d(generator_config['fc_bar3_more']['out_features'])
        self.fc_bar8 = nn.Linear(**generator_config['fc_bar3_more'])
        # self.batch_norm8_ = nn.BatchNorm1d(model_config['fc_bar3_more']['out_features'])
        # self.fc_bar9 = nn.Linear(**model_config['fc_bar3_more'])
        # self.batch_norm9_ = nn.BatchNorm1d(model_config['fc_bar3_more']['out_features'])
        # self.fc_bar10 = nn.Linear(**model_config['fc_bar3_more'])

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        입력 타입(화음 데이터): x.shape == (Batch, 128, 12)
        """
        x = x.type(torch.float32)
        x = x.to(self.device)
        x = x.reshape((-1, model_config['num_bars'], model_config['steps_per_bar'], model_config['chord_bins']))
        batch_size = x.shape[0]

        zt = torch.randn(batch_size, model_config['steps_per_bar'] * model_config['note_size'] // 2).to(self.device)
        z = torch.randn(batch_size, model_config['steps_per_bar'], model_config['note_size'] // 2).to(self.device)
        if self.train:
            z.requires_grad = True
            zt.requires_grad = True
        z = [z] * model_config['num_bars']
        z = torch.stack(z)
        z = torch.transpose(z, 0, 1)
        zt = self.leaky_relu(self.batch_norm1(self.fc_zt1(zt)))
        zt = self.leaky_relu(self.batch_norm2(self.fc_zt2(zt)))
        zt = self.leaky_relu(self.batch_norm3(self.fc_zt3(zt)))
        zt = self.leaky_relu(self.batch_norm4(self.fc_zt4(zt)))
        zt = self.leaky_relu(self.batch_norm5(self.fc_zt5(zt)))
        zt = self.leaky_relu(self.batch_norm6(self.fc_zt6(zt)))
        zt = self.leaky_relu(self.batch_norm7(self.fc_zt7(zt)))
        zt = self.leaky_relu(self.batch_norm8(self.fc_zt8(zt)))
        # zt = self.leaky_relu(self.batch_norm9(self.fc_zt9(zt)))
        # zt = self.leaky_relu(self.batch_norm10(self.fc_zt10(zt)))
        # zt = self.leaky_relu((self.fc_zt1(zt)))
        # zt = self.leaky_relu((self.fc_zt2(zt)))
        # zt = self.leaky_relu((self.fc_zt3(zt)))
        zt = zt.reshape((batch_size, model_config['num_bars'], model_config['steps_per_bar'], model_config['note_size'] // 2))
        bar_in = torch.concat([zt, z, x], dim=-1)
        bar_in = bar_in.reshape((batch_size, model_config['num_bars'], model_config['steps_per_bar'] * model_config['bits_per_step']))
        #bar_in = torch.transpose(bar_in, 0, 1)
        bar_in_split = torch.split(bar_in, 1, dim=1)
        #print(bar_in_split[0].shape)
        # 1, batch, step*bits
        bar_out_list = []
        for t in range(model_config['num_bars']):
            bar = bar_in_split[t]
            bar = torch.squeeze(bar, dim=1) # batch, step*bits
            #temp = self.sigmoid(self.fc_bar2(self.leaky_relu(self.batch_norm4(self.fc_bar1(bar)))))
            #temp = self.sigmoid(self.fc_bar2(self.leaky_relu((self.fc_bar1(bar)))))
            temp = self.leaky_relu(self.batch_norm1_(self.fc_bar1(bar)))
            temp = self.leaky_relu(self.batch_norm2_(self.fc_bar2(temp)))
            temp = self.leaky_relu(self.batch_norm3_(self.fc_bar3(temp)))
            temp = self.leaky_relu(self.batch_norm4_(self.fc_bar4(temp)))
            temp = self.leaky_relu(self.batch_norm5_(self.fc_bar5(temp)))
            temp = self.leaky_relu(self.batch_norm6_(self.fc_bar6(temp)))
            temp = self.leaky_relu(self.batch_norm7_(self.fc_bar7(temp)))
            # temp = self.leaky_relu(self.batch_norm8_(self.fc_bar8(temp)))
            # temp = self.leaky_relu(self.batch_norm9_(self.fc_bar9(temp)))
            temp = self.sigmoid(self.fc_bar8(temp))
            # batch, 1, step, note
            bar_out_list.append(temp.reshape(batch_size, 1, model_config['steps_per_bar'], model_config['note_size']))

        pitch_list = []
        onset_list = []
        for t in range(model_config['num_bars']):
            bar_split = torch.split(bar_out_list[t], [model_config['pitch_bins'], model_config['onset_bins']], dim=-1)
            pitch_list.append(bar_split[0])# batch, 1, step, pitch
            onset_list.append(bar_split[1])# batch, 1, step, onset

        x_split = torch.split(x, 1, dim=1)
        bar_list = []
        for t in range(model_config['num_bars']):
            if self.train:
                pitch_tensor = pitch_list[t]
                onset_tensor = onset_list[t]
            else:
                pitch_tensor = F.one_hot(torch.argmax(pitch_list[t], dim=-1), num_classes=model_config['pitch_bins'])
                onset_tensor = F.one_hot(torch.argmax(onset_list[t], dim=-1), num_classes=model_config['onset_bins'])
            bar_list.append(torch.concat([pitch_tensor, onset_tensor, x_split[t]], dim=-1))

        outputs = torch.concat(bar_list, dim=1)# batch, bar, step, note&chord

        return outputs
