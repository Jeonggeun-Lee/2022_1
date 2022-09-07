import torch
import torch.nn as nn
from config import config

batch_size = config['batch_size']
num_programs = config['num_programs']
num_steps = config['num_steps']
max_pitch = config['max_pitch']

class Classifier(nn.Module):
    def __init__(self)-> object:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(**config['classifier']['conv1']),
            nn.LeakyReLU(),
            nn.BatchNorm2d(**config['classifier']['bn2d1']),
            nn.Conv2d(**config['classifier']['conv2']),
            nn.LeakyReLU(),
            nn.BatchNorm2d(**config['classifier']['bn2d2']),
            nn.Conv2d(**config['classifier']['conv3']),
            nn.LeakyReLU(),
            nn.BatchNorm2d(**config['classifier']['bn2d3']),
            nn.Conv2d(**config['classifier']['conv4']),
            nn.LeakyReLU(),
            nn.BatchNorm2d(**config['classifier']['bn2d4']),
            nn.Conv2d(**config['classifier']['conv5']),
            nn.LeakyReLU(),
            nn.BatchNorm2d(**config['classifier']['bn2d5']),
            nn.Conv2d(**config['classifier']['conv6']),
            nn.LeakyReLU(),
            nn.BatchNorm2d(**config['classifier']['bn2d6']),
            nn.Conv2d(**config['classifier']['conv7']),
            nn.LeakyReLU(),
            nn.BatchNorm2d(**config['classifier']['bn2d7']),
        )
        self.fc = nn.Sequential(
            nn.Linear(**config['classifier']['fc1']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d1']),
            nn.Linear(**config['classifier']['fc2']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d2']),
            nn.Linear(**config['classifier']['fc3']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d3']),
            nn.Linear(**config['classifier']['fc4']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d4']),
            nn.Linear(**config['classifier']['fc5']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d5']),
            nn.Linear(**config['classifier']['fc6']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d6']),
            nn.Linear(**config['classifier']['fc7']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d7']),
            nn.Linear(**config['classifier']['fc8']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(**config['classifier']['bn1d8']),
            nn.Linear(**config['classifier']['fc9']),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(**config['classifier']['bn1d9']),
            # nn.Linear(**config['classifier']['fc10']),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(**config['classifier']['bn1d10']),
            # nn.Linear(**config['classifier']['fc11']),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(**config['classifier']['bn1d11']),
            #nn.Linear(**config['classifier']['fc12']),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(**config['classifier']['bn1d12']),
            #nn.Linear(**config['classifier']['fc13'])
        )
        self.softmax = nn.Softmax(**config['classifier']['softmax'])

    def forward(self, x):
        x = self.conv(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.fc(x)
        output = self.softmax(x)
        return output
