from piano_roll_dataset import DS
from torch.utils.data import DataLoader
from config import config
import torch
from model import Classifier
from train import eval_net

if __name__ == '__main__':
    print('test')
    batch_size = config['batch_size']
    piano_roll_test = DS(config['x_test_path'], config['y_test_path'])
    parameter = config['parameter_path']+config['parameter']
    net = Classifier()
    net.load_state_dict(torch.load(parameter))
    test_loader = DataLoader(piano_roll_test, batch_size, shuffle=False, drop_last=True)
    print('test_acc: ', eval_net(net, test_loader))
