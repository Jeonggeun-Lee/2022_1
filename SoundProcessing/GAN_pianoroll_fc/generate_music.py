from config.common_config import common_config
import torch
import pickle
import numpy as np
from model.generator import Generator


num_tracks = common_config['num_tracks']

parameter_path = './parameters/'
parameter = 'g_4000'
threshold = 0.8 #0.5125, 0.515625, 0.51875, 0.521875, 0.525, 0.53125, 0.534375, 0.5375, 0.55, 0.575 0.665
if __name__ == '__main__':
    g = Generator()
    g.to(common_config['device'])
    checkpoint = torch.load(parameter_path + parameter+'.prm')
    g.load_state_dict(checkpoint)
    g.eval()
    score = g(1)
    score = torch.squeeze(score, dim=0)
    print(score.shape)
    score = score.tolist()
    score = np.array(score, dtype=np.float32)
    score[score > threshold] = 80
    score[score <= threshold] = 0
    score = score.astype(dtype=np.uint8)
    print(score[0, 1])
    print(score.shape)
    np.save('./results/score_'+parameter, score)