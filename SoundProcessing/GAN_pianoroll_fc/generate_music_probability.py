from config.common_config import common_config
import torch
import pickle
import numpy as np
from model.generator import Generator


num_tracks = common_config['num_tracks']
num_steps = common_config['num_steps']
parameter_path = './parameters/'
parameter = 'g_1460'


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

if __name__ == '__main__':
    g = Generator()
    g.to(common_config['device'])
    checkpoint = torch.load(parameter_path + parameter+'.prm')
    g.load_state_dict(checkpoint)
    g.eval()
    score = g(1)
    score = torch.squeeze(score, dim=0)
    score = score.tolist()
    score = np.array(score, dtype=np.float64)

    new_score = np.zeros_like(score)
    for i in range(num_tracks):
        for j in range(num_steps):
            new_score[i][j] = np.random.multinomial(1, softmax(score[i][j]), size=1)*80

    new_score = new_score.astype(dtype=np.uint8)
    print(new_score[0, 1])
    print(new_score.shape)
    np.save('./results/score_'+parameter, new_score)