import numpy as np
import random as rd
import torch
from MuseGAN.chordRNN.model.generator import Generator
import pickle

def generate_chord_sample(data, parameter_path='parameter/lstm.prm', chord_result_path='result/generated_chord'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device).eval()
    generator.load_state_dict(torch.load(parameter_path))
    data_num = rd.randrange(len(data))
    chord = data[data_num]
    chord = np.reshape(chord, (8, 16, 12))
    first_bar = chord[0]
    bar = torch.tensor(first_bar).to(device=device, dtype=torch.float)
    bar = torch.unsqueeze(bar, dim=0)
    bar_list = []
    bar_list.append(bar)
    for b in range(7):
        bar = generator(bar)
        bar_list.append(bar)
    chord = torch.concat(bar_list, dim=0)
    chord = torch.reshape(chord, (128, 12))
    new_chord = torch.zeros_like(chord)
    for step in range(128):
        new_chord[step][torch.multinomial(chord[step], 3)] = 1

    result = np.array(new_chord.cpu().tolist())
    np.save(chord_result_path, result)

if __name__ == '__main__':
    with open('../chordRNN/data/data_train.pkl', 'rb') as f:
        data = pickle.load(f)
    generate_chord_sample(data)