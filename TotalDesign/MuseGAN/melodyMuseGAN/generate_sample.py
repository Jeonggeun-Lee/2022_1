from MuseGAN.melodyMuseGAN.model.generator import Generator
from MuseGAN.melodyMuseGAN.config.common_config import common_config
import torch
import numpy as np
import pickle

#chord_path = '../chordMuseGAN_out/result/chord_result.npy'
chord_path = '../chordLSTM/result/generated_chord_imsi 3.npy'
#chord_num = 1

def generate_melody_sample(chord_path='../chordMuseGAN/result/generated_chord (26).npy', parameter_path='./parameters/G_FC_D_FC_g_600.prm'):
    mg = Generator()
    mg.to(common_config['device'])
    checkpoint = torch.load(parameter_path)
    mg.load_state_dict(checkpoint)
    mg.eval()

    # with open('C:/Users/Administrator/PycharmProjects/TotalDesign/MuseGAN/melodyMuseGAN/data/chord_train_data.pkl', 'rb') as f:
    #     chord = pickle.load(f)[chord_num]
    chord = np.load(chord_path)
    chord = chord.astype(np.int16)
    chord = torch.tensor(chord)
    chord = torch.unsqueeze(chord, 0)
    score = mg(chord)

    score = torch.squeeze(score, dim=0)
    score = score.tolist()
    score = np.array(score)
    shape0 = score.shape[0]
    shape1 = score.shape[1]
    shape2 = score.shape[2]
    score = np.reshape(score, (shape0*shape1, shape2))
    score_split = np.split(score, [37, 40], axis=1)
    melody = score_split[0]
    temp_shape = melody.shape[1]
    melody = np.argmax(melody, axis=-1)
    melody = np.eye(temp_shape)[melody]
    melody = torch.tensor(melody)
    onset = score_split[1]
    chord = score_split[2]
    onset = np.argmax(onset, axis=1)
    for i in range(len(melody)):
        if onset[i] == 0:
            melody[i] = torch.zeros_like(melody[i])
        if i != 0 and onset[i] == 2:
            melody[i] = melody[i-1]
    score[:, :37] = melody

    score = np.reshape(score, (shape0, shape1, shape2))
    #np.save('./DG_GAN/result/generated_score', score)
    np.save('C:/Users/Administrator/PycharmProjects/TotalDesign/MuseGAN/melodyMuseGAN/result/generated_score', score)
if __name__=='__main__':
    generate_melody_sample(chord_path='../chordLSTM/result/generated_chord_imsi 2.npy')