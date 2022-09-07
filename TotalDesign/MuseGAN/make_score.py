#from MuseGAN.chordRNN.generate_sample import generate_chord_sample
from MuseGAN.chordLSTM.lstm_generator import generate_chord_sample
from MuseGAN.melodyMuseGAN.generate_sample import generate_melody_sample
from MuseGAN.melodyMuseGAN.np_to_music21_chord_split import np_to_music21
from MuseGAN.score_config import score_config
import pickle

if __name__ == '__main__':
    with open(score_config['chord_data_path'], 'rb') as f:
         data = pickle.load(f)
    generate_chord_sample(data, score_config['chord_g_path'], score_config['chord_path'][:-4])
    generate_melody_sample(score_config['chord_path'], score_config['melody_g_path'])
    np_to_music21(score_config['score_path'])
