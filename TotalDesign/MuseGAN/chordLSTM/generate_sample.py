from lstm_generator import generate_chord_sample
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_pickle("data/data_train.pkl")
    generate_chord_sample(data)
