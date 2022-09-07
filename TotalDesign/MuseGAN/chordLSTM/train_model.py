from lstm_generator import train
import pandas as pd

if __name__ == '__main__':
    data = pd.read_pickle("data/data_train.pkl")
    train(data)