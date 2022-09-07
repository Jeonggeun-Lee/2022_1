import csv
import pypianoroll
import numpy as np
import torch
from config import config

num_programs = config['num_programs']
num_bars = config['num_bars']
quarter_length_per_bar = config['quarter_length_per_bar']
steps_per_quarter_length = config['steps_per_quarter_length']
max_pitch = config['max_pitch']
num_steps = config['num_steps']

dir_path = '../../lmd_aligned.tar/lmd_aligned\\'
f = open('../../label.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

genre = [[], [], []]
rdr.__next__()
score_num = 0
while True:
    try:
        line = rdr.__next__()
    except:
        break
    score_num += 1
    print('score:', score_num)
    if line[0] != "['Country']" and line[0] != "['Electronic']" and line[0] != "['RnB']":
        continue
    file_name = line[2].replace('.npz', '.mid')
    try:
        s = pypianoroll.read(dir_path+file_name)
    except:
        continue
    s.set_resolution(steps_per_quarter_length)
    s.binarize()

    max_step = -1
    for t in range(len(s.tracks)):
        if len(s.tracks[t]) > max_step:
            max_step = len(s.tracks[t])

    temp_score = np.zeros((num_programs, max_step, max_pitch), dtype=np.bool8)
    for t in range(len(s.tracks)):
        if not s.tracks[t].program in [0, 1, 2, 3, 4, 5, 6, 7]:
            continue
        if len(s.tracks[t]) >= max_step:
            track = s.tracks[t][:max_step]
        else:
            diff = max_step - len(s.tracks[t])
            reciprocal = np.zeros((diff, max_pitch))
            track = np.concatenate([s.tracks[t], reciprocal], axis=0)
        if s.tracks[t].program in range(0, 8):
            temp_score[s.tracks[t].program] = np.logical_or(temp_score[s.tracks[t].program], track)
        else:
            temp_score[s.tracks[t].program-16] = np.logical_or(temp_score[s.tracks[t].program-16], track)

    if max_step < num_steps:
        quotient = num_steps // max_step
        remainder = num_steps % max_step
        temp_score = np.transpose(temp_score, [1, 0, 2])
        score = np.concatenate([temp_score]*quotient, axis=0)
        score = np.concatenate([score, temp_score[:remainder]], axis=0)
        score = np.transpose(score, [1, 0, 2])
    else:
        score = temp_score[:, :num_steps]

    print(score.shape)

    if line[0] == "['Country']":
        genre[0].append(score)
    elif line[0] == "['Electronic']":
        genre[1].append(score)
    elif line[0] == "['RnB']":
        genre[2].append(score)

for g in range(3):
    genre[g] = np.stack(genre[g], axis=0)
    print('genre['+str(g)+'].shape:', genre[g].shape)

array_country = genre[0]
array_electronic = genre[1]
array_rnb = genre[2]

train_country = array_country[:len(array_country)*6//10]
val_country = array_country[len(array_country)*6//10:len(array_country)*8//10]
test_country = array_country[len(array_country)*8//10:]
train_electronic = array_electronic[:len(array_electronic)*6//10]
val_electronic = array_electronic[len(array_electronic)*6//10:len(array_electronic)*8//10]
test_electronic = array_electronic[len(array_electronic)*8//10:]
train_rnb = array_rnb[:len(array_rnb)*6//10]
val_rnb = array_rnb[len(array_rnb)*6//10:len(array_rnb)*8//10]
test_rnb = array_rnb[len(array_rnb)*8//10:]

x_train = np.concatenate([train_country, train_electronic, train_rnb], axis=0)
x_val = np.concatenate([val_country, val_electronic, val_rnb], axis=0)
x_test = np.concatenate([test_country, test_electronic, test_rnb], axis=0)

y_train = [0]*len(train_country)+[1]*len(train_electronic)+[2]*len(train_rnb)
y_val = [0]*len(val_country)+[1]*len(val_electronic)+[2]*len(val_rnb)
y_test = [0]*len(test_country)+[1]*len(test_electronic)+[2]*len(test_rnb)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

tensor_x_train = torch.tensor(x_train, dtype=torch.bool)
tensor_y_train = torch.tensor(y_train)
tensor_x_val = torch.tensor(x_val, dtype=torch.bool)
tensor_y_val = torch.tensor(y_val)
tensor_x_test = torch.tensor(x_test, dtype=torch.bool)
tensor_y_test = torch.tensor(y_test)

torch.save(tensor_x_train, './data/x_train.pt', pickle_protocol=4)
torch.save(tensor_y_train, './data/y_train.pt', pickle_protocol=4)
torch.save(tensor_x_val, './data/x_val.pt', pickle_protocol=4)
torch.save(tensor_y_val, './data/y_val.pt', pickle_protocol=4)
torch.save(tensor_x_test, './data/x_test.pt', pickle_protocol=4)
torch.save(tensor_y_test, './data/y_test.pt', pickle_protocol=4)
