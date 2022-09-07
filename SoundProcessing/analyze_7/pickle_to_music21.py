import csv
import pypianoroll
from music21 import *
from fractions import Fraction
import numpy as np
import pickle


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

max_num_gen_notes = 52
target_programs = [0, 25, 33, 48]
program_to_index = {}
index_to_program = {}
for i, pr in enumerate(target_programs):
    index_to_program[i] = pr
    program_to_index[pr] = i

qlen_list = [Fraction(1, 12), Fraction(1, 6), 0.25, Fraction(1, 3), Fraction(5, 12), 0.5, Fraction(7, 12), Fraction(2, 3), 0.75, Fraction(5, 6), Fraction(11, 12), 1.0, Fraction(13, 12), Fraction(7, 6), 1.25, Fraction(4, 3), Fraction(17, 12), 1.5, Fraction(19, 12), Fraction(5, 3), 1.75, Fraction(11, 6), Fraction(23, 12), 2.0, Fraction(25, 12), Fraction(13, 6), 2.25, Fraction(7, 3), Fraction(29, 12), 2.5, Fraction(31, 12), Fraction(8, 3), 2.75, Fraction(17, 6), Fraction(35, 12), 3.0, Fraction(37, 12), Fraction(19, 6), 3.25, Fraction(10, 3), Fraction(41, 12), 3.5, Fraction(43, 12), Fraction(11, 3), 3.75, Fraction(23, 6), 4.0, Fraction(25, 6), 4.25, Fraction(13, 3), 4.5, Fraction(14, 3), 4.75, 5.0, 5.5, 6.0, 6.25, 6.75, 7.5, 9.0, 9.75, 10.75, Fraction(133, 12), 26.0]

index_to_qlen = {}
qlen_to_index = {}
for i, qlen in enumerate(qlen_list):
    index_to_qlen[i] = qlen
    qlen_to_index[qlen] = i

rest_bit = 0
note_bit = 1
chord_bit = 2
min_pitch_bit = 3
min_qlen_bit = 131
score_shape = (4, max_num_gen_notes, 3+128+64)

with open('train_data.pkl', 'rb') as f:
    data = pickle.load(f)

score_num = 50

data = data[score_num]

print(data[0][0])

score = stream.Score()
for i in range(4):
    part = stream.Part()
    for j in range(52):
        note_type = np.argmax(data[i][j][:3])
        if note_type == rest_bit:
            gn_note = note.Rest()
        elif note_type == note_bit:
            mi = np.argmax(data[i][j][min_pitch_bit:min_qlen_bit])
            gn_note = note.Note(midi=mi)
        elif note_type == chord_bit:
            arg_sort = np.argsort(data[i][j][min_pitch_bit:min_qlen_bit])
            arg_top3 = arg_sort[-3:]
            gn_note = chord.Chord(arg_top3.tolist())
        gn_note.quarterLength = index_to_qlen[np.argmax(data[i][j][min_qlen_bit:])]
        part.append(gn_note)
    score.append(part)
score.show()

'''
////////////////////////////////
min num gen notes 32
score with min num gen notes Y\\L\\K\\TRYLKWH128E0781BBA\\29abf584d5dadf811a9f9b4ecf7e343f.mid
max num gen notes 743
score with max num gen notes T\\D\\E\\TRTDEQO128F92CCCF0\\2db8dad5f753b029d2e11190faa215cf.mid
////////////////////////////////
qlen list [Fraction(1, 12), Fraction(1, 6), 0.25, Fraction(1, 3), Fraction(5, 12), 0.5, Fraction(7, 12), Fraction(2, 3), 0.75, Fraction(5, 6), Fraction(11, 12), 1.0, Fraction(13, 12), Fraction(7, 6), 1.25, Fraction(4, 3), Fraction(17, 12), 1.5, Fraction(19, 12), Fraction(5, 3), 1.75, Fraction(11, 6), Fraction(23, 12), 2.0, Fraction(25, 12), Fraction(13, 6), 2.25, Fraction(7, 3), Fraction(29, 12), 2.5, Fraction(31, 12), Fraction(8, 3), 2.75, Fraction(17, 6), Fraction(35, 12), 3.0, Fraction(37, 12), Fraction(19, 6), 3.25, Fraction(10, 3), Fraction(41, 12), 3.5, Fraction(43, 12), Fraction(11, 3), 3.75, Fraction(23, 6), 4.0, Fraction(25, 6), 4.25, Fraction(13, 3), 4.5, Fraction(14, 3), 4.75, 5.0, 5.5, 6.0, 6.25, 6.75, 7.5, 9.0, 9.75, 10.75, Fraction(133, 12), 26.0]
64
'''