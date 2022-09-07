import csv
import pypianoroll
from music21 import *
from fractions import Fraction
import numpy as np
import pickle

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


dir_path = '..\\lmd_aligned.tar\\lmd_aligned\\'
f = open('scores_with_unique_4_programs.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

ngn_f = open('number_of_general_notes.csv', 'r', encoding='utf-8')
ngn_rdr = csv.reader(ngn_f)

def has_max_num_gen_notes(file_name, ngn_rdr):
    num_gen_notes_info = None
    while True:
        try:
            line = ngn_rdr.__next__()
        except:
            break
        if file_name == line[0]:
            num_gen_notes_info = line[1:]
            break
    if not num_gen_notes_info:
        return False
    else:
        for n in num_gen_notes_info:
            if int(n) < max_num_gen_notes:
                return False
        return True

data = []

score_num = 0
while True:
    try:
        line = rdr.__next__()
    except:
        break
    score_num += 1
    print(score_num)

    file_name = line[0]

    if not has_max_num_gen_notes(file_name, ngn_rdr):
        continue

    s = pypianoroll.read(dir_path+file_name)
    score = converter.parse(dir_path+file_name)
    parts = score.getElementsByClass(stream.Part)

    program_to_part = {}
    for i in range(len(s.tracks)):
        if s.tracks[i].program in target_programs:
            program_to_part[s.tracks[i].program] = i

    sc = np.zeros(score_shape, dtype=np.float32)
    for pr in target_programs:
        flatten_part = parts[program_to_part[pr]].flatten()
        gen_notes = flatten_part.getElementsByClass(note.GeneralNote)
        gen_notes = gen_notes[:max_num_gen_notes]

        i = 0
        for gn in gen_notes:
            if isinstance(gn, note.Rest):
                sc[program_to_index[pr], i, rest_bit] = 1
                sc[program_to_index[pr], i, min_qlen_bit + qlen_to_index[gn.quarterLength]] = 1
            elif isinstance(gn, note.Note):
                sc[program_to_index[pr], i, note_bit] = 1
                sc[program_to_index[pr], i, min_pitch_bit + gn.pitch.midi] = 1
                sc[program_to_index[pr], i, min_qlen_bit + qlen_to_index[gn.quarterLength]] = 1
            elif isinstance(gn, chord.Chord):
                sc[program_to_index[pr], i, chord_bit] = 1
                for pit in gn.pitches:
                    sc[program_to_index[pr], i, min_pitch_bit + pit.midi] = 1
                sc[program_to_index[pr], i, min_qlen_bit + qlen_to_index[gn.quarterLength]] = 1
            i += 1
    data.append(sc)

data = np.stack(data, axis=0)
print('data.shape', data.shape)

f.close()
ngn_f.close()

with open('../GAN_flatten/data/train_data.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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