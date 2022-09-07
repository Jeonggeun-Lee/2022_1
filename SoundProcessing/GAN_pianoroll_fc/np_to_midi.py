import numpy as np
import pypianoroll


import os
import pickle
os.environ['KMP_DUPLICATE_LIB_OK']='True'

target_programs = [0, 25, 33, 48]
num_tracks = 4
num_steps = 128
num_measures = 8
steps_per_measure = num_steps//num_measures
steps_per_note = 4
num_pitch_bins = 128

file_name = 'score_g_4000' # 1000 is the best
src_path = './results/'+file_name+'.npy'
result_path = './results/'+file_name+'.mid'

with open(src_path, 'rb') as f:
    generatedMIDI = np.load(f)

ts = []
for i in range(num_tracks):
    st = pypianoroll.StandardTrack(pianoroll=generatedMIDI[i])
    ts.append(st)

mt = pypianoroll.Multitrack()
mt.set_resolution(4)

for st in ts:
    mt.append(st)
mt.tracks[0].program = 0
mt.tracks[1].program = 25
mt.tracks[2].program = 33
mt.tracks[3].program = 48

pypianoroll.write(result_path, multitrack=mt)
