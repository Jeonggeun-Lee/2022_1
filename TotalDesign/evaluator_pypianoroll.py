import pypianoroll
import numpy as np

mt = pypianoroll.read('G_FC_600epochs.mid')
mt.set_resolution(resolution=12)

melody = mt[0].pianoroll
chord = mt[1].pianoroll

melody = np.pad(melody, ((0, 384-381), (0, 0)), constant_values=((0, 0), (0, 0)))
chord = np.pad(chord, ((0, 384-339), (0, 0)), constant_values=((0, 0), (0, 0)))

print(len(melody))
print(len(chord))
td = pypianoroll.tonal_distance(melody, chord, resolution=12)
print('tonal distance', td)