import numpy as np
from music21 import *
from MuseGAN.melodyMuseGAN.config.trainer_config import trainer_config
import pickle

chord_diff_threshold = 10000

def num_different_pitches(chord1, chord2):
    chord1 = set(chord1)
    chord2 = set(chord2)
    diff = chord1 - chord2
    return len(diff)

def np_to_music21(fake_data_path = 'result/generated_score.npy'):
    is_real_data = False
    score_num = 1

    num_parts = 1
    num_measures = 8
    measure_steps = 16
    melody_and_onset_bits = 40
    num_chord_pitches = 12
    num_steps = 128
    batch_size = 1
    #0~24
    lowest_pitch = 1
    highest_pitch = 23
    quarter_len_ratio = 1/4
    melody_base_pitch = 47
    chord_base_pitch = 60-12

    onset_bit = 1
    hold_bit = 2
    rest_bit = 0

    if is_real_data:
        chord_path = trainer_config['data_path']['chord']
        melody_path = trainer_config['data_path']['np_array']

        with open(chord_path, "rb") as f:
            chord_data = pickle.load(f)
        with open(melody_path, "rb") as f:
            melody_data = pickle.load(f)

        chord_data = chord_data.tolist()
        chord_data = chord_data[score_num]
        chord_data = np.array(chord_data)
        melody_data = melody_data.tolist()

        melody_data = melody_data[score_num]
        melody_data = np.array(melody_data)
        chord_data = chord_data.reshape((num_measures, measure_steps, num_chord_pitches))
        melody_data = melody_data.reshape((num_measures, measure_steps, melody_and_onset_bits))
    else:
        fake_data = np.load(fake_data_path)
        print('fake_data.shape', fake_data.shape)
        #fake_data = np.squeeze(fake_data, axis=0)
        fake_data = np.split(fake_data, [40], axis=2)
        melody_data = fake_data[0]
        chord_data = fake_data[1]

    melody_and_onset = np.split(melody_data, [37], axis=2)
    melody_data = melody_and_onset[0]
    onset_data = melody_and_onset[1]

    melody_data = np.argmax(melody_data, axis=2)
    onset_data = np.argmax(onset_data, axis=2)

    #print(onset_data)

    score = stream.base.Score()
    melody_part = stream.Part()
    chord_part = stream.Part()
    score.append(melody_part)
    score.append(chord_part)

    num_ebs = 0
    sum_qn = 0
    num_notes = 0
    sum_upc = 0

    prev_chord = []
    melody_note = None
    note_type = None
    last_measure = -1

    note_count = 0
    pitch_adjust_count = 0
    pitch_adjust_amount = 0
    chord_tone_corresponds = 0

    for m in range(num_measures):
        eb_flag = True
        melody_measure = stream.base.Measure()
        chord_measure = stream.base.Measure()
        upc = set()

        chord_pitches = []
        for p in range(num_chord_pitches):
            if chord_data[m][0][p]:
                chord_pitches.append(p)
        for p in chord_pitches:
            d = duration.Duration()
            d.quarterLength = quarter_len_ratio * 4
            chord_note = note.Note(midi=chord_base_pitch + p, duration=d)
            chord_measure.append(chord_note)
        if len(chord_pitches) < 4:
            d = duration.Duration()
            d.quarterLength = quarter_len_ratio * 4
            chord_note = note.Note(midi=chord_base_pitch + chord_pitches[1], duration=d)
            chord_measure.append(chord_note)

        for s in range(measure_steps):
            #멜로디
            if onset_data[m][s] == onset_bit:
                chord_pitches = np.array(chord_pitches)
                melody_pitch = melody_data[m][s]+melody_base_pitch
                chord_pitches = np.concatenate([chord_pitches, (12+chord_pitches), (24 + chord_pitches), (36 + chord_pitches), ((36+12) + chord_pitches), ((36+12*2) + chord_pitches), ((36+12*3) + chord_pitches)], axis=-1)
                min_dist = 10000000
                nearest_pitch = melody_pitch
                for cp in chord_pitches:
                    dist = abs(cp-melody_pitch)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pitch = cp

                if min_dist <= 1:
                    melody_note = note.Note(midi=nearest_pitch)
                    chord_tone_corresponds += 1
                else:
                    melody_note = note.Note(midi=melody_pitch)
                    nearest_pitch = melody_pitch
                melody_note.quarterLength = quarter_len_ratio
                melody_measure.append(melody_note)
                eb_flag = False
                upc.add(melody_note.pitch.midi % 12)
                note_type = 'Note'
                num_notes += 1

                note_count += 1
                if nearest_pitch != melody_pitch:
                    pitch_adjust_count += 1
                    pitch_adjust_amount += abs(nearest_pitch - melody_pitch)

            elif onset_data[m][s] == hold_bit:
                if melody_note != None:
                    melody_note.quarterLength += quarter_len_ratio
                    if melody_note.quarterLength == 0.75 and note_type =='Note':
                        sum_qn += 1
                else:
                    melody_note = note.Rest()
                    melody_note.quarterLength = quarter_len_ratio
                    melody_measure.append(melody_note)
            else:
                melody_note = note.Rest()
                melody_note.quarterLength = quarter_len_ratio
                melody_measure.append(melody_note)
                note_type = 'Rest'
            #멜로디 끝

        if eb_flag:
            num_ebs += 1
        sum_upc += len(upc)
        melody_part.append(melody_measure)
        chord_part.append(chord_measure)

    eb = num_ebs/num_measures
    upc = sum_upc/num_measures
    qn = sum_qn/num_notes

    print('eb: '+str(eb))
    print('upc: '+str(upc))
    print('qn: '+str(qn))
    print('note count: '+str(note_count))
    print('pitch adjust count: '+str(pitch_adjust_count))
    print('pitch adjust amount: ' + str(pitch_adjust_amount))
    print(r'pac/nc: '+str(pitch_adjust_count/note_count))
    print(r'paa/pac: '+str(pitch_adjust_amount/pitch_adjust_count))
    print(r'paa/nc: '+str(pitch_adjust_amount/note_count))
    print(r'ctr: '+str(chord_tone_corresponds/note_count))

    score.show()

if __name__=='__main__':
    np_to_music21()
#################################
# G_LSTM, D_LSTM
# 670 epochs

# chord 0
# eb: 0.0
# upc: 3.5
# qn: 0.2692307692307692
# 'Code Tone Ratio': 0.0546875

# chord 1
# eb: 0.0
# upc: 4.375
# qn: 0.0625
# 'Code Tone Ratio': 0.2109375

# chord 2
# eb: 0.0
# upc: 3.875
# qn: 0.08196721311475409
# 'Code Tone Ratio': 0.1953125

# chord 3
# eb: 0.0
# upc: 4.375
# qn: 0.12280701754385964
# 'Code Tone Ratio': 0.140625

# chord 4
# eb: 0.0
# upc: 4.25
# qn: 0.08333333333333333
# 'Code Tone Ratio': 0.15625

#################################
# G_FC, D_FC
# 100 epochs

# chord 0
# eb: 0.0
# upc: 3.0
# qn: 0.6666666666666666
# 'Code Tone Ratio': 0.1171875

# chord 1
# eb: 0.0
# upc: 3.125
# qn: 0.5714285714285714
# 'Code Tone Ratio': 0.1015625

# chord 2
# eb: 0.0
# upc: 3.25
# qn: 0.75
# 'Code Tone Ratio': 0.1015625

# chord 3
# eb: 0.0
# upc: 3.375
# qn: 0.75
# 'Code Tone Ratio': 0.1015625

# chord 4
# eb: 0.0
# upc: 3.25
# qn: 0.7419354838709677
# 'Code Tone Ratio': 0.0625
################################
# G_FC, D_FC
# 500 epochs

# chord 0
# eb: 0.0
# upc: 2.75
# qn: 0.625
# 'Code Tone Ratio': 0.15625

# chord 1
# eb: 0.0
# upc: 2.5
# qn: 0.6956521739130435
# 'Code Tone Ratio': 0.1640625

# chord 2
# eb: 0.0
# upc: 2.25
# qn: 0.6521739130434783
# 'Code Tone Ratio': 0.15625

# chord 3
# eb: 0.0
# upc: 3.125
# qn: 0.6666666666666666
# 'Code Tone Ratio': 0.1171875

# chord 4
# eb: 0.0
# upc: 3.25
# qn: 0.6428571428571429
# 'Code Tone Ratio': 0.0703125


################################
# G_FC, D_LSTM
# 700 epochs

# chord 0
# eb: 0.0
# upc: 2.875
# qn: 0.5
# 'Code Tone Ratio': 0.109375

# chord 1
# eb: 0.0
# upc: 3.5
# qn: 0.45
# 'Code Tone Ratio': 0.078125

# chord 2
# eb: 0.0
# upc: 3.125
# qn: 0.5151515151515151
# 'Code Tone Ratio': 0.078125

# chord 3
# eb: 0.0
# upc: 3.25
# qn: 0.45714285714285713
# 'Code Tone Ratio': 0.03125

# chord 4
# eb: 0.0
# upc: 3.25
# qn: 0.5
# 'Code Tone Ratio': 0.0546875
