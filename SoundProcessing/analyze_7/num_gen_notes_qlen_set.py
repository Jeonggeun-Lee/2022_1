import csv
import pypianoroll
from music21 import *

target_programs = [0, 25, 33, 48]

num_gen_notes_set = set()
qlen_set = set()

dir_path = '..\\lmd_aligned.tar\\lmd_aligned\\'
f = open('scores_with_unique_4_programs.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

out_f = open('../analyze_7/number_of_general_notes.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(out_f)

score_num = 0
min_num_gen_notes = 100000000000
score_with_min_num_gen_notes = -1
max_num_gen_notes = -1
score_with_max_num_gen_notes = -1

while True:
    try:
        line = rdr.__next__()
    except:
        break
    score_num += 1
    print(score_num)
    num_gen_notes_info = []
    file_name = line[0]
    num_gen_notes_info.append(file_name)
    s = pypianoroll.read(dir_path+file_name)
    score = converter.parse(dir_path+file_name)
    parts = score.getElementsByClass(stream.Part)

    program_to_part = {}
    for i in range(len(s.tracks)):
        if s.tracks[i].program in target_programs:
            program_to_part[s.tracks[i].program] = i

    for p in target_programs:
        flatten_part = parts[program_to_part[p]].flatten()
        gen_notes = flatten_part.getElementsByClass(note.GeneralNote)
        num_gen_notes = len(gen_notes)
        num_gen_notes_info.append(num_gen_notes)

        if num_gen_notes <= min_num_gen_notes:
            min_num_gen_notes = num_gen_notes
            score_with_min_num_gen_notes = file_name
            print('min updated', min_num_gen_notes)
        if num_gen_notes >= max_num_gen_notes:
            max_num_gen_notes = num_gen_notes
            score_with_max_num_gen_notes = file_name
            print('max updated', max_num_gen_notes)
        for gn in gen_notes:
            qlen_set.add(gn.quarterLength)

    wr.writerow(num_gen_notes_info)
    print(score_num, num_gen_notes_info)

print('////////////////////////////////')
print('min num gen notes', min_num_gen_notes)
print('score with min num gen notes', score_with_min_num_gen_notes)
print('max num gen notes', max_num_gen_notes)
print('score with max num gen notes', score_with_max_num_gen_notes)
print('////////////////////////////////')
qlen_list = list(qlen_set)
qlen_list.sort()
print('qlen list', qlen_list)
print(len(qlen_list))

f.close()
out_f.close()

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