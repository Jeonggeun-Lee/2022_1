import csv
import pypianoroll

if __name__ == '__main__':
    num_gen_notes_set = set()
    qlen_set = set()
    dir_path = '../lmd_aligned.tar/lmd_aligned\\'
    f = open('scores_with_4_programs.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    out_f = open('scores_with_unique_4_programs.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(out_f)
    score_num = 0
    while True:
        try:
            line = rdr.__next__()
        except:
            break
        score_num += 1
        print(score_num)
        file_name = line[0]
        s = pypianoroll.read(dir_path+file_name)

        num_each_valid_program = [0]*4
        for t in s.tracks:
            if t.program == 0:
                num_each_valid_program[0] += 1
            elif t.program == 25:
                num_each_valid_program[1] += 1
            elif t.program == 33:
                num_each_valid_program[2] += 1
            elif t.program == 48:
                num_each_valid_program[3] += 1
        if num_each_valid_program == [1, 1, 1, 1]:
            wr.writerow([file_name])
    out_f.close()