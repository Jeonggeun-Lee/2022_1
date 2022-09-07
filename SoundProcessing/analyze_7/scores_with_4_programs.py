import csv
import pypianoroll

if __name__ == '__main__':
    dir_path = '..\\lmd_aligned.tar\\lmd_aligned\\'
    f = open('../label.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    genres = "['Pop_Rock']"
    num_program = [0]*128
    num_scores = 0

    out_f = open('../analyze_5/scores_with_4_programs.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(out_f)

    score_num = 0
    rdr.__next__()

    while True:
        try:
            line = rdr.__next__()
        except:
            break
        score_num += 1
        print()
        print("score:", score_num)
        if line == None:
            break
        if not (line[0] in genres):
            continue
        file_name = line[2].replace('.npz', '.mid')
        try:
            s = pypianoroll.read(dir_path+file_name)
        except:
            print('exception')
            continue
        flag = [False]*128

        for t in s.tracks:

            if 0 <= t.program <= 127:
                flag[t.program] = True

        if flag[0] and flag[25] and flag[33] and flag[48]:
            num_scores += 1
            print(num_scores)
            print(file_name)
            wr.writerow([file_name])
    out_f.close()

'''
1: 8142###
2: 1474###
3: 442
4: 204
5: 1055
6: 822
7: 228
8: 221
9: 291
10: 153
11: 122
12: 621
13: 117
14: 29
15: 46
16: 33
17: 822
18: 476
19: 909
20: 116
21: 75
22: 185
23: 268
24: 53
25: 1542###
26: 3339###
27: 1496
28: 2221
29: 1151
30: 1894
31: 1835
32: 121
33: 1281
34: 3771###
35: 724
36: 2634###
37: 135
38: 159
39: 563
40: 566
41: 397
42: 95
43: 213
44: 100
45: 158
46: 185
47: 253
48: 155
49: 2668###
50: 1758
51: 1099
52: 287
53: 1560###
54: 994
55: 676
56: 88
57: 398
58: 277
59: 103
60: 44
61: 338
62: 560
63: 579
64: 182
65: 196
66: 836
67: 593
68: 126
69: 296
70: 131
71: 54
72: 335
73: 123
74: 1005###
75: 136
76: 457
77: 22
78: 64
79: 143
80: 132
81: 318
82: 645
83: 473
84: 60
85: 160
86: 202
87: 8
88: 371
89: 416
90: 607
91: 479
92: 198
93: 64
94: 94
95: 235
96: 196
97: 58
98: 12
99: 51
100: 206
101: 197
102: 29
103: 81
104: 69
105: 64
106: 89
107: 37
108: 33
109: 36
110: 23
111: 42
112: 5
113: 18
114: 6
115: 52
116: 56
117: 82
118: 44
119: 151
120: 396
121: 159
122: 22
123: 68
124: 11
125: 47
126: 41
127: 37
128: 142
'''
