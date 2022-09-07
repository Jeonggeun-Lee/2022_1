import numpy as np
import pandas as pd


class Evaluator:
  def __init__(self, target, target_chords)->None:
    self.target = target
    self.target_chords = target_chords
    self.eb:float = 0.0
    self.upc:float = 0.0
    self.qn:float = 0.0
    self.ctr:float = 0.0
    self.quantization = 16
    self.eval_result = None
  
  def evaluate_eb(self)->float:
    # eb는 연주 마지막의 3비트를 가지고 측정합니다. Rest, onset, holding 이므로 Rest 의 비중을 세면 됩니다.
    note_status = self.target[:, -3]
    num_measure = len(note_status) / self.quantization # 데이터의 길이

    for idx in range(0, len(note_status), self.quantization): # 8마디 간격으로 돌면서 빈 마디를 구합니다.
      if np.sum(note_status[idx:idx+self.quantization]) == self.quantization: # 모두가 쉼표면 마디에 1추가 
        self.eb += 1.0
    self.eb /= num_measure # 전체 마디수로 나눠 평균을 구합니다.

    return self.eb

  def evaluate_upc(self)->float:
    # 마디 별 사용한 피치의 비중을 세는 것

    melodies = self.target[:, :-3]
    num_measure = len(melodies) / self.quantization # 데이터의 길이

    for idx in range(0, len(melodies), self.quantization): # 8마디 간격으로 돌면서 빈 마디를 구합니다.
      melody_count = np.sum(melodies[idx:idx+self.quantization, :], axis=0) # 전체 멜로디 수 만큼의 톤 개수를 구합니다.
      rest_measure = len(melody_count) % 12 # 계산의 편리성을 위해 배열을 12의 배수로 자릅니다.
      counted_measure = int(len(melody_count) / 12)
      pitch_count = np.sum(melody_count[:-rest_measure].reshape(12,-1), axis=1) # 각 멜로디별 개수를 구합니다.
      
      for i in range(rest_measure): # 남은 마디 카운트도 더해주기
        pitch_count[i] += melody_count[counted_measure + i]
      self.upc += np.sum(pitch_count > 0) # 1 이상인 행의 개수를 더하기

    self.upc /= num_measure # 모든 마디의 upc를 평균내기
    
    return self.upc

  def eveluate_qn(self)->float:
    # 3 timestep 을 구합니다.
    note_status = self.target[:, -3:]
    total_notes = np.sum(note_status[:, 1]) # onset 의 갯수와 노트갯수는 동일
    qn_count = 0 # 긴 음 카운트
    note_length = 0 # 노트의 길이 확인
    is_onset = False # 음이 연주중인지를 나타내는 임시 변수
    for idx in range(0, len(note_status), self.quantization): # 8마디 간격으로 돌면서 빈 마디를 구합니다.
      cut_measure = note_status[idx:idx+self.quantization]
      for m in cut_measure:
        if m[1] == 1: # onset 인 경우
          if is_onset and (note_length >= 3):
            is_onset = True 
            note_length = 1
            qn_count += 1
          else:
            is_onset = True
            note_length = 1
        elif m[0] == 1: # 쉼표에 걸린 경우
          if is_onset and (note_length >= 3):
            qn_count += 1
          is_onset = False 
          note_length = 1
        else: # holding인 경우
          note_length += 1
    self.qn = qn_count / total_notes
    return self.qn

  # def evaluate_ctr(self)->float:
  #   # 각 음과 코드 톤이 일치하는 비율을 나타냅니다.
  #   melodies = self.target[:, :-3]
  #
  #   # 코드랑 멜로디를 겹쳐보기 위한 윈도우
  #   num_octave = int(melodies.shape[1] / 12)
  #   # rest_pitch = 12 - melodies.shape[1] % 12
  #   chord_window = np.tile(self.target_chords, (1, num_octave + 1))[:, 11:] # TODO: 하드코딩된 부분 수정 현재는 B음부터 시작하므로
  #   match_notes = np.sum((melodies == chord_window) & (melodies != 0)) # 멜로디가 0이 아닌 것 들 중 chord window 와 일지하는 것의 개수
  #
  #   self.ctr = match_notes / np.sum(melodies)
  #
  #   return self.ctr

  def evaluate_ctr(self) -> float:
    # 각 음과 코드 톤이 일치하는 비율을 나타냅니다.
    melodies = self.target[:, :-3]
    melodies = np.pad(melodies, ((0, 0), (47, 128-(47+37))), constant_values=((0, 0), (0, 0)))
    chord = self.target_chords
    # 코드랑 멜로디를 겹쳐보기 위한 윈도우
    chord_window = np.tile(chord, (1, 11))
    chord_window = chord_window[:, :128]
    match_notes = np.sum((melodies == chord_window) & (melodies != 0))
    self.ctr = match_notes/np.sum(melodies)

    return self.ctr

  def evaluate(self)->dict:
    self.eval_result = {
      'Empty Bars': self.evaluate_eb(),
      'Used Pitch Class': self.evaluate_upc(),
      'Qualified Notes': self.eveluate_qn(),
      'Code Tone Ratio': self.evaluate_ctr()
    }

    return self.eval_result

  def show_df(self)->pd.DataFrame:
    if self.eval_result is None:
      self.evaluate()
      
    return pd.DataFrame([self.eval_result])
