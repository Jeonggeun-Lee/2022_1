# ChordMuseGAN 평가지표 정리

## 평가 지표가 필요한 부분

- 데이터셋의 타당성 지표

- 코드 생성 모델의 코드의 정밀도 평가

- Discriminator 의 손실 함수

- 생성이 완료된 음악의 음악성 평가

## 데이터셋의 타당성 지표

#### MuseGAN 논문의 정량적 지표를 사용하면 문제 없어보임 (이정근)

> Hao-Wen Dong,* Wen-Yi Hsiao,* Li-Chia Yang, and Yi-Hsuan Yang (* equal contribution) Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI), 2018.

1. 트랙 내부적 평가 척도

> Harte, C.; Sandler, M.; and Gasser, M. 2006. Detecting harmonic change in musical audio. In ACM MM workshop on Audio and music computing multimedia.

- EB(ratioof empty bar) : 공백 마디의 비율. %단위

- UPC(number of used pitch classes) : 마디 별 사용 음계 종류의 숫자(0~12)

- QN(ratioof qualified notes) : 32분의 1박자 이상의 음표의 비율. %단위. 음악이 지나치게 분절되었는가를 측정

- DP(drum pattern) : 록 음악에서 일반적인 4/4박자에서 8 또는 16비트 패턴의 음표의 비율. %단위
2) 트랙간 평가 척도
- TD(tonal distance) : 한 쌍의 트랙 간의 조화(harmonicity). TD가 크면 트랙간의 조화 관계가 약함. 참고 논문(Harte, Sandler, and Gasser 2006)에 따르면 Harmonic Change Distance Function을 사용. 이것은 하나의 화음을 12차원 벡터(크로마 벡터)로 표현하고 이것을 음계 간의 조화에 기초한 6차원 공간으로 변환하여 거기서 유클리디안 거리를 구한 것. HCDF 값을 각 시간 프레임마다 측정

## 코드 생성 모델의 코드의 정밀도 평가

- 코드 진행은 특별한 규칙이 없으며, 기승전결의 특징은 멜로디에 좌우되는 요소라고 생각되기 때문에, 특별한 정답을 제시하기 어려움

- 학습 시 오차와 음악의 기승전결을 형성하는 최소한의 포인트만 적용해보는 것이 어떨까 함

### 정량적 지표

> K. Choi, J. Park, W. Heo, S. Jeon and J. Park, "Chord Conditioned Melody Generation With Transformer Based Decoders," in IEEE Access, vol. 9, pp. 42071-42080, 2021, doi: 10.1109/ACCESS.2021.3065831.

- 코드 기반의 음악 생성 논문에서 코드 생성 모델의 성능을 측정할 수 있는 방법이 없음

- BLSTM의 학습 과정에서 발생한 오차로 측정 (Accuracy Test)

### 정성적 지표

- 생성된 코드가 주어진 곡의 Scale 에 맞지 않는 코드가 있는지

- 생성된 코드 진행의 마지막 부분이 1도 5도로 끝나는지

## Discriminator 의 손실 함수

- MuseGAN: minmax CE 함수

- **고려 필요**

## 생성이 완료된 음악의 음악성 평가

### MuseGAN

> Hao-Wen Dong,* Wen-Yi Hsiao,* Li-Chia Yang, and Yi-Hsuan Yang (* equal contribution) Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI), 2018.

앞서 데이터셋 평가 지표를 그대로 적용하는 것에 문제가 없음.

### 추가 사항 (안민준)

> K. Choi, J. Park, W. Heo, S. Jeon and J. Park, "Chord Conditioned Melody Generation With Transformer Based Decoders," in IEEE Access, vol. 9, pp. 42071-42080, 2021, doi: 10.1109/ACCESS.2021.3065831.

- MuseGAN은 코드 진행을 생성하는 부분이 없기 때문에 코드 진행에 합당한지 고려가 따로 필요하다.

- 위 논문에서는 대표적으로 `Code Tone Ratio` 지표를 사용함.
  
  - Code Tone Ratio 란 마디 내부 음들이 코드의 구성음에 얼마나 해당되는지 비율을 나타냄
  
  - 마디의 첫 음은 대부분 코드의 구성음이어야 하므로, 첫 음부분의 CTR 을 따로 계산함
  
  - (본 논문의 성과는 각각 72.53, 80.63)
