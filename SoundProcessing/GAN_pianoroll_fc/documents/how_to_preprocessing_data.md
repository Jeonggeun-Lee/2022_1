# 데이터 전처리 방식

MuseGAN과 다르게 우리 모델은 코드 진행에 대한 고려가 포함되어 있어서 이에 대한 전처리도 필요합니다.



# 사용 데이터 결정 문제

- Lakh 피아노롤 데이터
  
  - 기존에 있는 데이터에 코드를 추론해서 붙여야 함
  
  - 코드를 추론할 때 파트별로 코드가 다르면 어떡할 지 정해야 함

- OpenEWLD (?)
  
  - 코드는 이미 나와있지만 우리가 따로 전처리 해야 함

# 기초 전처리

MuseGAN의 기본적인 전처리 스타일을 따라 가고, 코드에 대한 부분만 추가로 요청하는 부분으로 사용해도 될 듯

## MuseGAN 전처리

- 심볼 타이밍을 사용하여 템포 정보를 폐기합니다.
- 속도 정보 폐기(2진수 값 피아노롤 사용)
- 음고 84개 가능성(C1~B7)
- 트랙을 5가지 범주로 병합: 베이스, 드럼, 기타, 피아노, 현악기
- 록(Rock) 태그가 있는 곡만 고려

따라서, 목표 출력 텐서의 크기 = 4(bar)×96(time step)×84(pitch)×5(track)

## 이외 코드 전처리

- 코드 키 통일 (Pitch Shift) 
  
  > K. Choi, J. Park, W. Heo, S. Jeon and J. Park, "Chord Conditioned Melody Generation With Transformer Based Decoders," in IEEE Access, vol. 9, pp. 42071-42080, 2021, doi: 10.1109/ACCESS.2021.3065831.
  
  - 코드 시퀀스가 너무 많아지는 것을 방지하기 위해 모두 C 키 또는 Am 키 (다장조 또는 가단조) 로 통일
  
  - C와 Am은 조표가 붙지 않는 나란한조이기 때문에 같은 조
  
  - 본 논문에서는 코드 생성때만 이를 적용하고 멜로디 생성 때는 사용하지 않았다는 언급이 있어서 확인 필요

- 코드 생략
  
  > L. C. Yang, S. Y. Chou, and Y. H. Yang, ‘‘MidiNet: A convolutional generative adversarial network for symbolic-domain music generation,’’ in Proc. ISMIR, Suzhou, China, 2017, pp. 324–331.
  
  - 각 12음 별로 Maj, min 부분만 추출하여 Triad 코드로 전환, 코드를 단순화함 (총 24개 코드)
    
    

- 코드 음 입력
  
  > K. Choi, J. Park, W. Heo, S. Jeon and J. Park, "Chord Conditioned Melody Generation With Transformer Based Decoders," in IEEE Access, vol. 9, pp. 42071-42080, 2021, doi: 10.1109/ACCESS.2021.3065831.
  
  - 코드를 단순히 one-hot vector로 토크나이징 하면 코드 간의 관계를 명확하게 파악하기 어렵다고 함
  
  - 코드의 구성음들에 대한 binary 저장 (총 12개)



### 파트 전처리 방법

Lakh 를 사용하지 않고 따로 전처리 해야 하는 경우에 지정할 필요성이 있음

- 각 트랙을 8 마디로 분할(Sliding Window 4마디)

- 쉼표의 길이는 전체의 25% 이하로 지정

- 음의 범위를 초과하는 데이터에 대해서  octave shift
