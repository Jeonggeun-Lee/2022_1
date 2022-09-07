import cv2 #VideoCapture, imwrite
import os #path.exists, makedirs

filepath = './dataset/hop_far.mp4'
video = cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)

#불러온 비디오 파일의 정보 출력
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)

#프레임을 저장할 디렉토리를 생성
try:
    if not os.path.exists(filepath[:-4]):
        os.makedirs(filepath[:-4])
except OSError:
    print ('Error: Creating directory. ' +  filepath[:-4])

count = 0

while (video.isOpened()):
    ret, image = video.read()
    try:
        cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
    except:
        break
    print('Saved frame number :', str(int(video.get(1))))
    count += 1

video.release()