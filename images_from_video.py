import cv2
video = cv2.VideoCapture('./video.mp4')
# 计数
n = 1

if video.isOpened():
    rval, frame = video.read()
else:
    rval = False

timeF = 6 # 帧间隔频率

i = 0
while rval:
    rval, frame = video.read()
    if n % timeF == 0:
        i += 1
        print(i)
        cv2.imwrite('./data/nerf_llff_data/my_data/images/{:03d}.JPG'.format(i), frame)
    n += 1
video.release()