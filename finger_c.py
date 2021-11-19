import cv2
import time
import os
from modules import HandTrackingModule2 as htm

# size of cam
wCam, hCam = 640, 480

# activate cam, use the wCam and hCam set the cam size
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


# img folder
folder_path = 'finger_images'
# search the folder
my_list = os.listdir(folder_path)
print(my_list)

# finger_images/p1.png ...
# overlaylist = []
# for img_path in my_list:
#     image = cv2.imread(f'{folder_path}/{img_path}')
#     # print(f'{folder_path}/img_path')
#     overlaylist.append(image)
# # print(len(overlaylist))
# # print(overlaylist)
overlaylist = [cv2.imread(f'{folder_path}/{img_path}') for img_path in my_list]


# detector
detector = htm.handDetector()

# 손가락 식별을 위한 인덱스 저장
finger_ids = [4, 8, 12, 16, 20]

# read the img from cam frames
p_time = 0
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # get landmarks coordinate (idx, x, y)
    img = detector.findHands(img, draw=True)  # self, img, draw
    lm_list = detector.findPosition(img, draw=False)
    # print(lm_list)

    # use the landmarks
    if len(lm_list) != 0:
        # y좌표를 이용해서 내가 지정한 손가락관절 인덱스보다 높이 있는가 낮게 있는가를 판별해서 손가락을 들었는지 내렸는지를 판별 할 것이다
        # lm_list = [[idx, x, y],[],[],[]...]
        # y 좌표 아래로 커짐
        fingers = []

        # 엄지의 판별은 일반적이지 않음
        # x 좌표를 이용해서 판변 오른손만 작동하고 왼손은 반대로 작동함
        if lm_list[finger_ids[0]][1] > lm_list[finger_ids[0] - 1][1]:
            fingers.append(1)
            # print('index finger up')
        else:
            fingers.append(0)
        # 나머지 네개 손가락의 판별은 동일함
        for id in range(1,5):
            if lm_list[finger_ids[id]][2] < lm_list[finger_ids[id] - 2][2]:
                fingers.append(1)
                # print('index finger up')
            else:
                fingers.append(0)
        print(fingers)

        # fingers 안의 1의 숫자를 세서 overlaylist 안의 인덱스에 접근해서 overlaylist내의 사진으로 변환한다
        total_fingers = fingers.count(1)

        h, w, c = overlaylist[total_fingers-1].shape
        img[:h, :w] = overlaylist[total_fingers-1]

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)



    cv2.imshow('image', img)
    # cv2.imshow('f', overlaylist[0])
    if cv2.waitKey(1) == ord('q'):
        break