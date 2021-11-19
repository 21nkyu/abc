from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

detector = HandDetector(maxHands=1, detectionCon=0.2, minTrackCon=0.2)  # 모델 토기화


cap_cam = cv2.VideoCapture(0)  # 캠 설정
cap_video = cv2.VideoCapture('video001.mp4')  # 비디오 설정


w = int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # 캠 가로길이
# print(w)
total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수 237
# print(total_frames) # 237

_, video_img = cap_video.read()  # 첫번째 프레임을 읽는다. 정지상태


# 타임라인
def draw_timeline(video_img, rel_x):
    img_h, img_w, img_c = video_img.shape  # 이미지 높이 넓이
    # rel_x = lm_list[4][0] / w -> 엄지의 x좌표 / 캠의 가로길이 => 좌표가 0~1사이 값으로 변환됨
    # timeline_w = 비디오 가로 * 비율(엄지의 좌표)
    timeline_w = int(img_w * rel_x)
    # 사각형 = 시작(좌상단), 끝(우하단) -> 좌표가 필요하다 tickness = -1 -> 채워진막대
    cv2.rectangle(video_img, pt1=(0, img_h-(img_h-5)), pt2=(timeline_w, img_h-(img_h-7)), color=(0, 0, 255),
                  thickness=-1)


# 타임라인 변수 초기화
rel_x = 0
frame_idx = 0
draw_timeline(video_img, rel_x)  # video_img = 정지, 엄지좌표(rel_x) = 0


while cap_cam.isOpened():
    ret, cam_img = cap_cam.read()  # 카메라 프레임을 읽는다.
    if not ret:
        break

    cam_img = cv2.flip(cam_img, 1)  # 거울 반전

    hands, cam_img = detector.findHands(cam_img)  # 손의 랜드마크를 찾는다

    if hands:
        lm_list = hands[0]['lmList']  # 랜드마크 리스트
        # fingerUp : 손가락 들면 그 위치에 1 내리면 0을 나타내주는 기능 -> return [0,0,0,0,0]
        fingers = detector.fingersUp(hands[0])   # 손가락을 들면 1 내리면 0을 [0,0,0,0,0]손가락 위치에 따라 판별해주는 기능

        # cam_img에 정보를 나타내기 위해 text입력
        cv2.putText(cam_img, text=str(fingers), org=(10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255),
                    thickness=3)
        # print(lm_list)
        # findDistance : 원하는 관절의 거리를 계산 -> return length info img
        length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img) #거리계산
        # print(length)

        if fingers == [0, 0, 0, 0, 0]:
            pass  # 손가락이 모두0 -> 주먹을 쥐었다면 -> stop
        else:
            if length < 50:  # navigate 탐색
                rel_x = (lm_list[4][0] / w)

                frame_idx = int(rel_x * total_frames)  # 엄지손가락 x 좌표에 따른 동영상의 프레임 번호계산

                # 예외처리를 통해 오류방지
                # frame_idx -> 0보다 작아지면 0, total_frame 보다 크면 total_frame
                if frame_idx < 0:
                    frame_idx = 0
                elif frame_idx > total_frames:
                    frame_idx = total_frames

                cap_video.set(1, frame_idx)  # 동영상 해당 프레임 idx 로 이동

                # cam_img 에 text 삽입
                cv2.putText(cam_img, text='navigate %.2f, %d' % (rel_x, frame_idx), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255,255,255), thickness=3)
                # lm_list[4][0] # (x, y, z)
            else:            # play 재생
                frame_idx = frame_idx + 1
                rel_x = frame_idx / total_frames
            if frame_idx < total_frames:
                _, video_img = cap_video.read()
                video_img = cv2.resize(video_img, (int(video_img.shape[1]*0.5), int(video_img.shape[0]*0.3)))
                draw_timeline(video_img, rel_x)

    cv2.imshow('cam', cam_img)
    cv2.imshow('vd', video_img)
    # cv2.imshow('video', video_img)
    if cv2.waitKey(1) == ord('q'):
        break