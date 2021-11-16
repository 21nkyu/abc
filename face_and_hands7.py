# Import Libraries
import cv2
import time
import mediapipe as mp
import numpy as np

# face: holistic
# hands: hands

# homography matrix 다루기 / cv2.findHomography( ), cv2.warpPerspective( )
# 큰 절차는 cv2.findhomography를 이용해 landmark를 찾고 cv2.warpperspective를 통해 합친다.
# 최소한 씌우고 싶은 이미지의 좌, 우 좌표를 양쪽 귀나 볼과 인덱스를 연결 -> 얼굴을 돌려도 완벽하지는 않지만 그림에 굴곡이 생기면서 얼굴 이미지를 커버한다
# 얼굴의 인덱스는 mediapipe로 구함
# mask01.png의 좌표는 mask01.csv파일에 저장 -> 1. csv파일을 읽는 함수 작성, 2. 읽은 파일은 idx(landmark) x y description(선택)로 구성 -> np.array로 변현
# 마스크에서 사용할 좌표를 저장
# 얼굴인식 landmark 추출
# output을 findhomography, warpperspective를 사용해서 연결 -> 이과정에서 정규화 -> 인덱스 슬라이싱 -> blur변형 등등 작업을 거쳐야함.


# 손인식 개수, 학습된 제스쳐
max_hands = 1
gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
}

# 사용할 제스쳐 rock paper scissors = rps
rps_gesture = {0: 'filter1', 5: 'filter2', 9: 'filter3'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=max_hands,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# # face
# facemesh = mp.solutions.face_mesh
# face = facemesh.FaceMesh()
# draw = mp.solutions.drawing_utils



# Grabbing the Holistic Model from Mediapipe
# Initializing the Model
# mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

# Gesture recognition model
# file = (99, 21)
# angle = (99, 20) column_indexes = 0~20
# label = (99, 1)  column_index   = 21
# opencv 에서 제공되는 machine learning 모델 중에서 knn 을(KNearest_create())사용
# knn.train 을 이용해서 학습
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

# load overlay images
overlay = cv2.imread('samples/face5.png', cv2.IMREAD_UNCHANGED)
overlay1 = cv2.imread('samples/face2.png', cv2.IMREAD_UNCHANGED)
overlay2 = cv2.imread('samples/face1.png', cv2.IMREAD_UNCHANGED)
overlay_scale = 2.0

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
          img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

        return bg_img
    except Exception: return background_img


while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # 좌우 반전 추가
    # resizing = frame -> flip, bgr2rgb = image 변수에 저장
    image = cv2.flip(frame, 1)
    # Converting the from from BGR to RGB
    # opencv: BGR
    # Mediapipe: RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손이미지 처리하기위해 변환된 image 를 hands.process 통해 result 변수에 저장
    result = hands.process(image)  # hands

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)  # holistic 을 이용해서 얼굴을 인식
    image.flags.writeable = True

    #########################################################################
    # writeable 사용하는 이유? 방법
    # Make an array immutable(read - only)
    # Z = np.zeros(10)
    # Z.flags.writeable = False
    # Z[0] = 1
    #
    # ValueError: assignment
    # destination is read - only
    #
    # Z.flags.writeable = False 로 설정함으로써 변수를 변경하는 것을 막았다.
    #########################################################################

    # Converting back the RGB image to BGR
    # 원래 캠의 색을 표현하기 위해 미디어파이프에서 사용한 RGB 를 다시 BGR 로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img1 = image.copy()

    # Drawing the Facial Landmarks
    # FACE_CONNECTIONS -> FACEMESH_TESSELATION 변경됨
    # 윤곽선 : FACEMESH_CONTOURS
    # https://github.com/google/mediapipe
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
    )

    # results = face.process(image)
    # if results.multi_face_landmarks is not None:
    #     # for idx in results.face_landmarks.landmark:
    #     #     print(idx)
    #
    #     for k in results.multi_face_landmarks:
    #         # org = (int(results.face_landmarks.landmark[4].x * image.shape[1]),
    #         #        int(results.face_landmarks.landmark[4].y * image.shape[0]))
    #
    #         draw.draw_landmarks(img1, k)
    #         print(k.landmark[0].x * int(width), k.landmark[0].y * int(height))
    #         cv2.put
    if results.face_landmarks is not None:
        # for idx in results.face_landmarks.landmark:
        #     print(idx)

        face_idx = np.zeros((468, 3))
        for k, fi in enumerate(results.face_landmarks.landmark):
            # org = (int(results.face_landmarks.landmark[4].x * image.shape[1]),
            #        int(results.face_landmarks.landmark[4].y * image.shape[0]))
            face_idx[k] = [fi.x, fi.y, fi.z]
            # draw.draw_landmarks(img1, fi)
            # print(face_idx[k])
        print(int(face_idx[4][0] * image.shape[1]), int(face_idx[4][0] * image.shape[0]))

    # print(results.face_landmarks.landmark[0]['x'])
    #     cv2.circle(image, center=((int(face_idx[4][0]) * image.shape[1]), (int(face_idx[4][1]) * image.shape[0])), radius=2, color=(0, 0, 255), lineType=cv2.LINE_AA)
    #     cv2.circle(img1, center=((int(face_idx[4][0]) * image.shape[1]), (int(face_idx[4][1]) * image.shape[0])),
    #                radius=4, color=(0, 0, 255), lineType=cv2.LINE_AA)

    # print(results.face_landmarks)
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_contours_style())


    # hands와 face mesh를 사용하면 손인식륳이 떨어짐
    # hands model, holistic face model만 사용


    # # Drawing Right hand Land Marks
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.right_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS
    # )
    #
    # # Drawing Left hand Land Marks
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.left_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS

    # 랜드마크확인해서 필터입혀 보려고 한건데 안됨
    # if results.face_landmarks is not None:
    #     for fl in results.face_landmarks:
    #         f_joint = np.zeros((486, 3))
    #         for i, fll in enumerate(fl.landmark):
    #             print(f_joint = [fll.x, fll.y, fll.z])

    if result.multi_hand_landmarks is not None:
        rps_result = []
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
                # joint = [ 0번 landmark[x ,y ,z],
                #           1번 landmark[x ,y ,z],
                #           2번 landmark[x ,y ,z],
                #
                #          21번 landmark[x ,y ,z]]
                # joint 에 인덱스를 넣어준다

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]  # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            # v 는 새로운 행렬벡터임 joint 가 아님
            # v 는 열이 하나인 series 객체임? 행과열을 갖는 행렬로 변환하기 위해 np.newaxis를 사용해서 (21, 1)로 변환
            # norm = (x^2 + y^2 + z^2)^(1/2)
            # axis=1 -> 같은 행에서만 계산해야함
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            # A dot B = |A||B|cos theta -> A dot B = cos theta -> (A dot B)/cos = theta
            # 1/cos -> arcos
            # norm 해줬기 때문에 a, b벡터의 크기는 1이다
            # 위치에 관계없이 포즈를 인식시키기 위해서 각도를 이용함
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            # 학습된 모델을 사용하여 제스쳐 추측
            data = np.array([angle], dtype=np.float32)
            ret, knn_results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(knn_results[0][0])

# # 손인식 개수, 학습된 제스쳐
# max_hands = 1
# gesture = {
#     0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
#     6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
# }
#
# # 사용할 제스쳐 rock paper scissors = rps
# rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

            # Draw gesture result
            if idx in rps_gesture.keys():
                # (y, x) ????
                # org: text 의 좌표
                org = (int(res.landmark[0].x * image.shape[1]), int(res.landmark[0].y * image.shape[0]))
                cv2.putText(image, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

                #rps_result = [{'rps': 'rock', 'org': (x, y)}]

            # mp_drawing.draw_landmarks(image, res, mp_hands.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(image, res, mp_hands.HAND_CONNECTIONS)

            # depends on pose shows different image
            if len(rps_result) >= 1:
                pic = None
                text = ''

                if rps_result[0]['rps'] == 'filter1':
                    text = 'face : ryan'
                    pic = 1
                # print(winner)
                # print(text)

                elif rps_result[0]['rps'] == 'filter2':
                    text = 'face : bart'
                    pic = 2

                elif rps_result[0]['rps'] == 'filter3':
                    text = 'face : dot'
                    pic = 3

                if pic == 1:
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                    image = overlay_transparent(image, overlay, int(face_idx[1][0] * image.shape[1]), int(face_idx[1][1] * image.shape[0]),
                                                overlay_size=(450, 450))
                elif pic == 2:
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                    image = overlay_transparent(image, overlay1, int(face_idx[1][0] * image.shape[1]), int(face_idx[1][1] * image.shape[0] - 15),
                                                overlay_size=(250, 250))


                elif pic == 3:
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                    image = overlay_transparent(image, overlay2, int(face_idx[1][0] * image.shape[1]), int(face_idx[1][1] * image.shape[0]-25),
                                                overlay_size=(170, 170))

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)
    # cv2.imshow("Facial and Hand Landmarks", img1)

    # Enter key 'q' to break the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()


# # Code to access landmarks
# for landmark in mp_holistic.HandLandmark:
# 	print(landmark, landmark.value)
#
# print(mp_holistic.HandLandmark.WRIST.value)


