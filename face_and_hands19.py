from modules import Display_Image as display_img
from modules import Warp_Image as warp_image
from modules import Read_CSV as read_csv

import mediapipe as mp
import numpy as np
import time
import cv2

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

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

start = time.time()
mouse_coord = []
click = [0]

# List the images' paths (PNG files must have transparent backgrounds)
mask_filenames = ['image/ryan_transparent.png', 'masks/anti_covid.png',
                  'image/iron_man_2.png', 'image/none.png']
# mask_filenames = ['data/ryan_transparent.png']
# List their corresponding landmark and pixel coordinates 마스크의 픽셀 좌표
# Landmarks correspond to the value given by MediaPipes library 미디어파이프로 얻은 좌표
# Pixel coordinates must match from the images to each landmark 둘이 일치해야한다
csv_filenames = ['image/ryan_transparent1.csv', 'masks/anti_covid1.csv',
                 'image/iron_man_2.csv']
# csv_filenames = ['data/ryan_transparent.csv']
# MediaPipe's functions to extract landmarks' coordinates and
#   face detection
# 랜드마크 좌표 추출 및 얼굴 인식
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,0,0))
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Read mask images to display on output image
# 마스크 이미지를 읽어 이미지를 출력한다
mask_width = 70
mask_height = 100
masks_files = []

for file in mask_filenames:
    mask = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask,(mask_width,mask_height))
    mask = mask / 255.0
    masks_files.append(mask)

# Selection of mask variables
selected = 3
hover = -1

# Define image size parameters and open camera
height, width = 576,768
video = cv2.VideoCapture(0)
ret, image = video.read()
image = cv2.resize(image,(width,height))

# Link the output imshow to the mouse callback function
# 웹캠에서 마우스 인식
face_land_img = None
cv2.namedWindow("Live")


while(ret):
    frame_start = time.time()
##############################################################################
    # Depending on the mask selected, its corresponding image and  선택된 마스크에 따라 대응 하는 이미지와 랜드마크 위치를 읽어온다
    #   landmark location are read
    csv_filename = csv_filenames[0 if selected == 3 else selected]
    img_filename = mask_filenames[selected]
    # csv_filename = csv_filenames[0]
    # img_filename = mask_filenames[0]
    landmarks,ids,mask_coordinates = read_csv.readCSV(csv_filename)
##############################################################################
    # Detection of landmarks
    image = cv2.flip(image,1)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    black_image = np.zeros((height,width,3), np.uint8)
    black_image[::,::] = (230,230,230)

    # 손이미지 처리하기위해 변환된 image 를 hands.process 통해 result 변수에 저장
    result = hands.process(image)  # hands

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
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
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
                cv2.putText(image, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

                # rps_result = [{'rps': 'rock', 'org': (x, y)}]

            # mp_drawing.draw_landmarks(image, res, mp_hands.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(image, res, mp_hands.HAND_CONNECTIONS)

            # depends on pose shows different image
            if len(rps_result) >= 1:
                pic = None
                text = ''

                if rps_result[0]['rps'] == 'filter1':
                    text = 'face : mask1'
                    selected = 0
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=3)
                # print(winner)
                # print(text)

                elif rps_result[0]['rps'] == 'filter2':
                    text = 'face : mask2'
                    selected = 1
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=3)

                elif rps_result[0]['rps'] == 'filter3':
                    text = 'face : mask3'
                    selected = 2
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=3)



    # Drawing the landmarks on a black image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Save the coordinates of the landmarks of interest
    landmarks_coordinates = []
    for landmark_of_interest in ids:
        x = int(face_landmarks.landmark[landmark_of_interest].x*width)
        y = int(face_landmarks.landmark[landmark_of_interest].y*height)
        landmarks_coordinates.append([x, y])

    # Face detection
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        continue

    # Extraction of face location from image
    for detection in results.detections:
        area = detection.location_data.relative_bounding_box
        x1 = int(area.xmin*width)
        y1 = int(area.ymin*height*0.75)
        x2 = x1+int(area.width*width)
        y2 = int(area.ymin*height)+int(area.height*height*1.05)

        face_land_img = black_image[y1:y2,x1:x2]
        face_land_img = cv2.resize(face_land_img,(width//5,height//3))

    # Call warp function to apply homography with the face orientation
    #   and mask image
    output = warp_image.warpImage(image,landmarks_coordinates,img_filename,
                                  mask_coordinates, selected)

    # Combine results in a single image for output
    frame,positions = display_img.displayImage(output,face_land_img,masks_files,selected,hover)


    # Display of result and read the next frame
    cv2.imshow("Live",frame)
    ret,image = video.read()
    image = cv2.resize(image,(width,height))

    # Press "q" to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # Verify performance of program
    print('\rFPS: {:7.5} Time Elapsed: {:7.5} seconds'.format(1/(time.time()-frame_start),time.time()-start), end='')

print("")
video.release()