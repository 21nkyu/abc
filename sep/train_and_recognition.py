import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

# 손인식 개수, 학습된 제스쳐
gesture = {0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
           7: 'rock', 8: 'spider man', 9: 'yeah', 10: 'ok'}

# MediaPipe hands model
mp_hands = mp.solutions.hands

# MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe drawing model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Gesture recognition model
file = np.genfromtxt('../data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# overlay image
overlay = cv2.imread('image/kn002.png', cv2.IMREAD_UNCHANGED)
overlay1 = cv2.imread('samples/btss.png', cv2.IMREAD_UNCHANGED)
overlay2 = cv2.imread('image/batman_1.png', cv2.IMREAD_UNCHANGED)
overlay3 = cv2.imread('image/lens001.png', cv2.IMREAD_UNCHANGED)
overlay4 = cv2.imread('image/star001.png', cv2.IMREAD_UNCHANGED)

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

    except Exception:
        return background_img


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



# options
detection_confidence = 0.5
tracking_confidence = 0.5
max_faces = 1
max_hands = 1
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

# import modules as face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=detection_confidence,
                                  min_tracking_confidence=tracking_confidence,
                                  max_num_faces=max_faces)
# import modules as hands
hands = mp_hands.Hands(max_num_hands=max_hands,
                       min_detection_confidence=detection_confidence,
                       min_tracking_confidence=tracking_confidence)

prevTime = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    final_frame = frame.copy()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame)  # face
    result = hands.process(frame)       # hands

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    face_count = 0
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1
            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_contours_style())
    hand_count = 0
    if result.multi_hand_landmarks is not None:
        rps_result = []
        for res in result.multi_hand_landmarks:
            hand_count += 1
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]     #  Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  #  Child joint
            v = v2 - v1  # [20,3]

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            # A dot B = |A||B|cos theta -> A dot B = cos theta -> (A dot B)/cos = theta
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            # 학습된 모델을 사용하여 제스쳐 추측
            data = np.array([angle], dtype=np.float32)
            ret, knn_results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(knn_results[0][0])

            # gesture = {0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
            #            7: 'rock', 8: 'spider man', 9: 'yeah', 10: 'ok'}
            # Draw gesture result
            if idx in gesture.keys():
                # (y, x) ????
                # org: text 의 좌표
                org = (int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0]))
                cv2.putText(frame, text=gesture[idx].upper(), org=(org[0], org[1] + 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': gesture[idx],
                    'org': org
                })

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

            # depends on pose shows different image
            if len(rps_result) >= 1:
                text = ''
                if rps_result[0]['rps'] == 'fist':
                    text = 'face : overlay'
                    # pic = 1
                    cv2.putText(frame,
                                text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)

                    # final_frame = overlay_transparent(final_frame,
                    #                             overlay,
                    #                             int(face_landmarks.landmark[8].x * width),
                    #                             int(face_landmarks.landmark[8].y * height),
                    #                             overlay_size=(350, 350))
                    #
                    # frame = overlay_transparent(frame,
                    #                             overlay,
                    #                             int(face_landmarks.landmark[8].x * width),
                    #                             int(face_landmarks.landmark[8].y * height),
                    #                             overlay_size=(350, 350))
                    frame = overlay_transparent(frame,
                                                overlay,
                                                int(face_landmarks.landmark[8].x * width),
                                                int(face_landmarks.landmark[8].y * height),
                                                overlay_size=(350, 350))


                elif rps_result[0]['rps'] == 'five':
                    text = 'face : overlay1'
                    cv2.putText(frame,
                                text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2,
                                color=(0, 255, 0),
                                thickness=3)
                    # final_frame = overlay_transparent(final_frame,
                    #                             overlay1,
                    #                             int(face_landmarks.landmark[4].x * width),
                    #                             int(face_landmarks.landmark[4].y * height),
                    #                             overlay_size=(250, 250))
                    frame = overlay_transparent(frame,
                                                overlay1,
                                                int(face_landmarks.landmark[4].x * width),
                                                int(face_landmarks.landmark[4].y * height),
                                                overlay_size=(250, 250))

                elif rps_result[0]['rps'] == 'yeah':
                    text = 'face : overlay2'
                    cv2.putText(frame,
                                text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2,
                                color=(0, 255, 0),
                                thickness=3)
                    # final_frame = overlay_transparent(final_frame,
                    #                             overlay2,
                    #                             int(face_landmarks.landmark[8].x * width),
                    #                             int(face_landmarks.landmark[8].y * height),
                    #                             overlay_size=(150, 150))
                    frame = overlay_transparent(frame,
                                                overlay2,
                                                int(face_landmarks.landmark[8].x * width),
                                                int(face_landmarks.landmark[8].y * height),
                                                overlay_size=(150, 150))

                elif rps_result[0]['rps'] == 'ok':
                    text = 'face : overlay3'
                    cv2.putText(frame,
                                text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2,
                                color=(0, 255, 0),
                                thickness=3)
                    # final_frame = overlay_transparent(final_frame,
                    #                             overlay2,
                    #                             int(face_landmarks.landmark[8].x * width),
                    #                             int(face_landmarks.landmark[8].y * height),
                    #                             overlay_size=(150, 150))
                    frame = overlay_transparent(frame,
                                                overlay3,
                                                int(res.landmark[3].x * width - 80),
                                                int(res.landmark[3].y * height - 25),
                                                overlay_size=(250, 250))

                    frame = overlay_transparent(frame,
                                                overlay4, int(640/2), int(480/2), overlay_size=(300, 300))

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    # Displaying FPS on the image
    cv2.putText(frame, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", frame)
    # cv2.imshow("Facial and Hand Landmarks", img1)

    # Enter key 'q' to break the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()


# 제스쳐 트레인등의 csv 파일을 이용해서 설명




