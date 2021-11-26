import mediapipe as mp
import cv2
import numpy as np
import time


# 손인식 개수, 학습된 제스쳐
gesture = {0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
           7: 'rock', 8: 'spider man', 9: 'yeah', 10: 'ok'}

# MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe hands model
mp_hands = mp.solutions.hands

# MediaPipe drawing model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Gesture recognition model
file = np.genfromtxt('../data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# Use Webcam, Get width and height
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# options
detection_confidence = 0.5
tracking_confidence = 0.5
max_faces = 3
max_hands = 3
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame)  # face
    result = hands.process(frame)  # hands

    # Draw the face mesh annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_contours_style())
    if result.multi_hand_landmarks:
        rps_result = []
        for res in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      res,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
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



            # depends on pose shows different image
            if len(rps_result) >= 1:
                text = ''
                if rps_result[0]['rps'] == 'fist':
                    # text = 'face : overlay'
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


                elif rps_result[0]['rps'] == 'five':
                    # text = 'face : overlay1'
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


                elif rps_result[0]['rps'] == 'yeah':
                    # text = 'face : overlay2'
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

                elif rps_result[0]['rps'] == 'ok':
                    # text = 'face : overlay3'
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


    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    # Displaying FPS on the image
    cv2.putText(frame, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Face Mesh", cv2.flip(frame, 1))

    # Enter key 'q' to break the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
