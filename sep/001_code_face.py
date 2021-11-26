import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

# MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe drawing model
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

# Use Webcam, Get width and height
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# options
detection_confidence = 0.5
tracking_confidence = 0.5
max_faces = 1

# import modules as face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=detection_confidence,
                                  min_tracking_confidence=tracking_confidence,
                                  max_num_faces=max_faces)

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
