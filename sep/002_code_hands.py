import mediapipe as mp
import cv2
import time


# MediaPipe hands model
mp_hands = mp.solutions.hands

# MediaPipe drawing model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Use Webcam, Get width and height
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# options
detection_confidence = 0.5
tracking_confidence = 0.5
max_hands = 2
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)


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
    result = hands.process(frame)       # hands

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

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