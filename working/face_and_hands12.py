from modules import Display_Image as display_img
from modules import Warp_Image as warp_image
from modules import Read_CSV as read_csv

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

# 손인식 개수, 학습된 제스쳐
gesture = {0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
           6: 'six', 7: 'rock', 8: 'spider man', 9: 'yeah', 10: 'ok'}

# 사용할 제스쳐 rock paper scissors = rps
rps_gesture = {0: 'filter1', 5: 'filter2', 9: 'filter3'}

# MediaPipe hands model
mp_hands = mp.solutions.hands

# MediaPipe face mesh and detection models
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# MediaPipe drawing model
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition model
file = np.genfromtxt('../data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# initialize params_01 -> 제거 할 것은 제거해야함
start = time.time()
mouse_coord = []
click = [0]

# List the images' paths (PNG files must have transparent backgrounds)
mask_filenames = ['image/ryan_transparent.png', 'image/batman_2.png', 'image/iron_man_2.png', 'image/none.png']

# Pixel coordinates must match from the images to each landmark
csv_filenames = ['image/ryan_transparent.csv', 'image/batman_2.csv', 'image/iron_man_2.csv']


# Stream lit start
# roots of videos
DEMO_VIDEO = 'demos/demo.mp4'
DEMO_IMAGE = 'demos/demo.jpg'

st.title('Project')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]  # x, y 사용 -> print(img.shape) -> (x, y, z) -> indexing -> 0~1 -> tuple -> (x, y)

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


app_mode = st.sidebar.selectbox('Choose the App mode', ['Run on Video'])


if app_mode == 'Run on Video':
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')

    st.markdown(
                """
                <style>
                [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                    width: 400px;
                }
                [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                    width: 400px;
                    margin-left: -400px;
                }
                </style>
                """,
                unsafe_allow_html=True,
                )

    # left side dashboard
    # max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    max_hands = st.sidebar.number_input('Maximum Number of Hands', value=1, min_value=1)

    # options 1
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    # video upload
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    # video 파일이 type 리스트내의 형식이 아닐경우
    # use_webcam 이면 웹캠 사용
    # webcam 사용하지 않는다면 demo_video 재생
    if not video_file_buffer:
        if use_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    # 파일 형식이 지정된 형식이라면 업로드한 비디오를 사용한다
    else:
        tfflie.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tfflie.name)

    # 사용된 video 의 width, height 구하기
    # MediaPipe 좌표는 0~1 사이 정규화 -> 원하는 좌표를 얻기 위해서는 현재 frame 의 width 와 height 를 알아야한다
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    # VideoWriter_fourcc -> 찾아보기기
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('demos/output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # 동영상 밑에 있는 정보 창
    kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi4:
        st.markdown("**Detected Hands**")
        kpi4_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # import modules as face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=detection_confidence,
                                      min_tracking_confidence=tracking_confidence,
                                      max_num_faces=max_faces)

    # import modules as face detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=detection_confidence)

    # import modules as hands
    hands = mp_hands.Hands(max_num_hands=max_hands,
                           min_detection_confidence=detection_confidence,
                           min_tracking_confidence=tracking_confidence)

    # Read mask images to display on output image
    mask_width = 70
    mask_height = 100
    masks_files = []

    for file in mask_filenames:
        mask = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (mask_width, mask_height))
        mask = mask / 255.0
        masks_files.append(mask)

    # Selection of mask variables
    selected = 3
    hover = -1

    # Define image size parameters and open camera
    # black_height, black_width = 576, 768
    video = cv2.VideoCapture(0)
    ret, image = video.read()
    image = cv2.resize(image, (width, height))

    # Link the output imshow to the mouse callback function
    # 웹캠에서 마우스 인식
    face_land_img = None

    prevTime = 0
    while ret:
        frame_start = time.time()

        result = hands.process(image)  # hands


        csv_filename = csv_filenames[0 if selected == 3 else selected]
        img_filename = mask_filenames[selected]
        landmarks, ids, mask_coordinates = read_csv.readCSV(csv_filename)

        image = cv2.flip(image, 1)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        black_image = np.zeros((height, width, 3), np.uint8)
        black_image[::, ::] = (230, 230, 230)

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
            x = int(face_landmarks.landmark[landmark_of_interest].x * width)
            y = int(face_landmarks.landmark[landmark_of_interest].y * height)
            landmarks_coordinates.append([x, y])



        # Face detection
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            continue

        # Extraction of face location from image
        for detection in results.detections:
            area = detection.location_data.relative_bounding_box
            x1 = int(area.xmin * width)
            y1 = int(area.ymin * height * 0.75)
            x2 = x1 + int(area.width * width)
            y2 = int(area.ymin * height) + int(area.height * height * 1.05)

            face_land_img = black_image[y1:y2, x1:x2]
            face_land_img = cv2.resize(face_land_img, (width // 5, height // 3))



##########################################################################################
        # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # results = face_mesh.process(frame)
        #
        # frame.flags.writeable = True
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 얼굴인식
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

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

                # Draw gesture result
                if idx in rps_gesture.keys():
                    # (y, x) ????
                    # org: text 의 좌표
                    org = (int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0]))
                    cv2.putText(frame, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)

                    rps_result.append({
                        'rps': rps_gesture[idx],
                        'org': org
                    })

                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

                # depends on pose shows different image
                if len(rps_result) >= 1:
                    text = ''
                    if rps_result[0]['rps'] == 'filter1':
                        text = 'face : overlay'
                        # pic = 1
                        cv2.putText(frame,
                                    text=text,
                                    org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)

                    elif rps_result[0]['rps'] == 'filter2':
                        text = 'face : overlay1'
                        cv2.putText(frame,
                                    text=text,
                                    org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2,
                                    color=(0, 255, 0),
                                    thickness=3)

                    elif rps_result[0]['rps'] == 'filter3':
                        text = 'face : overlay2'
                        cv2.putText(frame,
                                    text=text,
                                    org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2,
                                    color=(0, 255, 0),
                                    thickness=3)




        # Call warp function to apply homography with the face orientation
        #   and mask image
        output = warp_image.warpImage(image, landmarks_coordinates, img_filename,
                                      mask_coordinates, selected)

        # Combine results in a single image for output
        frame, positions = display_img.displayImage(output, face_land_img, masks_files, selected, hover)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # record
        if record:
            # st.checkbox("Recording", value=True)
            out.write(frame)

        # Dashboard
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        kpi4_text.write(f"<h1 style='text-align: center; color: red;'>{hand_count}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open(tfflie, 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    cap.release()
    out.release()
