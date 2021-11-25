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
# 손인식 개수, 학습된 제스쳐
# gesture = {0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
#            6: 'six', 7: 'rock', 8: 'spider man', 9: 'yeah', 10: 'ok'}
# 사용할 제스쳐 rock paper scissors = rps
# rps_gesture = {0: 'filter1', 5: 'filter2', 9: 'filter3'}

# MediaPipe hands model
mp_hands = mp.solutions.hands

# MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe drawing model
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
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


# roots of videos
DEMO_VIDEO = 'demos/demo.mp4'
DEMO_IMAGE = 'demos/demo.jpg'

st.title('Project')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('project')
st.sidebar.subheader('Parameters')


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


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About Project', 'Team', 'Run on Image', 'Run on Video'])

# app_mode: about Project
if app_mode == 'About Project':
    st.markdown('About Project comment no.1 학습된 제스쳐를 인식해 웹캠이나 사진에 원하는 필터(이미지를) 변환하는 프로젝트'
                '시작은 동영상에 필터를 입히는 작업을 진행 동영상에서 얼굴을 인식하고 dlib을 통해 랜드마크를 얻어 원하는 좌표에 이미지를 넣는 것을 진행 오버레이 하는 전반적인 방법에 대해 익힘'
                '이후 얼굴을 인식 후 포즈는 손으로 인식해야 하기 때문에 손을 인식하는 방법을 찾기 위해 mediapipe의 hands모델을 사용함'
                '손인식 인식 후 제스쳐를 인식 시키기 위해 mediapipe에서 학습된 좌표를 사용하고'
                'opencv에서 ml 학습을 위해 제공된 것을 이용 knn 사용해 학습시킴'
                '얼굴과 손을 함께 웹캠에서 인식을 하기 위해 hoistic 모델이나 face mesh모델을 사용해서 웹캠에서 손과 얼굴을 동시에 인식시킴'
                '제스쳐를 활용하기 위해서 학습된 모델을 손과 얼굴을 인식한 코드에 더해서 손과 얼굴을 인식함과 동시에 학습된 제스쳐를 인삭하는 단계로 발전'
                '제일 처음 시도했던 동영상에 필터 입히기를 통해 배웠던 오버레이하는 방법을 통해 제스쳐에 따라 오버레이 하는 이미지를 바꿔가며 웹캠에 출력하는 단계'
                '원하는 좌표에 이미지를 넣기 위해서 landmark를 찾아 적용시키는 방법을 사용하여 웹캠이나 동영상 혹은 사진에 제스쳐에 따라 바뀌는 이미지를 넣는 프로젝트를 마무리함'
                '더 발전된 방법은 cv2.homograpy와 cv2.warpPerspective를 사용하여 특징점을 연결하는 방법으로 좌표에 2d 이미지를 띄우는 것이 아닌 원하는 landmark의 다수의 점과 이미지를 연결시켜'
                '입체감을 더하고 얼굴이 정면이 아니더라도 이미지가 원하는 부붙을 덮게 만드는 방법을 사용할 수 있음'
                'streamlit을 통해 웹에는 못올렸지만 좌표의 이미지를 연결해서 시현가능한 정도까지 발전 시켰음'
                '원하는 이미지의 픽셀값을 찾고 랜드마크와 이어서 출력하는 방법은 전문적인 이미지 처리가 없다면 완벽 할 수 없기 때문에 현재 수준에서 이질감없이'
                '원하는 신체부위에 3d혹은 굴곡을 줘서 인식시키는 부분은 무리임')
    st.markdown("""
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
                unsafe_allow_html=True
                )

    st.video('https://www.youtube.com/watch?v=Zt_q6NOuihk')

    st.markdown('''
                comment 2 comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2
                comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2
                comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2
                comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2comment 2
                ''')

elif app_mode == 'Team':
    st.markdown('team members'
                '기호 1번 전경서'
                '기호 2번 조영은'
                '기호 3번 이인규'
                '기호 1번 전경서'
                '기호 2번 조영은'
                '기호 3번 이인규'
                '기호 1번 전경서'
                '기호 2번 조영은'
                '기호 3번 이인규'
                '기호 1번 전경서'
                '기호 2번 조영은'
                '기호 3번 이인규'
                '기호 1번 전경서'
                '기호 2번 조영은'
                '기호 3번 이인규'
                '기호 1번 전경서'
                '기호 2번 조영은'
                '기호 3번 이인규'
                )
    st.markdown("""
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
                unsafe_allow_html=True
                )
    img1 = Image.open('pji/team.jpg')
    st.image(img1, width=400, caption='Image')
    # st.image(')

    st.markdown('''
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                우리는 열심히 프로젝트를 하여 피곤에 쩔은 모습이다
                ''')




# app_mode = Video
elif app_mode == 'Run on Video':

    # st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    # record = st.sidebar.checkbox("Record Video")
    # if record:
    #     st.checkbox("Recording", value=True)

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

    # # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    # # VideoWriter_fourcc -> 찾아보기기
    # codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    # out = cv2.VideoWriter('demos/output1.mp4', codec, fps_input, (width, height))

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

    with kpi3:
        st.markdown("**Detected Hands**")
        kpi4_text = st.markdown("0")

    with kpi4:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

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
        i += 1
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

        # # record
        # if record:
        #     # st.checkbox("Recording", value=True)
        #     out.write(frame)

        # Dashboard
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{hand_count}</h1>", unsafe_allow_html=True)
        kpi4_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        # if not record: output image(frame)
        # frame = cv2.resize(final_frame, (0, 0), fx=0.8, fy=0.8)
        # frame = image_resize(image=final_frame, width=640)
        # stframe.image(final_frame, channels='BGR', use_column_width=True)
        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open(tfflie, 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    cap.release()
    # out.release()


# app_mode = Image
elif app_mode == 'Run on Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

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
    # options and info
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0

    # Image
    # face mesh, hands detection and overlay image
    # Dashboard
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=max_faces,
                                      min_detection_confidence=detection_confidence)
    results = face_mesh.process(image)
    out_image = image.copy()

    for face_landmarks in results.multi_face_landmarks:
        face_count += 1
        #print('face_landmarks:', face_landmarks)

        mp_drawing.draw_landmarks(image=out_image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)
