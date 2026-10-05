import os

import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO
from collections import Counter


# ============================================================
# ตั้งค่า
# ============================================================

MODEL_PATH = "model/best.pt"
UPLOAD_DIR = "upload"

# ขนาดภาพที่ส่งเข้า YOLO
IMAGE_SIZE = 1280

# ความหนาของ Bounding Box
BOX_THICKNESS = 3

# ขนาดตัวอักษร
FONT_SCALE = 0.65

# ความหนาตัวอักษร
FONT_THICKNESS = 2

# สี Bounding Box
# OpenCV ใช้ BGR
# สีส้ม/ทอง
BOX_COLOR = (0, 165, 255)

# สีพื้นหลัง Label
LABEL_BG_COLOR = (15, 15, 15)

# สีตัวอักษร
LABEL_TEXT_COLOR = (255, 255, 255)


os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Cloud Detection",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# โหลด Model
# ============================================================

@st.cache_resource(show_spinner="☁️ กำลังโหลดโมเดล YOLO...")
def load_model():

    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# Cloud Classes
# ============================================================

class_names = {

    0: "Cirrocumulus",

    1: "Cirrus",

    2: "Cloudless",

    3: "Cumulonimbus",

    4: "Cumulus",

    5: "group_Altocumulus_and_Cirrostratus",

    6: "group_Altostratus_Stratus_Nimbostratus"
}


# ============================================================
# ฟังก์ชันวาด Bounding Box
# ============================================================

def draw_detections(frame, results):

    """
    วาด Bounding Box และ Label ลงบนภาพ

    รูปแบบ:
        ┌──────────────────────────────┐
        │ Cumulus 0.81                │
        │                              │
        │           ☁️                 │
        │                              │
        └──────────────────────────────┘
    """

    # --------------------------------------------------------
    # ตรวจสอบว่ามีผลลัพธ์หรือไม่
    # --------------------------------------------------------

    if results is None or len(results) == 0:

        return frame


    result = results[0]


    # --------------------------------------------------------
    # ตรวจสอบว่ามี Bounding Box หรือไม่
    # --------------------------------------------------------

    if result.boxes is None:

        return frame


    if len(result.boxes) == 0:

        return frame


    boxes = result.boxes


    # --------------------------------------------------------
    # ดึงข้อมูล Bounding Box
    # --------------------------------------------------------

    xyxy = boxes.xyxy.cpu().numpy()

    class_ids = (
        boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    confs = (
        boxes.conf
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # วาดทุก Detection
    # --------------------------------------------------------

    for box, class_id, conf in zip(
        xyxy,
        class_ids,
        confs
    ):

        # ====================================================
        # พิกัด Bounding Box
        # ====================================================

        x1, y1, x2, y2 = map(
            int,
            box
        )


        # ----------------------------------------------------
        # ป้องกันพิกัดเกินภาพ
        # ----------------------------------------------------

        frame_height, frame_width = frame.shape[:2]


        x1 = max(
            0,
            min(x1, frame_width - 1)
        )

        y1 = max(
            0,
            min(y1, frame_height - 1)
        )

        x2 = max(
            0,
            min(x2, frame_width - 1)
        )

        y2 = max(
            0,
            min(y2, frame_height - 1)
        )


        # ----------------------------------------------------
        # ถ้าพิกัดผิด
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:

            continue


        # ====================================================
        # ชื่อ Class
        # ====================================================

        if class_id in class_names:

            class_name = class_names[class_id]

        else:

            class_name = f"Class {class_id}"


        # ====================================================
        # Label
        # ====================================================

        label = f"{class_name} {conf:.2f}"


        # ====================================================
        # วาด Bounding Box
        # ====================================================

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            BOX_COLOR,

            BOX_THICKNESS,

            cv2.LINE_AA

        )


        # ====================================================
        # ขนาด Label
        # ====================================================

        font = cv2.FONT_HERSHEY_SIMPLEX


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            font,

            FONT_SCALE,

            FONT_THICKNESS

        )


        # ====================================================
        # Padding
        # ====================================================

        padding_x = 8
        padding_y = 6


        label_width = (
            text_width
            + padding_x * 2
        )

        label_height = (
            text_height
            + padding_y * 2
        )


        # ====================================================
        # ตำแหน่ง Label
        # ====================================================

        # ปกติให้ Label อยู่ด้านบนของ Bounding Box

        label_x1 = x1

        label_y2 = y1

        label_y1 = (
            y1
            - label_height
        )


        # ----------------------------------------------------
        # ถ้า Label หลุดด้านบน
        # ให้ย้ายเข้าไปใน Bounding Box
        # ----------------------------------------------------

        if label_y1 < 0:

            label_y1 = y1

            label_y2 = (
                y1
                + label_height
            )


        # ----------------------------------------------------
        # ป้องกัน Label หลุดด้านขวา
        # ----------------------------------------------------

        if (
            label_x1
            + label_width
            > frame_width
        ):

            label_x1 = (
                frame_width
                - label_width
            )


        # ----------------------------------------------------
        # ป้องกันค่าเป็นลบ
        # ----------------------------------------------------

        label_x1 = max(
            0,
            label_x1
        )

        label_y1 = max(
            0,
            label_y1
        )

        label_x2 = min(
            frame_width - 1,
            label_x1 + label_width
        )

        label_y2 = min(
            frame_height - 1,
            label_y2
        )


        # ====================================================
        # วาดพื้นหลัง Label
        # ====================================================

        cv2.rectangle(

            frame,

            (
                label_x1,
                label_y1
            ),

            (
                label_x2,
                label_y2
            ),

            LABEL_BG_COLOR,

            -1

        )


        # ====================================================
        # วาดเส้นขอบ Label
        # ====================================================

        cv2.rectangle(

            frame,

            (
                label_x1,
                label_y1
            ),

            (
                label_x2,
                label_y2
            ),

            BOX_COLOR,

            1

        )


        # ====================================================
        # ตำแหน่งข้อความ
        # ====================================================

        text_x = (
            label_x1
            + padding_x
        )


        text_y = (
            label_y1
            + padding_y
            + text_height
        )


        # ====================================================
        # วาดข้อความ
        # ====================================================

        cv2.putText(

            frame,

            label,

            (
                text_x,
                text_y
            ),

            font,

            FONT_SCALE,

            LABEL_TEXT_COLOR,

            FONT_THICKNESS,

            cv2.LINE_AA

        )


    return frame


# ============================================================
# ฟังก์ชันวิเคราะห์สภาพอากาศ
# ============================================================

def predict_weather(detected_clouds):

    """
    วิเคราะห์สภาพอากาศเบื้องต้น
    จากประเภทเมฆที่ตรวจพบ
    """

    if not detected_clouds:

        return {

            "icon": "❓",

            "condition": "ไม่สามารถทำนายได้",

            "description": (
                "ไม่พบเมฆหรือสภาพท้องฟ้า "
                "ที่ระบบสามารถตรวจจับได้"
            )

        }


    # ========================================================
    # Cumulonimbus
    # ========================================================

    if "Cumulonimbus" in detected_clouds:

        return {

            "icon": "⛈️",

            "condition": (
                "มีแนวโน้มเกิดฝนตกหนัก "
                "หรือพายุฝนฟ้าคะนอง"
            ),

            "description": (
                "ตรวจพบเมฆ Cumulonimbus "
                "ซึ่งเป็นเมฆที่มีการก่อตัวในแนวตั้งสูง "
                "และอาจสัมพันธ์กับฝนตกหนัก ลมแรง "
                "หรือพายุฝนฟ้าคะนอง"
            )

        }


    # ========================================================
    # Altostratus / Stratus / Nimbostratus
    # ========================================================

    elif (
        "group_Altostratus_Stratus_Nimbostratus"
        in detected_clouds
    ):

        return {

            "icon": "🌧️",

            "condition": (
                "ท้องฟ้าครึ้มและอาจมีฝน"
            ),

            "description": (
                "ตรวจพบกลุ่มเมฆ Altostratus, "
                "Stratus หรือ Nimbostratus "
                "ซึ่งเป็นกลุ่มเมฆที่อาจสัมพันธ์ "
                "กับสภาพอากาศครึ้มและฝนตก"
            )

        }


    # ========================================================
    # Altocumulus / Cirrostratus
    # ========================================================

    elif (
        "group_Altocumulus_and_Cirrostratus"
        in detected_clouds
    ):

        return {

            "icon": "🌥️",

            "condition": (
                "สภาพอากาศอาจมีการเปลี่ยนแปลง"
            ),

            "description": (
                "ตรวจพบกลุ่มเมฆ Altocumulus "
                "และ Cirrostratus ซึ่งอาจสัมพันธ์ "
                "กับการเปลี่ยนแปลงของสภาพอากาศ"
            )

        }


    # ========================================================
    # Cumulus
    # ========================================================

    elif "Cumulus" in detected_clouds:

        return {

            "icon": "🌤️",

            "condition": (
                "ก้อนเล็กๆ สภาพอากาศค่อนข้างดี "
                "แต่เมฆก้อนใหญ่อาจพัฒนาเป็นเมฆฝนได้"
            ),

            "description": (
                "ตรวจพบเมฆ Cumulus "
                "ซึ่งมักพบในสภาพอากาศทั่วไป "
                "แต่หากเมฆก่อตัวสูงขึ้นอาจเกิดฝนได้"
            )

        }


    # ========================================================
    # Cirrus
    # ========================================================

    elif "Cirrus" in detected_clouds:

        return {

            "icon": "☀️",

            "condition": (
                "สภาพอากาศโดยทั่วไปค่อนข้างดี"
            ),

            "description": (
                "ตรวจพบเมฆ Cirrus ซึ่งเป็นเมฆชั้นสูง "
                "มีลักษณะบางและเป็นเส้น"
            )

        }


    # ========================================================
    # Cirrocumulus
    # ========================================================

    elif "Cirrocumulus" in detected_clouds:

        return {

            "icon": "🌤️",

            "condition": (
                "สภาพอากาศโดยทั่วไปค่อนข้างดี"
            ),

            "description": (
                "ตรวจพบเมฆ Cirrocumulus "
                "ซึ่งเป็นเมฆชั้นสูง "
                "มีลักษณะเป็นก้อนเล็ก ๆ เรียงตัวกัน"
            )

        }


    # ========================================================
    # Cloudless
    # ========================================================

    elif "Cloudless" in detected_clouds:

        return {

            "icon": "☀️",

            "condition": "ท้องฟ้าแจ่มใส",

            "description": (
                "ตรวจพบสภาพท้องฟ้าที่ไม่มีเมฆปกคลุม"
            )

        }


    # ========================================================
    # Default
    # ========================================================

    else:

        return {

            "icon": "☁️",

            "condition": "มีเมฆปกคลุม",

            "description": (
                "ระบบตรวจพบข้อมูลสภาพท้องฟ้า "
                "แต่ยังไม่มีข้อมูลสำหรับประเมินเพิ่มเติม"
            )

        }


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        "## ☁️ Cloud Detection"
    )


    st.markdown(
        "ระบบตรวจจับประเภทเมฆด้วย **YOLOv8**"
    )


    st.markdown("---")


    st.markdown(
        "### ☁️ Cloud Classes"
    )


    for class_id, class_name in class_names.items():

        st.markdown(
            f"- **{class_name}**"
        )


    st.markdown("---")


    st.caption(
        f"โมเดล: `{MODEL_PATH}`"
    )


    st.caption(
        "ระบบรองรับ Image และ Video"
    )


    st.markdown("---")


    st.markdown(
        "### 🎨 Detection Style"
    )


    st.write(
        "🟧 Orange Bounding Box"
    )

    st.write(
        "⬛ Dark Label"
    )

    st.write(
        "⚪ White Text"
    )


# ============================================================
# Header
# ============================================================

st.title(
    "☁️ Cloud Detection"
)


st.caption(
    "ระบบตรวจจับ จำแนกประเภทเมฆ "
    "และประเมินสภาพอากาศเบื้องต้น"
)


st.markdown("---")


# ============================================================
# Tabs
# ============================================================

tab_image, tab_video = st.tabs(

    [
        "📷 Image Detection",
        "🎥 Video Detection"
    ]

)


# ============================================================
# IMAGE DETECTION
# ============================================================

with tab_image:

    st.subheader(
        "📷 ตรวจจับเมฆจากรูปภาพ"
    )


    # --------------------------------------------------------
    # Upload + Setting
    # --------------------------------------------------------

    col_upload, col_setting = st.columns(
        [2, 1]
    )


    # --------------------------------------------------------
    # Upload Image
    # --------------------------------------------------------

    with col_upload:

        uploaded_image = st.file_uploader(

            "📤 เลือกรูปภาพ",

            type=[
                "jpg",
                "jpeg",
                "png"
            ],

            key="image_upload"

        )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    with col_setting:

        conf_image = st.slider(

            "Confidence threshold",

            0.0,

            1.0,

            0.25,

            0.05,

            key="conf_image"

        )


    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    if uploaded_image is not None:

        image_bytes = uploaded_image.read()


        image_array = np.frombuffer(

            image_bytes,

            np.uint8

        )


        image = cv2.imdecode(

            image_array,

            cv2.IMREAD_COLOR

        )


        if image is None:

            st.error(
                "❌ ไม่สามารถอ่านรูปภาพได้"
            )


        else:

            # ------------------------------------------------
            # Original
            # ------------------------------------------------

            st.image(

                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),

                caption="Original Image",

                use_container_width=True

            )


            # ------------------------------------------------
            # Detect Button
            # ------------------------------------------------

            detect_button = st.button(

                "🔍 Detect Cloud",

                type="primary",

                use_container_width=True,

                key="detect_image"

            )


            # =================================================
            # Detection
            # =================================================

            if detect_button:

                with st.spinner(
                    "☁️ กำลังตรวจจับเมฆ..."
                ):

                    results = model.predict(

                        image,

                        conf=conf_image,

                        imgsz=IMAGE_SIZE,

                        verbose=False

                    )


                # ------------------------------------------------
                # วาดกรอบ
                # ------------------------------------------------

                result_image = draw_detections(

                    image.copy(),

                    results

                )


                # ------------------------------------------------
                # Detection Result
                # ------------------------------------------------

                st.markdown("---")


                st.subheader(
                    "🎯 Detection Result"
                )


                st.image(

                    cv2.cvtColor(

                        result_image,

                        cv2.COLOR_BGR2RGB

                    ),

                    caption="Cloud Detection Result",

                    use_container_width=True

                )


                # ------------------------------------------------
                # Boxes
                # ------------------------------------------------

                boxes = results[0].boxes


                if (
                    boxes is not None
                    and len(boxes) > 0
                ):

                    # =================================================
                    # Class IDs
                    # =================================================

                    class_ids = (

                        boxes.cls
                        .cpu()
                        .numpy()
                        .astype(int)

                    )


                    # =================================================
                    # Confidence
                    # =================================================

                    confs = (

                        boxes.conf
                        .cpu()
                        .numpy()

                    )


                    # =================================================
                    # Summary
                    # =================================================

                    st.markdown("---")


                    st.subheader(
                        "📊 Detection Summary"
                    )


                    col1, col2, col3 = st.columns(3)


                    with col1:

                        st.metric(

                            "Detected Objects",

                            len(boxes)

                        )


                    with col2:

                        st.metric(

                            "Cloud Types",

                            len(set(class_ids))

                        )


                    with col3:

                        st.metric(

                            "Average Confidence",

                            f"{np.mean(confs) * 100:.2f}%"

                        )


                    # =================================================
                    # Detected Clouds
                    # =================================================

                    st.markdown("---")


                    st.subheader(
                        "☁️ Detected Clouds"
                    )


                    detected_clouds = []


                    for class_id, conf in zip(

                        class_ids,

                        confs

                    ):

                        if class_id in class_names:

                            class_name = (
                                class_names[class_id]
                            )

                        else:

                            class_name = (
                                f"Class {class_id}"
                            )


                        detected_clouds.append(
                            class_name
                        )


                        st.write(

                            f"☁️ **{class_name}** — "
                            f"Confidence: "
                            f"**{conf * 100:.2f}%**"

                        )


                    # =================================================
                    # Count
                    # =================================================

                    cloud_counts = Counter(
                        detected_clouds
                    )


                    st.markdown("---")


                    st.subheader(
                        "📊 Cloud Class Count"
                    )


                    for cloud_name, count in (
                        cloud_counts.most_common()
                    ):

                        st.write(

                            f"☁️ **{cloud_name}** : "
                            f"**{count}** detections"

                        )


                    # =================================================
                    # Most Common
                    # =================================================

                    (
                        most_common_cloud,
                        most_common_count
                    ) = cloud_counts.most_common(1)[0]


                    st.markdown("---")


                    st.subheader(
                        "🏆 Most Detected Cloud"
                    )


                    st.info(

                        f"""
☁️ **Cloud ที่ตรวจพบมากที่สุด:**

### {most_common_cloud}

📊 **จำนวน Detection:**

### {most_common_count}
"""

                    )


                    # =================================================
                    # Weather
                    # =================================================

                    st.markdown("---")


                    st.subheader(
                        "🌦️ Weather Prediction"
                    )


                    weather = predict_weather(

                        [most_common_cloud]

                    )


                    weather_col1, weather_col2 = (
                        st.columns([1, 3])
                    )


                    with weather_col1:

                        st.markdown(
                            f"# {weather['icon']}"
                        )


                    with weather_col2:

                        st.markdown(
                            f"### {weather['condition']}"
                        )


                        st.write(
                            weather["description"]
                        )


                else:

                    st.warning(
                        "⚠️ ไม่พบเมฆในรูปภาพ"
                    )


                    st.markdown("---")


                    st.subheader(
                        "🌦️ Weather Prediction"
                    )


                    st.info(

                        "ไม่สามารถประเมินสภาพอากาศได้ "
                        "เนื่องจากไม่พบเมฆที่ระบบสามารถตรวจจับได้"

                    )


# ============================================================
# VIDEO DETECTION
# ============================================================

with tab_video:

    st.subheader(
        "🎥 ตรวจจับเมฆจากวิดีโอ"
    )


    # --------------------------------------------------------
    # Upload + Settings
    # --------------------------------------------------------

    col_upload, col_setting = st.columns(
        [2, 1]
    )


    # --------------------------------------------------------
    # Upload Video
    # --------------------------------------------------------

    with col_upload:

        uploaded_video = st.file_uploader(

            "📤 เลือกไฟล์วิดีโอ",

            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],

            key="video_upload"

        )


    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    with col_setting:

        conf_video = st.slider(

            "Confidence threshold",

            0.0,

            1.0,

            0.25,

            0.05,

            key="conf_video"

        )


        skip_frame = st.checkbox(

            "ข้ามเฟรมเพื่อเพิ่มความเร็ว",

            value=True

        )


    # --------------------------------------------------------
    # Video Uploaded
    # --------------------------------------------------------

    if uploaded_video is not None:

        save_path = os.path.join(

            UPLOAD_DIR,

            uploaded_video.name

        )


        # ----------------------------------------------------
        # Save Video
        # ----------------------------------------------------

        with open(

            save_path,

            "wb"

        ) as f:

            f.write(

                uploaded_video.getbuffer()

            )


        st.success(

            f"บันทึกไฟล์ไว้ที่ `{save_path}`"

        )


        # ----------------------------------------------------
        # Original Video
        # ----------------------------------------------------

        st.video(
            uploaded_video
        )


        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        start_video = st.button(

            "▶️ Start Cloud Detection",

            type="primary",

            use_container_width=True,

            key="start_video"

        )


        # ====================================================
        # Video Processing
        # ====================================================

        if start_video:

            cap = cv2.VideoCapture(

                save_path

            )


            # ------------------------------------------------
            # ตรวจสอบ Video
            # ------------------------------------------------

            if not cap.isOpened():

                st.error(
                    "❌ ไม่สามารถเปิดไฟล์วิดีโอได้"
                )

                st.stop()


            # ------------------------------------------------
            # Streamlit Placeholders
            # ------------------------------------------------

            frame_placeholder = st.empty()

            progress_bar = st.progress(0)

            status_text = st.empty()


            # ------------------------------------------------
            # จำนวน Frame
            # ------------------------------------------------

            total_frames = int(

                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )

            )


            if total_frames <= 0:

                total_frames = 1


            # ------------------------------------------------
            # Counter
            # ------------------------------------------------

            frame_count = 0


            # ------------------------------------------------
            # Detection Count
            # ------------------------------------------------

            cloud_count = {

                class_name: 0

                for class_name
                in class_names.values()

            }


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            cloud_confidence = {

                class_name: []

                for class_name
                in class_names.values()

            }


            # =================================================
            # Video Loop
            # =================================================

            while cap.isOpened():

                ret, frame = cap.read()


                if not ret:

                    break


                frame_count += 1


                # ------------------------------------------------
                # Skip Frame
                # ------------------------------------------------

                if (

                    skip_frame
                    and frame_count % 2 != 0

                ):

                    continue


                # ------------------------------------------------
                # Resize
                # ------------------------------------------------

                frame = cv2.resize(

                    frame,

                    (1020, 600),

                    interpolation=cv2.INTER_AREA

                )


                # ------------------------------------------------
                # YOLO
                # ------------------------------------------------

                results = model.predict(

                    frame,

                    conf=conf_video,

                    imgsz=IMAGE_SIZE,

                    verbose=False

                )


                # =================================================
                # Get Boxes
                # =================================================

                boxes = results[0].boxes


                if (
                    boxes is not None
                    and len(boxes) > 0
                ):

                    class_ids = (

                        boxes.cls
                        .cpu()
                        .numpy()
                        .astype(int)

                    )


                    confs = (

                        boxes.conf
                        .cpu()
                        .numpy()

                    )


                    # ------------------------------------------------
                    # เก็บข้อมูล
                    # ------------------------------------------------

                    for class_id, conf in zip(

                        class_ids,

                        confs

                    ):

                        if class_id not in class_names:

                            continue


                        class_name = (
                            class_names[class_id]
                        )


                        cloud_count[
                            class_name
                        ] += 1


                        cloud_confidence[
                            class_name
                        ].append(

                            float(conf)

                        )


                # =================================================
                # วาด Bounding Box
                # =================================================

                frame = draw_detections(

                    frame,

                    results

                )


                # =================================================
                # RGB
                # =================================================

                frame_rgb = cv2.cvtColor(

                    frame,

                    cv2.COLOR_BGR2RGB

                )


                # =================================================
                # Display
                # =================================================

                frame_placeholder.image(

                    frame_rgb,

                    channels="RGB",

                    use_container_width=True

                )


                # =================================================
                # Progress
                # =================================================

                progress = min(

                    frame_count
                    / total_frames,

                    1.0

                )


                progress_bar.progress(

                    progress

                )


                status_text.caption(

                    f"กำลังประมวลผล "
                    f"{frame_count} / "
                    f"{total_frames} frames"

                )


            # =================================================
            # ปิด Video
            # =================================================

            cap.release()


            status_text.empty()

            progress_bar.empty()


            st.success(
                "✅ ตรวจจับวิดีโอเสร็จสิ้น"
            )


            # ====================================================
            # Video Summary
            # ====================================================

            st.markdown("---")


            st.subheader(
                "📊 Video Detection Summary"
            )


            detected_clouds = {

                name: count

                for name, count
                in cloud_count.items()

                if count > 0

            }


            # ====================================================
            # มี Detection
            # ====================================================

            if detected_clouds:

                # ------------------------------------------------
                # จำนวนแต่ละ Class
                # ------------------------------------------------

                for name, count in (
                    detected_clouds.items()
                ):

                    st.write(

                        f"☁️ **{name}** : "
                        f"**{count}** detections"

                    )


                # ------------------------------------------------
                # Chart
                # ------------------------------------------------

                st.markdown("---")


                st.subheader(
                    "📊 Detection Chart"
                )


                st.bar_chart(
                    detected_clouds
                )


                # =================================================
                # Most Common
                # =================================================

                (
                    most_common_cloud,
                    most_common_count
                ) = max(

                    detected_clouds.items(),

                    key=lambda x: x[1]

                )


                # =================================================
                # Weather
                # =================================================

                st.markdown("---")


                st.subheader(
                    "🌦️ Weather Prediction"
                )


                weather = predict_weather(

                    [most_common_cloud]

                )


                weather_col1, weather_col2 = (
                    st.columns([1, 3])
                )


                with weather_col1:

                    st.markdown(
                        f"# {weather['icon']}"
                    )


                with weather_col2:

                    st.markdown(
                        f"### {weather['condition']}"
                    )


                    st.write(
                        weather["description"]
                    )


                # =================================================
                # Most Detected
                # =================================================

                st.markdown("---")


                st.info(

                    f"""
☁️ **Cloud ที่ตรวจพบมากที่สุด:**

### {most_common_cloud}

📊 **จำนวน Detection:**

### {most_common_count}
"""

                )


            # =================================================
            # ไม่พบ Detection
            # =================================================

            else:

                st.warning(
                    "⚠️ ไม่พบเมฆในวิดีโอ"
                )


                st.markdown("---")


                st.subheader(
                    "🌦️ Weather Prediction"
                )


                st.info(

                    "ไม่สามารถประเมินสภาพอากาศได้ "
                    "เนื่องจากไม่พบเมฆที่ระบบสามารถตรวจจับได้"

                )