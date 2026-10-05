import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="About Model",
    page_icon="ℹ️",
    layout="wide"
)


# ============================================================
# Header
# ============================================================

st.title("ℹ️ About Model")

st.write(
    "ข้อมูลเกี่ยวกับโมเดลสำหรับตรวจจับและจำแนกประเภทเมฆ"
)

st.markdown("---")


# ============================================================
# Model Information
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("🤖 Model Information")

    st.write("**Model:** YOLOv8s")
    st.write("**Task:** Object Detection")
    st.write("**Image Size:** 800 × 800")
    st.write("**Epoch:** 200")
    st.write("**Batch Size:** 4")
    st.write("**Optimizer:** AdamW")
    st.write("**Learning Rate:** 0.001")
    st.write("**Classes:** 7")


# ============================================================
# Cloud Classes
# ============================================================

with col2:

    st.subheader("☁️ Cloud Classes")

    st.write("""
    - **Cirrocumulus**
    - **Cirrus**
    - **Cloudless**
    - **Cumulonimbus**
    - **Cumulus**
    - **group_Altocumulus_and_Cirrostratus**
    - **group_Altostratus_Stratus_Nimbostratus**
    """)


st.markdown("---")


# ============================================================
# Data Augmentation
# ============================================================

st.subheader("🔄 Hyperparameter Tuning File")

col1, col2, col3 = st.columns(3)


with col1:

    st.write("**Mosaic:** 1.0")
    st.write("**Mixup:** 0.2")
    st.write("**Degrees:** 15°")


with col2:

    st.write("**Scale:** 0.5")
    st.write("**Horizontal Flip:** 0.5")
    st.write("**HSV Hue:** 0.015")


with col3:

    st.write("**HSV Saturation:** 0.7")
    st.write("**HSV Value:** 0.4")
    st.write("**Seed:** 42")


# ============================================================
# Description
# ============================================================

st.markdown("---")

st.subheader("📌 About the System")

st.write(
    """
    ระบบนี้ใช้โมเดล YOLOv8s สำหรับตรวจจับและจำแนก
    ประเภทเมฆจากภาพและวิดีโอ โดยสามารถตรวจจับเมฆได้
    ทั้งหมด 7 ประเภท ได้แก่ Cirrocumulus, Cirrus,
    Cloudless, Cumulonimbus, Cumulus,
    group_Altocumulus_and_Cirrostratus และ
    group_Altostratus_Stratus_Nimbostratus
    """
)


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "☁️ Cloud Detection System | Applied Machine Learning"
)

