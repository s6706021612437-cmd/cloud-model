import streamlit as st

home_page = st.Page(
    "pages/home.py",
    title="Home",
    icon="🏠"
)

detection_page = st.Page(
    "pages/object_detection_app.py",
    title="Detection",
    icon="🔍"
)

cloud_information_page = st.Page(
    "pages/cloud_information.py",
    title="Cloud Information",
    icon="📖"
)

about_model_page = st.Page(
    "pages/about_model.py",
    title="About Model",
    icon="ℹ️"
)


# =====================================================
# Navigation
# =====================================================

pg = st.navigation(
    {
        "☁️ Cloud Detection": [
            home_page,
            detection_page,
            cloud_information_page,
            about_model_page
        ]
    }
)


# =====================================================
# Run
# =====================================================

pg.run()