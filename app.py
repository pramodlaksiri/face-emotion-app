import streamlit as st
import cv2
import numpy as np
from fer import FER

st.set_page_config(page_title="Facial Expression Detector", page_icon="🎭")
st.title("🎭 Live Facial Expression Detector")
st.write("ඔබගේ මුහුණේ Emotion එක (හැඟීම) සජීවීව පරීක්ෂා කරගන්න!")

@st.cache_resource
def load_detector():
    return FER(mtcnn=False)

detector = load_detector()

img_file = st.camera_input("Camera එකෙන් Photo එකක් ගන්න")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with st.spinner("Analyzing emotion..."):
        results = detector.detect_emotions(cv_img)
        if results:
            emotions = results[0]["emotions"]
            box = results[0]["box"]
            x, y, w, h = box

            # Draw Box
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            dominant_emotion = max(emotions, key=emotions.get)

            st.image(rgb_img, caption="Analyzed Face", use_container_width=True)
            st.success(f"**Detected Expression:** {dominant_emotion.upper()}")

            st.subheader("📊 Emotion Breakdown")
            st.bar_chart(emotions)
        else:
            st.warning("මුහුණ පැහැදිලිව අඳුනාගැනීමට නොහැකි විය. කරුණාකර නැවත Photo එකක් ගන්න.")
