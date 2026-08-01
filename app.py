import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Facial Expression Detector", page_icon="🎭")
st.title("🎭 Live Facial Expression Detector")
st.write("ඔබගේ මුහුණේ Emotion එක (හැඟීම) සජීවීව පරීක්ෂා කරගන්න!")

@st.cache_resource
def load_deepface():
    from deepface import DeepFace
    return DeepFace

DeepFace = load_deepface()

img_file = st.camera_input("Camera එකෙන් Photo එකක් ගන්න")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with st.spinner("Analyzing emotion..."):
        try:
            results = DeepFace.analyze(cv_img, actions=['emotion'], enforce_detection=False)
            res = results[0] if isinstance(results, list) else results
            dominant_emotion = res['dominant_emotion']

            # Draw Box
            region = res.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            if w > 0 and h > 0:
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Display Result
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Analyzed Face", use_container_width=True)
            st.success(f"**Detected Expression:** {dominant_emotion.upper()}")

            # Show Emotion Breakdown Chart
            st.subheader("📊 Emotion Breakdown")
            st.bar_chart(res['emotion'])

        except Exception as e:
            st.error(f"Error analyzing image: {e}")
