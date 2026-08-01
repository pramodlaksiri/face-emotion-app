import streamlit as st
import cv2
import numpy as np
import urllib.request
import os

st.set_page_config(page_title="Facial Expression Detector", page_icon="🎭")
st.title("🎭 Live Facial Expression Detector")
st.write("ඔබගේ මුහුණේ Emotion එක (හැඟීම) සජීවීව පරීක්ෂා කරගන්න!")

# FER+ Emotion Labels
EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

@st.cache_resource
def load_models():
    # Haar Cascade face detector (OpenCV එකෙන්ම ලැබේ)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # ONNX 3MB Lightweight Emotion Model එක Download කරගැනීම
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    model_path = "emotion-ferplus-8.onnx"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
        
    emotion_net = cv2.dnn.readNetFromONNX(model_path)
    return face_cascade, emotion_net

try:
    face_cascade, emotion_net = load_models()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

img_file = st.camera_input("Camera එකෙන් Photo එකක් ගන්න")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    with st.spinner("Analyzing emotion..."):
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Face Area Crop කරගැනීම
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (64, 64))
                blob = cv2.dnn.blobFromImage(face_resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)

                # Emotion Predict කිරීම
                emotion_net.setInput(blob)
                preds = emotion_net.forward()[0]
                
                exp_preds = np.exp(preds - np.max(preds))
                probs = exp_preds / exp_preds.sum()
                
                dominant_idx = np.argmax(probs)
                dominant_emotion = EMOTIONS[dominant_idx]

                # Box එක ඇඳීම
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                emotion_dict = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}

            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Analyzed Face", use_container_width=True)
            st.success(f"**Detected Expression:** {dominant_emotion.upper()}")

            st.subheader("📊 Emotion Breakdown")
            st.bar_chart(emotion_dict)
        else:
            st.warning("මුහුණ පැහැදිලිව අඳුනාගැනීමට නොහැකි විය. කරුණාකර නැවත Photo එකක් ගන්න.")
