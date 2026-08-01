import streamlit as st
import cv2
import numpy as np
import requests
import os

st.set_page_config(page_title="Face, Emotion & Age Analyzer", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Live Face, Emotion & Age Analyzer")
st.write("ඔබගේ මුහුණේ Emotion එක (හැඟීම) සහ Age (වයස) සජීවීව පරීක්ෂා කරගන්න!")

# FER+ Emotion Labels
EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
# Age buckets (approximate ranges)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

@st.cache_resource
def load_models():
    # 1. Download Haar Cascade Face Detector XML
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        res = requests.get(cascade_url)
        with open(cascade_path, "wb") as f:
            f.write(res.content)
            
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # 2. Download Binary ONNX Emotion Model via Direct LFS Media URL (35MB)
    model_path = "emotion-ferplus-8.onnx"
    # Clean up previous text pointer file if exists
    if os.path.exists(model_path) and os.path.getsize(model_path) < 100000:
        os.remove(model_path)
        
    if not os.path.exists(model_path):
        model_url = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(model_url, headers=headers, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    emotion_net = cv2.dnn.readNetFromONNX(model_path)
    
    # 3. Download Age Detector Models (Proto and Caffemodel)
    age_proto = "deploy_age.prototxt"
    age_model = "age_net.caffemodel"
    
    if not os.path.exists(age_proto):
        age_proto_url = "https://raw.githubusercontent.com/spandeyraw/Facial-Expression-and-Age-Detection/master/AgeNet/deploy_age.prototxt"
        res = requests.get(age_proto_url)
        with open(age_proto, "wb") as f:
            f.write(res.content)
            
    if not os.path.exists(age_model):
        age_model_url = "https://raw.githubusercontent.com/spandeyraw/Facial-Expression-and-Age-Detection/master/AgeNet/age_net.caffemodel"
        # Since caffemodel can be larger, increase chunk size and retry
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(age_model_url, headers=headers, stream=True)
        with open(age_model, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                
    age_net = cv2.dnn.readNet(age_proto, age_model)
    
    return face_cascade, emotion_net, age_net

try:
    with st.spinner("🕵️‍♂️ Analyzing Tools Loading... (විනාඩියක් පමණ ගත විය හැක)"):
        face_cascade, emotion_net, age_net = load_models()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

img_file = st.camera_input("Camera එකෙන් Photo එකක් ගන්න")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    with st.spinner("🕵️‍♂️ Deep Analyzing emotion and age..."):
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Box with extra padding for age model
                padding = 20
                face_crop_orig = cv_img[max(0,y-padding):min(cv_img.shape[0]-1,y+h+padding), max(0,x-padding):min(cv_img.shape[1]-1,x+w+padding)]
                
                # EMOTION ANALYSIS
                face_roi_gray = gray[y:y+h, x:x+w]
                face_resized_gray = cv2.resize(face_roi_gray, (64, 64))
                blob_emotion = cv2.dnn.blobFromImage(face_resized_gray, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)

                emotion_net.setInput(blob_emotion)
                preds_emotion = emotion_net.forward()[0]
                
                exp_preds = np.exp(preds_emotion - np.max(preds_emotion))
                probs_emotion = exp_preds / exp_preds.sum()
                
                dominant_idx = np.argmax(probs_emotion)
                dominant_emotion = EMOTIONS[dominant_idx]
                emotion_dict = {EMOTIONS[i]: float(probs_emotion[i]) for i in range(len(EMOTIONS))}

                # AGE ANALYSIS
                # Caffemodel normalizes with MEAN values
                MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
                blob_age = cv2.dnn.blobFromImage(face_crop_orig, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                age_net.setInput(blob_age)
                preds_age = age_net.forward()[0]
                
                dominant_age_idx = np.argmax(preds_age)
                dominant_age = AGE_BUCKETS[dominant_age_idx]

                # Box & Labels on image
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                label_text = f"E: {dominant_emotion.upper()}, A: {dominant_age}"
                # Optional: draw text on image if needed
                # cv2.putText(cv_img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Analyzed Face", use_container_width=True)
            
            st.markdown(f"## **Detection Results:**")
            st.success(f"**Expression (Emotion):** {dominant_emotion.upper()} 🎭")
            st.info(f"**Age Bracket (वయස සීමාව):** {dominant_age} 🗓️")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Emotion Breakdown")
                st.bar_chart(emotion_dict)
            with col2:
                st.subheader("📊 Age Ranges (Buckets)")
                st.bar_chart(preds_age, use_container_width=True, y_label="Buckets")
        else:
            st.warning("මුහුණ පැහැදිලිව අඳුනාගැනීමට නොහැකි විය. කරුණාකර නැවත Photo එකක් ගන්න.")
