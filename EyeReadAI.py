import streamlit as st
import cv2
import numpy as np
import os

# --- CONFIGURATION & IN-MEMORY CLASSIFIERS ---
@st.cache_resource
def load_classifiers():
    # Looks for the uploaded XML files directly in the repository directory
    face_path = "haarcascade_frontalface_default.xml"
    eye_path = "haarcascade_eye_tree_eyeglasses.xml"
    
    # Error checking to ensure files exist
    if not os.path.exists(face_path) or not os.path.exists(eye_path):
        st.error("❌ Model XML files missing from repository! Please ensure you uploaded them.")
        return None, None
        
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_classifiers()

st.set_page_config(
    page_title="Eye-Read AI | Assistive Study Tool",
    page_icon="👁️",
    layout="wide"
)

st.sidebar.title("Eye-Read AI 👁️")
st.sidebar.markdown(
    """
    **AI Computer Vision Assistive Tool** Designed for students with physical motor impairments. This application runs entirely in the cloud, bypassing system errors.
    """
)

st.title("👁️ Eye-Read: Cloud AI Reading Companion")
st.write("Take a snapshot below using your webcam to test the computer vision engine.")

col1, col2 = st.columns([2, 1.2])

with col1:
    img_file_buffer = st.camera_input("Snapshot Interface")

with col2:
    st.subheader("📊 Assistive Reading Insights")
    
    if img_file_buffer is not None and face_cascade is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
        
        face_detected = len(faces) > 0
        eyes_detected = 0
        status_message = "No face detected. Please align yourself."
        posture_status = "Unknown"
        
        for (x, y, w, h) in faces:
            cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = cv2_img[y : y + h, x : x + w]
            
            if eye_cascade is not None:
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
                eyes_detected = len(eyes)
                
                for (ex, ey, ew, eh) in eyes:
                    center = (int(ex + ew/2), int(ey + eh/2))
                    radius = int((ew + eh)/4)
                    cv2.circle(roi_color, center, radius, (0, 255, 0), 2)
            
            if w > 190:
                posture_status = "🔴 Too Close"
            elif w < 100:
                posture_status = "🟡 Too Far"
            else:
                posture_status = "🟢 Perfect Screen Distance!"
                
        if face_detected:
            if eyes_detected >= 2:
                status_message = "🟢 Reading active!"
            elif eyes_detected == 1:
                status_message = "🔵 Wink detected!"
            else:
                status_message = "🟡 Eyes closed."
        
        with col1:
            st.image(cv2_img, channels="BGR", caption="Processed Frame")
            
        st.markdown(f"### Status: {status_message}")
        st.info(f"Posture Monitor: {posture_status}")
        st.write(f"Faces: {len(faces)} | Eyes Active: {eyes_detected}")
    else:
        st.info("Turn on your web camera using the button on the left.")
