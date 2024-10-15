#!/usr/bin/env python
# coding: utf-8

# In[2]:
from keras.models import load_model
import cv2
import numpy as np
import streamlit as st
# Load your face mask detection model
model = load_model("mask6_detector.model.h5",compile=False)

# Optionally define image size as per your model's input size
IMG_SIZE = (224, 224)  # Assuming your model expects 224x224 input images

def try_open_webcam(index=0):
    cv2.VideoCapture(index + cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.warning("No webcam detected. Please upload an image instead.")
        return None
    return cap

st.title("Face Mask Detection App")

use_camera = st.checkbox("Use webcam")

if use_camera:
    cap = try_open_webcam()
    if cap:
        # Capture frame from webcam
        ret, frame = cap.read()
        if ret:
            st.image(frame, channels="BGR", caption="Webcam Image")
        else:
            st.error("Failed to capture image from webcam.")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Process the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Image")



# In[ ]:




