#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import streamlit as st
import numpy as np
from keras.models import load_model

#Load the trained mask detection model
model = load_model("mask6_detector.model.h5",compile=False)

def try_open_webcam():
    cap = cv2.VideoCapture(0)
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
        image1 = cv2.imdecode(file_bytes, 1)
        #st.image(image, channels="BGR", caption="Uploaded Image")
        # Run prediction
    prediction = model.predict(image)

    # Determine the label
    if prediction[0][0] > 0.5:
        label = "Mask"
    else:
        label = "No Mask"

    # Display the result
    st.image(image1, channels="BGR", caption=f"Prediction: {label}")



# In[ ]:




