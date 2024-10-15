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

cap = cv2.VideoCapture(0)

st.title("Real-Time Face Mask Detection")

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Preprocess frame
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_array = np.expand_dims(frame_resized, axis=0) / 255.0

        # Run prediction
        prediction = model.predict(frame_array)

        if prediction[0][0] > 0.5:
            label = "Mask"
        else:
            label = "No Mask"

        # Display frame with label
        st.image(frame, channels="BGR", caption=f"Prediction: {label}")
else:
    st.error("Failed to open webcam.")


# In[ ]:




