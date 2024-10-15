#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import streamlit as st
import numpy as np
from keras.models import load_model

import cv2
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# Load your face mask detection model
model = load_model("mask6_detector.model.h5", compile=False)

# Define image size as per your model's input size
IMG_SIZE = (224, 224)  # Assuming your model expects 224x224 input images

st.title("Real-Time Face Mask Detection")

# Option to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Try to open webcam
cap = cv2.VideoCapture(0)

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

# If the webcam is not accessible, use the uploaded image instead
if uploaded_file is not None:
    # Read and preprocess the uploaded image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure it's RGB format
    image_resized = image.resize(IMG_SIZE)
    
    # Convert the image to an array and preprocess
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Run prediction on the uploaded image
    prediction = model.predict(image_array)

    if prediction[0][0] > 0.5:
        label = "Mask"
    else:
        label = "No Mask"

    # Display uploaded image with label
    st.image(image_resized, caption=f"Prediction: {label}")



# In[ ]:




