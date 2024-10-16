#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load your face mask detection model
model = load_model("mask6_detector.model.h5",compile = False)
IMG_SIZE = (224, 224)  # Model input size

st.title("Face Mask Detection App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Resize and preprocess the image
    image_resized = cv2.resize(image, IMG_SIZE)
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    # Run prediction
    prediction = model.predict(image_array)
    # Determine the label
    if prediction[0][0] > 0.5:
        label = "The person wearning mask"
        color = (0, 255, 0)
    else:
        label = "The pearson is not wearing mask"
        color=(255, 0, 0)
        #color = (0, 255, 0) if label == "The person wearning mask" else (255, 0, 0)

    # Display the result
    st.image(image, channels="BGR", caption=f"Prediction: {label}")



# In[ ]:




