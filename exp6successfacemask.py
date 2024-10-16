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
        label = "Mask"
        color = (0, 255, 0)
    else:
        label = "No mask"
        color=(255, 0, 0)
    # Overlay the label on the image using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Define position for the text (top-left corner with some padding)
    text_offset_x = 10
    text_offset_y = text_height + 10

    # Draw a filled rectangle behind the text for better visibility
    cv2.rectangle(image, (text_offset_x, text_offset_y - text_height - 10),
                  (text_offset_x + text_width, text_offset_y + baseline), (0, 0, 0), thickness=cv2.FILLED)

    # Put the text (label) on the image
    cv2.putText(image, label, (text_offset_x, text_offset_y), font, font_scale, color, thickness)

    # Display the result
    st.image(image, channels="BGR", caption="Prediction with label overlay")

    # Display the result
    #st.image(image, channels="BGR", caption=f"Prediction: {label}")



# In[ ]:




