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

# Load the trained mask detection model  
mask_model = load_model("mask6_detector.model.h5", compile=False)

# Load OpenCV's pre-trained face detector model
prototxt_path = "deploy.prototxt"  # Path to your prototxt file
weights_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the caffemodel
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

# Function to detect faces and predict mask usage
def detect_and_predict_mask(frame, face_net, mask_model):
    (h, w) = frame.shape[:2]
    
    # Prepare the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    locs = []
    preds = []
    
    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Adjust the confidence threshold to detect smaller faces (default is 0.5)
        if confidence > 0.3:  # Lowered threshold to 0.3 to detect smaller faces
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box falls within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract the face ROI, preprocess it, and add to the list
            face = frame[startY:endY, startX:endX]
            
            # Check if the face is too small or too large and resize accordingly
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue  # Skip small face detections that are too small to predict

            # Resize the face to the model input size, maintaining aspect ratio
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))  # Adjust the input size as per your model
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    # Only make a mask prediction if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_model.predict(faces, batch_size=32)
    
    return (locs, preds)

# Streamlit UI
st.title("Face Mask Detection with Face Detection")

# Image upload functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read and process the uploaded image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure it's RGB format
    image_np = np.array(image)  # Convert to NumPy array
        
    # Perform face detection and mask prediction
    (locs, preds) = detect_and_predict_mask(image_np, face_net, mask_model)

    # Display the processed image with predictions
    st.image(image_np, caption='Uploaded Image with Predictions', channels='RGB')

st.write("Stopped")


# In[ ]:




