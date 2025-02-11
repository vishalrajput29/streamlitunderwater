import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2

# Load the trained YOLO model (update the path to your model)
model = YOLO('best (1).pt')

# Streamlit UI
st.title("Object Detection App")
st.write("Upload an image to detect objects using YOLOv8")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform inference
    results = model(image)
    result = results[0]
    
    # Get the result image with bounding boxes
    result_image = result.plot()
    
    # Convert numpy array to PIL Image
    result_image_pil = Image.fromarray(result_image)
    
    # Display the result image
    st.image(result_image_pil, caption="Detected Objects", use_column_width=True)
    
    # Save the result image in memory for download
    output = io.BytesIO()
    result_image_pil.save(output, format="JPEG")
    output.seek(0)
    
    # Provide download link
    st.download_button(label="Download Processed Image", data=output, file_name="detected_image.jpg", mime="image/jpeg")
