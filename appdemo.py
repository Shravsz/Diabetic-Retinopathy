import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Set Streamlit page configuration
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal image to predict the stage of Diabetic Retinopathy.")

# Load trained model
model = load_model('DR_Detection_Model.keras')
class_names = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferative DR', 'Severe DR']

# Ensure input dimensions match training
img_height, img_width = 180, 180  # Change these values if trained on a different size

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the image
    img = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch

    # Display the resized image
    st.image(img, caption="Uploaded Image", width=300, use_container_width=False)

    # Predict using the model
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display results
    st.write(f"The image likely belongs to: **{class_names[np.argmax(score)]}**")
    st.write(f"Accuracy: **{100 * np.max(score):.2f}%**")
else:
    st.write("Please upload an image.")
