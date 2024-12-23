import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the style transfer model
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

model = load_model()

# Function to load and preprocess images
def load_image(image_file):
    img = tf.image.convert_image_dtype(np.array(image_file), tf.float32)
    img = tf.image.resize(img, (256, 256))  # Resize to a fixed size
    img = img[tf.newaxis, ...]
    return img

# Streamlit UI
st.title("Image Style Transfer App")
st.write("Upload a content image and a style image, and this app will blend the two!")

# Image uploads
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    # Load images
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)

    # Display uploaded images
    st.write("**Content Image:**")
    st.image(content_image, use_column_width=True)

    st.write("**Style Image:**")
    st.image(style_image, use_column_width=True)

    # Preprocess images for the model
    content_tensor = load_image(content_image)
    style_tensor = load_image(style_image)

    # Perform style transfer
    stylized_tensor = model(tf.constant(content_tensor), tf.constant(style_tensor))[0]

    # Convert the tensor to an image
    stylized_image = np.squeeze(stylized_tensor.numpy())
    stylized_image = (stylized_image * 255).astype(np.uint8)

    # Display the result
    st.write("**Stylized Image:**")
    st.image(stylized_image, use_column_width=True)

# Footer
st.write("---")
st.write("Built with TensorFlow Hub and Streamlit")
