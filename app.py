import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ------------------------------------------------------
# 1. LOAD THE TRAINED MODEL
# ------------------------------------------------------
# Make sure 'my_model.keras' is in the same folder as this app.py
MODEL_PATH = "my_model.keras"

@st.cache_resource  # cache so model isn't reloaded on every run
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# ------------------------------------------------------
# 2. HELPER: PREPROCESS IMAGE LIKE TRAINING
# ------------------------------------------------------
def preprocess_image(image: Image.Image, target_size=(128, 128)):
    """
    - Converts to RGB
    - Resizes to target_size
    - Converts to numpy array
    - Rescales to [0, 1]
    - Adds batch dimension
    """
    # Ensure 3 channels (RGB), even if image is grayscale
    image = image.convert("RGB")
    image = image.resize(target_size)

    img_array = np.array(image)
    img_array = img_array.astype("float32") / 255.0  # rescale like ImageDataGenerator(rescale=1./255)
    img_array = np.expand_dims(img_array, axis=0)    # shape: (1, 128, 128, 3)

    return img_array

# ------------------------------------------------------
# 3. STREAMLIT UI
# ------------------------------------------------------
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image, and the model will predict whether it is **NORMAL** or **PNEUMONIA**.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Chest X-ray", use_column_width=True)

    # Button to run prediction
    if st.button("Run Diagnosis"):
        # Preprocess
        input_data = preprocess_image(image)

        # Predict (model outputs probability for class '1' since last layer is sigmoid)
        prob = model.predict(input_data)[0][0]  # scalar like 0.83

        # Interpret result
        # NOTE: With DirectoryIterator + class_mode='binary', usual mapping is:
        #   NORMAL   -> 0
        #   PNEUMONIA -> 1
        # So higher probability means more likely PNEUMONIA.
        threshold = 0.5
        if prob >= threshold:
            diagnosis = "PNEUMONIA"
            color = "red"
        else:
            diagnosis = "NORMAL"
            color = "green"

        # Show results
        st.subheader("Model Prediction")
        st.markdown(
            f"**Prediction:** <span style='color:{color}; font-size: 24px;'>{diagnosis}</span>",
            unsafe_allow_html=True
        )
        st.write(f"**Pneumonia probability:** `{prob*100:.2f}%`")
        st.write(f"**Threshold used:** {threshold} (≥ means PNEUMONIA)")

        st.info("⚠️ This tool is for educational/demo purposes only and not a medical diagnosis.")
else:
    st.write("👉 Please upload a chest X-ray image to begin.")
