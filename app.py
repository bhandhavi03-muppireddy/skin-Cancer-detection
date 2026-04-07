import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os
from tensorflow.keras.models import load_model

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

# =========================
# Title
# =========================
st.markdown("<h1 style='text-align:center;'>Skin Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image to check for BCC</p>", unsafe_allow_html=True)

# =========================
# Model Config
# =========================
MODEL_PATH = "skin_cancer_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1mMVUjC6vhE1Ot-F2lzOum00F2fB1K35o" 

# =========================
# Load Model
# =========================
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False,
                fuzzy=True   # ✅ IMPORTANT FIX
            )

    # 🔍 Check file size (to detect fake download)
    if os.path.getsize(MODEL_PATH) < 1000000:  # <1MB = WRONG FILE
        st.error("❌ Model file corrupted or not downloaded correctly")
        return None

    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None
# =========================
# Hair Removal Function
# =========================
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return result

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")  # ✅ force valid format
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = np.array(image)
        img = cv2.resize(img, (128, 128))
        img = remove_hair(img)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)[0][0]

        st.write("### Prediction Result")

        if prediction > 0.5:
            st.error(f"⚠️ Cancer Detected (Confidence: {prediction:.2f})")
        else:
            st.success(f"✅ No Cancer Detected (Confidence: {1 - prediction:.2f})")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")