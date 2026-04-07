import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

# =========================
# Title
# =========================
st.title("🧠 Skin Cancer Detection")
st.write("Upload an image to check for Basal Cell Carcinoma (BCC)")

# =========================
# Load Model (LOCAL ONLY)
# =========================
import gdown
import os
from tensorflow.keras.models import load_model

@st.cache_resource
def load_my_model():
    try:
        MODEL_PATH = "skin_cancer_model.h5"

        if not os.path.exists(MODEL_PATH):
            url = "https://drive.google.com/uc?id=1mV-Lu6TBKBy2mUhEGBFnOhbd37921aF2"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

        model = load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None
model =load_my_model()

if model is None:
    st.stop()

# =========================
# Hair Removal
# =========================
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = np.array(image)
        img = cv2.resize(img, (128, 128))
        img = remove_hair(img)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)[0][0]

        st.subheader("Result")

        if prediction > 0.5:
            st.error(f"⚠️ Cancer Detected ({prediction:.2f})")
        else:
            st.success(f"✅ No Cancer Detected ({1 - prediction:.2f})")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
