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
# Custom CSS (UI Upgrade)
# =========================
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #bbbbbb;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown('<p class="title">Skin Cancer Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image to check for Basal Cell Carcinoma (BCC)</p>', unsafe_allow_html=True)

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
        with st.spinner("📥 Downloading AI model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

    if os.path.getsize(MODEL_PATH) < 1000000:
        st.error("❌ Model file corrupted. Please re-upload.")
        return None

    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# ✅ LOAD MODEL HERE
model = load_my_model()
if model is None:
    st.stop()

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
# Upload Section
# =========================
uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Processing animation
        with st.spinner("🔍 Analyzing image..."):
            img = np.array(image)
            img = cv2.resize(img, (128, 128))
            img = remove_hair(img)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0][0]

        # =========================
        # Results
        # =========================
        st.write("## 🧾 Prediction Result")

        confidence = float(prediction)

        # Progress bar
        st.progress(confidence if confidence < 1 else 1)

        if prediction > 0.5:
            st.markdown(f"""
                <div class="result-box" style="background-color:#ffcccc;">
                ⚠️ Cancer Detected <br>
                Confidence: {confidence:.2f}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box" style="background-color:#ccffcc;">
                ✅ No Cancer Detected <br>
                Confidence: {1 - confidence:.2f}
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")