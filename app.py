import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import difflib

# Set page config
st.set_page_config(
    page_title="Fake Logo Detector",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size:40px;
            font-weight:bold;
            text-align:center;
            color:#FF4B4B;
            margin-bottom: 0;
        }
        .sub {
            text-align:center;
            font-size:20px;
            margin-top: 0;
            margin-bottom: 1rem;
        }
        .result {
            font-size:24px;
            font-weight:bold;
            color:green;
        }
        .fake {
            font-size:24px;
            font-weight:bold;
            color:red;
        }
        .brand-pred {
            background-color: #f9f9f9;
            padding: 1rem;
            border-radius: 12px;
            margin-top: 10px;
        }
        .sidebar .sidebar-content {
            background-image: url("https://cdn-icons-png.flaticon.com/512/906/906334.png");
            background-repeat: no-repeat;
            background-position: top center;
            background-size: 120px 120px;
            padding-top: 140px;
        }
    </style>
""", unsafe_allow_html=True)

# Cache model loading for faster reloads
@st.cache_resource
def load_model_cached():
    return load_model('best_model3.h5')


model = load_model_cached()

# Brand classes list
classes = [
    'Adidas', 'Amazon', 'Android', 'Apple', 'Ariel', 'Bic', 'BMW', 'Burger King', 'Cadbury', 'Chevrolet',
    'Chrome', 'Coca Cola', 'Cowbell', 'Dominos', 'Fila', 'Gillette', 'Google', 'Goya oil', 'Guinness', 'Heinz',
    'Honda', 'Hp', 'Huawei', 'Instagram', 'Kfc', 'Krisspy Kreme', 'Lays', "Levi's", 'Lg', 'Lipton', 'M&m', 'Mars',
    'Marvel', 'McDonald', 'Mercedes Benz', 'Microsoft', 'Mtn', 'Mtn dew', 'NASA', 'Nescafe', 'Nestle', 'Nestle milo',
    'Netflix', 'Nike', 'Nutella', 'Oral b', 'Oreo', 'Pay pal', 'Peak milk', 'Pepsi', 'PlayStation', 'Pringles',
    'Puma', 'Reebok', 'Rolex', 'Samsung', 'Sprite', 'Starbucks', 'Tesla', 'Tiktok', 'Twitter', 'YouTube', 'Zara'
]

# Sidebar for upload and input
st.sidebar.header("Upload Logo")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
brand_name = st.sidebar.text_input("Expected Brand Name")

# Header
st.markdown('<div class="title">🕵️‍♂️ Fake Logo Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload a logo image and enter the expected brand name.</div>', unsafe_allow_html=True)
st.markdown("---")

def is_close_match(predicted, user_input, threshold=0.7):
    seq = difflib.SequenceMatcher(None, predicted.lower(), user_input.lower())
    return seq.ratio() >= threshold

def predict_logo(uploaded_file, user_brand_name):
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        max_prob = np.max(preds)
        predicted_class_idx = np.argmax(preds)
        predicted_class_name = classes[predicted_class_idx]

        if max_prob < 0.5:
            return "Fake", predicted_class_name, max_prob
        else:
            matches = difflib.get_close_matches(user_brand_name.lower(), [predicted_class_name.lower()], n=1, cutoff=0.7)
            match = len(matches) > 0

            return ("Real" if match else "Fake"), predicted_class_name, max_prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# MAIN UI LOGIC
if uploaded_file:
    if brand_name.strip() == "":
        st.warning("⚠️ Please enter the expected brand name.")
    else:
        st.image(uploaded_file, caption="Uploaded Logo", use_container_width=True)

        if st.button("🔍 Check Logo"):
            with st.spinner("Analyzing..."):
                result, predicted_brand, confidence = predict_logo(uploaded_file, brand_name)

            if result is None:
                # Prediction error already handled inside predict_logo
                pass
            else:
                if confidence < 0.5:
                    st.warning("⚠️ Model confidence is low. The logo might be fake or unclear.")

                if result == "Real":
                    st.markdown(f'<div class="result">✅ Logo is REAL</div>', unsafe_allow_html=True)
                    brand_color = "green"
                    confidence_color = "blue"
                else:
                    st.markdown(f'<div class="fake">❌ Logo is FAKE</div>', unsafe_allow_html=True)
                    brand_color = "red"
                    confidence_color = "red"

                if result == "Fake" and confidence >= 0.5:
                    st.error(f"Expected brand '{brand_name}' does not match predicted '{predicted_brand}'.")

                st.markdown(f"""
                <div class="brand-pred">
                    <b>Predicted Brand:</b> <span style="color:{brand_color}">{predicted_brand}</span> <br>
                    <b>Confidence:</b> <span style="color:{confidence_color}">{confidence:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
