import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from train_model import StrokeNet
import numpy as np
import cv2
import pandas as pd
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Brain Stroke Detection Dashboard", layout="wide")

# -----------------------------
# Custom CSS (Apple-style UI)
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Card style */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
}

/* Buttons */
.stButton button {
    border-radius: 12px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
}

/* Text */
h1, h2, h3 {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StrokeNet().to(device)
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

def apply_clahe(pil_img):

    img = np.array(pil_img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8,8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

classes = ['haemorrhage', 'ishemia', 'normal']
display_names = {
    'haemorrhage': 'Hemorrhagic Stroke',
    'ishemia': 'Ischemic Stroke',
    'normal': 'Normal'
}

# -----------------------------
# Grad-CAM Function
# -----------------------------
def generate_gradcam(model, image_tensor):

    device = next(model.parameters()).device

    target_layers = [
        model.features[-1],   # layer4
    ]

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers
    )

    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)

    pred = torch.argmax(outputs, dim=1).item()

    targets = [ClassifierOutputTarget(pred)]

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )

    cam_map = grayscale_cam[0]

    # Better smoothing
    cam_map = cv2.GaussianBlur(cam_map, (11,11), 0)

    # Stronger normalization
    cam_map = (cam_map - cam_map.min()) / (
        cam_map.max() - cam_map.min() + 1e-8
    )

    # Sharper localization
    cam_map[cam_map < 0.35] = 0

    return cam_map
    
# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CT Scan", type=["png","jpg","jpeg"])
alpha = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.5)

# -----------------------------
# Title Card
# -----------------------------
st.markdown("""
<div class="card">
<h1>Stroke Detection Dashboard</h1>
<p>AI-powered CT scan analysis with explainable insights</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    image = apply_clahe(image)
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    raw_pred = classes[pred.item()]
    prediction = display_names[raw_pred]
    confidence = min(round(confidence.item()*100, 2), 95)

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    cam = generate_gradcam(model, img_tensor)

    orig = np.array(image.resize((224,224)))

    # -----------------------------
    # ✅ SIMPLE brain mask (correct way)
    # -----------------------------
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # Keep brain, remove background ONLY
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = mask / 255.0
    cam = cam * mask
    kernel = np.array([
    [-1,-1,-1],
    [-1, 9,-1],
    [-1,-1,-1]
     ])

    cam = cv2.filter2D(cam, -1, kernel)

    # -----------------------------
    # ✅ LIGHT cleaning (not aggressive)
    # -----------------------------
    cam = cv2.GaussianBlur(cam, (9,9), 0)

    # Mild threshold (don’t kill signal)
    cam[cam < 0.2] = 0

    # -----------------------------
    # Heatmap
    # -----------------------------
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Better overlay
    overlay = cv2.addWeighted(orig, 0.75, heatmap, 0.45, 0)

    # -----------------------------
    # Top Section
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("CT Scan")
        st.image(image, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Diagnosis")

        st.metric("Prediction", prediction)
        st.metric("Confidence", f"{confidence}%")

        if raw_pred != "normal":
            st.markdown('<p style="color:#f87171;font-weight:600;">⚠️ Model indicates possible abnormality. Consult a medical professional.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#34d399;font-weight:600;">Scan appears normal</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Bottom Section
    # -----------------------------
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Probability Distribution")

        df = pd.DataFrame({
            'Class': classes,
            'Probability': probs.squeeze().tolist()
        })

        st.bar_chart(df.set_index('Class'), height=300)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload a CT scan from the sidebar to begin")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr style="border: 0.5px solid #334155;">
<p style="text-align:center; font-size:12px; color:#94a3b8;">
©NeuroAi | All rights reserved • For educational use only
</p>
""", unsafe_allow_html=True)