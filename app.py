import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from train_model import StrokeNet
import numpy as np
import cv2
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Brain Stroke Detection Dashboard", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
}

.stButton button {
    border-radius: 12px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
}

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

device = next(model.parameters()).device

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------------
# Classes
# -----------------------------
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

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # -----------------------------
    # AUTO FIND LAST CONV LAYER
    # -----------------------------
    target_layer = None

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError("No Conv2d layer found in model!")

    # attach hooks
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    image_tensor = image_tensor.to(device)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0][0]   # (C, H, W)
    acts = activations[0][0]  # (C, H, W)

    # -----------------------------
    # Grad-CAM computation
    # -----------------------------
    weights = torch.mean(grads, dim=(1, 2))  # (C,)

    cam = torch.zeros(acts.shape[1:], device=device)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cv2.GaussianBlur(cam, (7, 7), 0)

    cam[cam < 0.25] = 0

    fh.remove()
    bh.remove()

    return cam

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CT Scan", type=["png","jpg","jpeg"])
alpha = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.5)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="card">
<h1>Stroke Detection Dashboard</h1>
<p>AI-powered CT scan analysis with explainable insights</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
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

    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    mask = mask / 255.0
    cam = cam * mask

    cam = cv2.GaussianBlur(cam, (9,9), 0)
    cam[cam < 0.2] = 0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig, 0.7, heatmap, 0.3, 0)

    # -----------------------------
    # Layout
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("CT Scan")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Diagnosis")

        st.metric("Prediction", prediction)
        st.metric("Confidence", f"{confidence}%")

        if raw_pred != "normal":
            st.warning("Possible abnormality detected")
        else:
            st.success("Normal scan")

        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Probability Distribution")

        df = pd.DataFrame({
            'Class': classes,
            'Probability': probs.squeeze().tolist()
        })

        st.bar_chart(df.set_index('Class'))
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload a CT scan to begin")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr style="border: 0.5px solid #334155;">
<p style="text-align:center; font-size:12px; color:#94a3b8;">
©NeuroAi | Educational Use Only
</p>
""", unsafe_allow_html=True)