import os
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

import numpy as np
import cv2
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Stroke Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "model",
    "best_model.pth"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "hemorrhagic",
    "ischemic",
    "normal"
]

DISPLAY_NAMES = {
    "hemorrhagic": "Hemorrhagic Stroke",
    "ischemic": "Ischemic Stroke",
    "normal": "Normal"
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class StrokeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze complete backbone initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Fine-tuned layers used during training
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Three classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            3
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = StrokeNet().to(device)

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Your training code saves a dictionary containing
    # "model_state_dict".
    # --------------------------------------------------------

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        # Fallback in case a raw state_dict is uploaded
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error("❌ Failed to load the trained model.")

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

# IMPORTANT:
# These are the same ImageNet normalization values used
# during training in train_model.py.

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    target_class=None
):

    model.eval()

    device = next(
        model.parameters()
    ).device

    image_tensor = image_tensor.to(device)

    # Store activations and gradients
    activations = []
    gradients = []

    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output.detach()
        )

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0].detach()
        )

    # Use final ResNet convolutional block
    target_layer = model.model.layer4[-1]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(
        image_tensor
    )

    if target_class is None:

        target_class = torch.argmax(
            output,
            dim=1
        ).item()

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    model.zero_grad()

    score = output[
        0,
        target_class
    ]

    score.backward()

    # --------------------------------------------------------
    # Get gradients + activations
    # --------------------------------------------------------

    grads = gradients[0][0]

    acts = activations[0][0]

    # Global average pooling of gradients
    weights = torch.mean(
        grads,
        dim=(1, 2)
    )

    # Weighted combination
    cam = torch.zeros(
        acts.shape[1:],
        dtype=torch.float32,
        device=device
    )

    for i, weight in enumerate(weights):

        cam += weight * acts[i]

    # ReLU
    cam = torch.relu(cam)

    # Normalize
    cam_min = cam.min()
    cam_max = cam.max()

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min + 1e-8
    )

    # Convert to NumPy
    cam = cam.detach().cpu().numpy()

    # Resize to image size
    cam = cv2.resize(
        cam,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Smooth
    cam = cv2.GaussianBlur(
        cam,
        (7, 7),
        0
    )

    # Mild threshold
    cam[cam < 0.20] = 0

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return cam


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CT Scan",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)

alpha = st.sidebar.slider(
    "Heatmap Intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)


# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="card">

<h1>🧠 Stroke Detection Dashboard</h1>

<p>
AI-powered brain CT scan analysis using
Deep Learning and Explainable AI
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN LOGIC
# ============================================================

if uploaded_file:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # CREATE TENSOR
    # --------------------------------------------------------

    img_tensor = transform(
        image
    ).unsqueeze(0)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        img_tensor_device = img_tensor.to(
            device
        )

        output = model(
            img_tensor_device
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, prediction_index = torch.max(
            probabilities,
            dim=1
        )


    predicted_index = prediction_index.item()

    raw_prediction = CLASS_NAMES[
        predicted_index
    ]

    prediction = DISPLAY_NAMES[
        raw_prediction
    ]

    confidence_percentage = (
        confidence.item() * 100
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    cam = generate_gradcam(
        model,
        img_tensor,
        target_class=predicted_index
    )


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    orig = np.array(
        image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )


    # --------------------------------------------------------
    # BRAIN MASK
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        orig,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        20,
        255,
        cv2.THRESH_BINARY
    )

    mask = mask.astype(
        np.float32
    ) / 255.0

    cam = cam * mask


    # --------------------------------------------------------
    # LIGHT CLEANING
    # --------------------------------------------------------

    cam = cv2.GaussianBlur(
        cam,
        (9, 9),
        0
    )

    cam[cam < 0.20] = 0


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    # OpenCV produces BGR.
    # Convert to RGB before displaying in Streamlit.

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        orig,
        1 - alpha,
        heatmap,
        alpha,
        0
    )


    # ========================================================
    # TOP SECTION
    # ========================================================

    col1, col2 = st.columns(
        2
    )


    # --------------------------------------------------------
    # CT SCAN
    # --------------------------------------------------------

    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🩻 CT Scan"
        )

        st.image(
            image,
            width="stretch"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🔬 Diagnosis"
        )

        st.metric(
            "Prediction",
            prediction
        )

        st.metric(
            "Confidence",
            f"{confidence_percentage:.2f}%"
        )


        if raw_prediction != "normal":

            st.markdown(
                """
                <p style="
                    color:#f87171;
                    font-weight:600;
                    font-size:16px;
                ">
                ⚠️ Model indicates a possible abnormality.
                Please consult a qualified medical professional.
                </p>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <p style="
                    color:#34d399;
                    font-weight:600;
                    font-size:16px;
                ">
                ✓ No stroke-related abnormality detected by the model.
                </p>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # BOTTOM SECTION
    # ========================================================

    col3, col4 = st.columns(
        2
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    with col3:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🔥 Grad-CAM Heatmap"
        )

        st.image(
            overlay,
            width="stretch"
        )

        st.caption(
            "Highlighted regions represent areas that "
            "contributed to the model's prediction."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    with col4:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "📊 Probability Distribution"
        )

        probability_values = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )


        probability_df = pd.DataFrame({

            "Class": [
                "Hemorrhagic",
                "Ischemic",
                "Normal"
            ],

            "Probability": (
                probability_values * 100
            )

        })


        probability_df = probability_df.set_index(
            "Class"
        )


        st.bar_chart(
            probability_df,
            height=300
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "📋 Model Information"
    )

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)


    with info_col1:

        st.metric(
            "Architecture",
            "ResNet18"
        )


    with info_col2:

        st.metric(
            "Classes",
            "3"
        )


    with info_col3:

        st.metric(
            "Input Size",
            "224 × 224"
        )


    with info_col4:

        st.metric(
            "Device",
            str(device).upper()
        )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


else:

    st.info(
        "👈 Upload a CT scan from the sidebar to begin analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<hr style="border: 0.5px solid #334155;">

<p style="
    text-align:center;
    font-size:12px;
    color:#94a3b8;
">

© NeuroAI | All rights reserved •
For educational and research purposes only

</p>
""", unsafe_allow_html=True)