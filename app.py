import os
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F

from torchvision import transforms

from model import StrokeNet

import numpy as np
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
# PATH CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "model",
    "best_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = StrokeNet().to(device)

    if not os.path.exists(MODEL_PATH):

        st.error(
            f"Model file not found: {MODEL_PATH}"
        )

        st.stop()


    # --------------------------------------------------------
    # IMPORTANT:
    # train_model.py saves a CHECKPOINT dictionary:
    #
    # {
    #     "model_state_dict": ...,
    #     "optimizer_state_dict": ...,
    #     "epoch": ...,
    #     "val_f1": ...,
    #     ...
    # }
    #
    # Therefore we must extract model_state_dict.
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


    # New checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    # Fallback in case a raw state_dict is uploaded
    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

# EXACT normalization used during training.
# Your train_model.py uses ImageNet statistics.

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# CLASS DEFINITIONS
# ============================================================

# EXACT order produced by ImageFolder:
#
# hemorrhagic -> 0
# ischemic    -> 1
# normal      -> 2

classes = [
    "hemorrhagic",
    "ischemic",
    "normal"
]


display_names = {

    "hemorrhagic":
        "Hemorrhagic Stroke",

    "ischemic":
        "Ischemic Stroke",

    "normal":
        "Normal"
}


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, image_tensor):

    model.eval()

    device = next(
        model.parameters()
    ).device


    gradients = []
    activations = []


    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = model.model.layer4[-1]


    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output.detach()
        )


    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0].detach()
        )


    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image_tensor = image_tensor.to(device)

    image_tensor.requires_grad_(True)


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(
        image_tensor
    )


    pred = torch.argmax(
        output,
        dim=1
    ).item()


    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    model.zero_grad()

    output[0, pred].backward()


    # --------------------------------------------------------
    # Get gradients and activations
    # --------------------------------------------------------

    grads = gradients[0][0]

    acts = activations[0][0]


    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    weights = torch.mean(
        grads,
        dim=(1, 2)
    )


    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    cam = torch.zeros(
        acts.shape[1:],
        dtype=torch.float32,
        device=device
    )


    for i, weight in enumerate(weights):

        cam += (
            weight * acts[i]
        )


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


    # Move to CPU
    cam = cam.detach().cpu().numpy()


    # Resize
    cam = cv2.resize(
        cam,
        (224, 224)
    )


    # Smooth
    cam = cv2.GaussianBlur(
        cam,
        (7, 7),
        0
    )


    # Mild threshold
    cam[cam < 0.25] = 0


    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()


    return cam


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Controls"
)


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
    0.0,
    1.0,
    0.5
)


# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="card">

<h1>🧠 Brain Stroke Detection Dashboard</h1>

<p>
AI-powered CT scan classification with
explainable Grad-CAM visualization
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.markdown("---")

    st.write(
        "**Model:** ResNet18"
    )

    st.write(
        "**Classes:** 3"
    )

    st.write(
        "**Input:** 224 × 224"
    )

    st.write(
        "**Device:** " + str(device)
    )


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
    # TRANSFORM
    # --------------------------------------------------------

    img_tensor = transform(
        image
    ).unsqueeze(0)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            img_tensor.to(device)
        )

        probs = F.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )


    # --------------------------------------------------------
    # GET CLASS
    # --------------------------------------------------------

    predicted_index = pred.item()

    raw_pred = classes[
        predicted_index
    ]


    prediction = display_names[
        raw_pred
    ]


    confidence_value = (
        confidence.item() * 100
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    cam = generate_gradcam(
        model,
        img_tensor
    )


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    orig = np.array(
        image.resize(
            (224, 224)
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


    mask = mask / 255.0


    cam = cam * mask


    # --------------------------------------------------------
    # CLEAN CAM
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
        np.uint8(
            255 * cam
        ),
        cv2.COLORMAP_JET
    )


    # OpenCV produces BGR.
    # Convert to RGB for Streamlit.

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
    # CT IMAGE
    # --------------------------------------------------------

    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "CT Scan"
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
            "Diagnosis"
        )


        st.metric(
            "Prediction",
            prediction
        )


        st.metric(
            "Confidence",
            f"{confidence_value:.2f}%"
        )


        if raw_pred != "normal":

            st.markdown(
                """
                <p style="
                    color:#f87171;
                    font-weight:600;
                ">
                ⚠️ Model indicates a possible
                abnormality. Please consult a
                qualified medical professional.
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
                ">
                ✓ Scan appears normal
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
            "Grad-CAM Heatmap"
        )

        st.image(
            overlay,
            width="stretch"
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
            "Probability Distribution"
        )


        probability_values = (
            probs.squeeze()
            .detach()
            .cpu()
            .numpy()
        )


        df = pd.DataFrame({

            "Class": [
                "Hemorrhagic",
                "Ischemic",
                "Normal"
            ],

            "Probability": probability_values

        })


        st.bar_chart(
            df.set_index(
                "Class"
            ),
            height=300
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # DETAILED PROBABILITIES
    # ========================================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "Class Probabilities"
    )


    probability_cols = st.columns(
        3
    )


    for i, class_name in enumerate(classes):

        with probability_cols[i]:

            st.metric(
                display_names[class_name],
                f"{probability_values[i] * 100:.2f}%"
            )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


else:

    # ========================================================
    # NO IMAGE
    # ========================================================

    st.info(
        "Upload a CT scan from the sidebar to begin."
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

© NeuroAi | All rights reserved •
For educational and research purposes only

</p>
""", unsafe_allow_html=True)
