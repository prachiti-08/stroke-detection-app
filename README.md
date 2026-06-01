# Automated Brain CT Scan Analysis for Stroke Detection

AI-powered multiclass brain stroke detection and explainable medical imaging system using Deep Learning, ResNet18, and Grad-CAM visualization.

# Project Overview

This project is an AI-assisted clinical decision support system developed for automated brain stroke classification using CT/MRI medical images.

The system detects and classifies brain scans into:

- Normal
- Ischemic Stroke
- Hemorrhagic Stroke

The platform integrates:

- Deep Learning-based image classification
- Explainable AI heatmaps using Grad-CAM
- Interactive Streamlit dashboard
- Flask + HTML/CSS frontend integration
- PDF report generation
- Medical imaging preprocessing pipeline

# Key Features

- Multiclass Stroke Classification  
- Transfer Learning using ResNet18  
- Grad-CAM Explainability Heatmaps  
- Confidence Score Prediction  
- Medical Image Preprocessing  
- Streamlit Diagnostic Dashboard  
- Flask-based Backend Integration  
- Professional Healthcare Website UI  
- PDF Report Generation  
- Responsive Frontend Design  
- GPU-supported Training  

# Problem Statement

Stroke is one of the leading causes of death and long-term disability worldwide. Early detection is critical for effective treatment, but manual interpretation of brain scans is time-consuming and requires expert radiologists.

Many regions face:

- Shortage of radiologists
- Delayed diagnosis
- Human interpretation errors
- Lack of explainable AI systems

This project aims to develop an AI-powered medical imaging system capable of assisting clinicians through fast, explainable, and accurate stroke classification.

# Objectives

- Develop an automated AI system for stroke detection
- Classify brain scans into Normal, Ischemic, and Hemorrhagic classes
- Improve diagnostic efficiency
- Provide visual explainability using Grad-CAM
- Build a deployable healthcare AI platform
- Create a user-friendly diagnostic dashboard

# System Architecture

Medical Image Upload
          ↓
Image Preprocessing
(Resize, Normalize, Augmentation)
          ↓
Deep Learning Model
(ResNet18 + CNN)
          ↓
Prediction Engine
          ↓
Confidence Score
          ↓
Grad-CAM Heatmap Generation
          ↓
Dashboard Visualization
          ↓
PDF Report Generation

# Technologies Used

## Programming Language

- Python

## Deep Learning & AI

- PyTorch
- CNN (Convolutional Neural Network)
- ResNet18
- Transfer Learning
- Grad-CAM
- Explainable AI (XAI)

## Image Processing

- OpenCV
- NumPy
- PIL (Python Imaging Library)

## Data Handling

- Pandas
- Scikit-learn

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- Flask
- Streamlit

## Visualization

- Matplotlib
- Heatmap Visualization

## Report Generation

- FPDF

## Development Tools

- VS Code
- GitHub

# Deep Learning Model

## Model Used

### ResNet18

ResNet18 is a pretrained Convolutional Neural Network architecture used through transfer learning.

### Why ResNet18?

- Lightweight and efficient
- Excellent image feature extraction
- Reduced vanishing gradient problem
- Faster training
- Good accuracy on medical imaging tasks

## Transfer Learning

Instead of training from scratch, pretrained ImageNet weights were used to improve:

- Accuracy
- Training speed
- Generalization capability

# Explainable AI using Grad-CAM

The project integrates Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explainability.

Grad-CAM helps identify:

- Important brain regions
- Stroke-affected areas
- Model attention zones

This improves trust and interpretability in medical AI systems.

# Dataset

The model was trained using publicly available medical imaging datasets from Kaggle and healthcare imaging sources.

## Classes Used

- Normal
- Ischemic Stroke
- Hemorrhagic Stroke

# Image Preprocessing

The following preprocessing techniques were applied:

- Image resizing
- Normalization
- Tensor conversion
- Data augmentation
- Noise reduction
- Contrast enhancement

# Model Workflow

## Step 1 — Dataset Collection

Medical brain CT/MRI images were collected from public datasets.

## Step 2 — Data Preprocessing

Images were cleaned, resized, normalized, and augmented.

## Step 3 — Model Training

ResNet18 was trained using transfer learning in PyTorch.

## Step 4 — Prediction

The trained model predicts the stroke category.

## Step 5 — Explainability

Grad-CAM generates heatmaps showing affected regions.

## Step 6 — Dashboard Output

Results are displayed through Streamlit and web frontend.

## Step 7 — Report Generation

Diagnostic PDF reports are generated automatically.

# Project Structure

```text
stroke-detection-ai/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model/
│   └── stroke_model.pth
│
├── dataset/
│   ├── normal/
│   ├── ischemic/
│   └── hemorrhagic/
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── script.js
│   │
│   ├── uploads/
│   │
│   └── images/
│
└── streamlit_app.py
```

# Requirements
```text
torch
torchvision
streamlit
flask
opencv-python-headless
numpy
pandas
matplotlib
scikit-learn
pillow
fpdf
```

# Evaluation Metrics

The model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

# Applications

- Hospital diagnostic assistance
- Emergency stroke screening
- AI-assisted radiology systems
- Clinical decision support
- Medical research

# Future Scope

- Real-time hospital deployment
- Integration with PACS systems
- Cloud-based diagnosis
- Mobile healthcare integration
- Segmentation-based lesion detection
- 3D CT scan analysis
- Federated learning for medical privacy
- Integration with Electronic Health Records (EHR)

# Limitations

- Dependent on dataset quality
- Requires larger clinical datasets
- Not a replacement for radiologists
- Performance may vary across imaging devices

# Ethical Considerations

- Patient privacy must be protected
- AI should assist, not replace clinicians
- Medical AI requires explainability
- Bias mitigation is essential in healthcare datasets

# Sample Outputs

## Prediction Dashboard

- Uploaded CT image
- Predicted class
- Confidence score
- Grad-CAM heatmap

# Team Members

- Prachiti Shivalkar
- Kajal Koli
- Tirtha Mahabde

## Guide

- Prof. Aritri Sen

## Institute

Usha Mittal Institute of Technology  
SNDT Women’s University

# Research References

- [1] M. Kanchana, R. Shankar, and G. Hariharan, “Brain Stroke Detection using CT Images,” Proc.
ICACRS, IEEE, 2024.
- [2] Z. N. Izdihar, “Pretrained Deep CNN Model Evaluation for Brain Stroke Detection Using CT
Scan Data,” Proc. BTS-I2C, IEEE, 2024.
- [3] P. Venkadesh et al., “Deep Learning Based Brain Stroke Detection System using CNN and
Federated Learning,” Proc. ICCPCT, IEEE, 2025.
- [4] P. S. Khan et al., “Real-time Brain Stroke Detection using CT Scans,” Proc. AIDE, IEEE, 2025.
- [5] C. Bhole et al., “Automated AI-Driven Detection of Brain Infarct and Hemorrhage,” Proc.
ICACC Tech, IEEE, 2024.
- [6] A. Fontanella et al., “Deep learning method to identify acute ischemic stroke lesions on CT,”
arXiv:2309.17320, 2023

## Live Website:
https://stroke-detection-app.vercel.app

## Live AI Dashboard:
https://stroke-detection-app-neuro-ai.streamlit.app

# License

This project is developed for educational and research purposes.

# Contact
## Prachiti Shivalkar
- LinkedIn: https://www.linkedin.com/in/prachiti-shivalkar-ai/
- GitHub: [https://github.com/](https://github.com/prachiti-08)

# ⭐ If you found this project useful, give it a star on GitHub!
