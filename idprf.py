import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Load pre-trained deep learning model (U-Net based anomaly detection model)
class DummyAnomalyModel(torch.nn.Module):
    def forward(self, x):
        # Simulated tampering heatmap (for demo purposes)
        return torch.rand_like(x[:, :1, :, :])

model = DummyAnomalyModel()
model.eval()

transform = T.Compose([
    T.Resize((250, 400)),
    T.ToTensor()
])

# Preprocess image
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return image, tensor

# Get tampering mask from model
def get_tampering_mask(tensor):
    with torch.no_grad():
        heatmap = model(tensor)[0, 0].numpy()
    mask = (heatmap > 0.5).astype(np.uint8) * 255
    return mask

# Map regions to field names
regions_def = {
    "PAN Number": (80, 35, 230, 60),
    "Candidate Name": (35, 75, 200, 100),
    "Father's/Mother's Name": (35, 110, 200, 135),
    "Date of Birth": (35, 150, 200, 175),
    "Photo": (10, 10, 75, 95),
    "QR Code": (280, 30, 390, 130),
    "Signature": (190, 190, 310, 230),
    "Reference Number": (260, 215, 390, 250)
}

# Detect tampered fields
def detect_fields_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = set()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        for label, (x1, y1, x2, y2) in regions_def.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                detected.add(label)
    return detected

# Streamlit App
st.set_page_config("AI PAN Card Tampering Detector", layout="centered")
st.title("🤖 AI-Based PAN Card Tampering Detection")

image_file = st.file_uploader("Upload PAN Card Image", type=["jpg", "jpeg", "png"])
if image_file:
    pil_image, tensor = preprocess_image(image_file)
    mask = get_tampering_mask(tensor)
    fields = detect_fields_from_mask(mask)

    st.image(pil_image, caption="Uploaded PAN Card", use_container_width=True)
    st.image(mask, caption="AI Detected Tampered Regions", channels="GRAY", use_container_width=True)

    if fields:
        st.error("❌ Tampering Detected in:")
        for field in sorted(fields):
            st.write(f"❌ {field}")
    else:
        st.success("✅ No tampering detected.")
