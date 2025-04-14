import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Preprocess image (grayscale + resize + CLAHE)
def preprocess_image(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.resize(image, (400, 250))
    return image

# Compute SSIM difference
def compute_ssim_diff(img1, img2):
    _, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return diff

# Extract tampered regions using contours
def extract_tampered_regions(diff):
    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(diff)
    regions = []
    for c in contours:
        if cv2.contourArea(c) > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            regions.append((x, y, w, h))
    return mask, regions

# Field regions
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

# Check if tampered regions overlap with known fields
def detect_tampered_fields(tampered_regions):
    tampered_fields = set()
    for (x, y, w, h) in tampered_regions:
        cx, cy = x + w // 2, y + h // 2
        for label, (x1, y1, x2, y2) in regions_def.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                tampered_fields.add(label)
    return tampered_fields

# Streamlit App
st.set_page_config("PAN Card Tampering Detector", layout="centered")
st.title("🔍 PAN Card Tampering Detection (Clean Output)")
st.write("Upload the original and one or more suspected PAN card images.")

ref_file = st.file_uploader("Upload ORIGINAL PAN Card", type=["jpg", "jpeg", "png"])
test_files = st.file_uploader("Upload SUSPECT PAN Card(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if ref_file and test_files:
    ref_img = preprocess_image(ref_file)

    for test_file in test_files:
        st.subheader(f"Results for: {test_file.name}")
        test_img = preprocess_image(test_file)

        diff = compute_ssim_diff(ref_img, test_img)
        mask, tampered_regions = extract_tampered_regions(diff)
        detected_fields = detect_tampered_fields(tampered_regions)

        # Overlay tampered areas
        overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in tampered_regions:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.image(ref_img, caption="Original PAN", channels="GRAY", use_container_width=True)
        with col2:
            st.image(test_img, caption="Suspect PAN", channels="GRAY", use_container_width=True)

        st.image(mask, caption="Tampered Region Mask", channels="GRAY", use_container_width=True)
        st.image(overlay, caption="Tampered Areas Highlighted", use_container_width=True)

        if detected_fields:
            st.error("❌ This PAN card is TAMPERED.")
            st.markdown("### Tampered Fields Detected:")
            for field in sorted(detected_fields):
                st.write(f"❌ {field}")
        else:
            st.success("✅ This PAN card appears to be genuine.")
        st.markdown("---")
