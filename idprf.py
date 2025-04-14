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

# Compare two images and compute SSIM & difference
def compute_ssim_diff(img1, img2):
    score, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

# Highlight differences visually using contours
def extract_tampered_regions(diff):
    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(diff)
    regions = []
    for c in contours:
        if cv2.contourArea(c) > 40:  # Ignore small noise
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            regions.append((x, y, w, h))
    return mask, regions

# Define fixed regions in PAN card (coordinates based on 400x250 resize)
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

# Compare cropped fields individually for more precise tampering detection
def analyze_regions(ref_img, test_img):
    tampered_fields = []
    for label, (x1, y1, x2, y2) in regions_def.items():
        ref_crop = ref_img[y1:y2, x1:x2]
        test_crop = test_img[y1:y2, x1:x2]
        score, _ = compute_ssim_diff(ref_crop, test_crop)
        if score < 0.90:  # Customize threshold for tampering
            tampered_fields.append((label, score))
    return tampered_fields

# Streamlit App
st.set_page_config("PAN Card Tampering Detector", layout="centered")
st.title("🛡️ PAN Card Tampering Detection Tool")
st.write("Upload the original PAN card and one or more suspect images to detect tampering.")

# Upload files
ref_file = st.file_uploader("Upload ORIGINAL PAN Card", type=["jpg", "jpeg", "png"])
test_files = st.file_uploader("Upload SUSPECT PAN Card(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if ref_file and test_files:
    ref_img = preprocess_image(ref_file)

    for test_file in test_files:
        st.subheader(f"Results for: {test_file.name}")
        test_img = preprocess_image(test_file)

        # Global SSIM and Difference
        global_score, global_diff = compute_ssim_diff(ref_img, test_img)
        mask, tampered_regions = extract_tampered_regions(global_diff)

        # Overlay mask on test image
        result_overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in tampered_regions:
            cv2.rectangle(result_overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Region-wise tampering detection
        tampered_fields = analyze_regions(ref_img, test_img)

        # Display Images
        col1, col2 = st.columns(2)
        with col1:
            st.image(ref_img, caption="Original PAN (Preprocessed)", channels="GRAY", use_container_width=True)
        with col2:
            st.image(test_img, caption="Test PAN (Preprocessed)", channels="GRAY", use_container_width=True)

        st.image(global_diff, caption="SSIM Difference Image", channels="GRAY", use_container_width=True)
        st.image(mask, caption="Detected Change Mask", channels="GRAY", use_container_width=True)
        st.image(result_overlay, caption="Detected Tampered Areas", use_container_width=True)

        # Output
        st.write(f"**Global SSIM Score:** `{global_score:.4f}`")
        if tampered_fields:
            st.error("❌ This PAN card appears to be TAMPERED.")
            st.markdown("### 🔍 Tampered Fields Detected:")
            for field, score in tampered_fields:
                st.write(f"❌ {field} — SSIM: `{score:.4f}`")
        else:
            st.success("✅ This PAN card appears to be genuine.")
        st.markdown("---")
