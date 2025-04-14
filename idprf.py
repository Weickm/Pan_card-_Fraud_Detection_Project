import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Preprocess uploaded images
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.resize(image, (400, 250))
    return image

# Compute SSIM difference
def compare_images(image1, image2):
    score, diff = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

# Highlight tampered areas using contours
def highlight_tampered_sections(reference, test, diff):
    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    tampered_areas = []

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            tampered_areas.append((x, y, w, h))

    return result_image, thresh, tampered_areas

# PAN field regions
regions = {
    "PAN Number": (80, 35, 230, 60),
    "Candidate Name": (35, 75, 200, 100),
    "Father's/Mother's Name": (35, 110, 200, 135),
    "Date of Birth": (35, 150, 200, 175),
    "Photo": (10, 10, 75, 95),
    "QR Code": (280, 30, 390, 130),
    "Signature": (190, 190, 310, 230),
    "Reference Number": (260, 215, 390, 250)
}

# Streamlit app
st.set_page_config(page_title="PAN Card Tampering Detector", layout="centered")
st.title("🔍 PAN Card Tampering Detection App")
st.write("Upload the original and suspected PAN card images to detect tampered areas.")

reference_file = st.file_uploader("Upload Original PAN Card", type=["png", "jpg", "jpeg"])
test_files = st.file_uploader("Upload Suspected PAN Card(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if reference_file and test_files:
    ref_img = preprocess_image(reference_file)

    for test_file in test_files:
        st.subheader(f"Results for: {test_file.name}")
        test_img = preprocess_image(test_file)

        score, diff = compare_images(ref_img, test_img)
        result_img, thresh_img, tampered_areas = highlight_tampered_sections(ref_img, test_img, diff)

        # Use contour-based region mapping
        detected_sections = set()
        for (x, y, w, h) in tampered_areas:
            cx, cy = x + w // 2, y + h // 2
            for label, (x1, y1, x2, y2) in regions.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    detected_sections.add(f"❌ {label} is tampered.")

        result = "✅ Valid PAN Card" if not detected_sections else "❌ Fake PAN Card"

        # Show images
        col1, col2 = st.columns(2)
        with col1:
            st.image(ref_img, caption="Original PAN Card", channels="GRAY", use_container_width=True)
        with col2:
            st.image(test_img, caption="Uploaded Test PAN Card", channels="GRAY", use_container_width=True)

        st.image(diff, caption="SSIM Difference Image", channels="GRAY", use_container_width=True)
        st.image(thresh_img, caption="Thresholded Tampering Image", channels="GRAY", use_container_width=True)
        st.image(result_img, caption="Tampered Area Highlighted", use_container_width=True)

        # Final output
        st.write(f"*Global SSIM Score:* {score:.4f}")
        st.markdown(f"*Result:* {result}")

        if detected_sections:
            st.markdown("### Tampered Sections Detected:")
            for section in sorted(detected_sections):
                st.write(section)
        else:
            st.write("✅ No tampered fields detected.")

        st.markdown("---")
