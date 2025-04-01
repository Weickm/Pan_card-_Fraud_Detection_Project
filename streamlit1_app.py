import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load and preprocess images
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (400, 250))
    return image

# Compare two images using SSIM
def compare_images(image1, image2):
    score, diff = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

# Detect and highlight tampered regions
def highlight_tampered_sections(reference, test, diff):
    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    tampered_areas = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Reduced area threshold for better detection
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            tampered_areas.append((x, y, w, h))
    
    return result_image, tampered_areas

# Streamlit UI
st.title("PAN Card Tampering Detection App")
st.write("Upload an original PAN card image and test images to check for tampering section-wise.")

# Upload Reference PAN Card
reference_file = st.file_uploader("Upload Original PAN Card", type=["png", "jpg", "jpeg"])
test_files = st.file_uploader("Upload Test PAN Cards", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if reference_file and test_files:
    reference_image = preprocess_image(reference_file)
    results = []
    
    for test_file in test_files:
        test_image = preprocess_image(test_file)
        score, diff = compare_images(reference_image, test_image)
        result_image, tampered_areas = highlight_tampered_sections(reference_image, test_image, diff)

        # Analyzing tampered sections
        tampered_sections = []
        for (x, y, w, h) in tampered_areas:
            if 30 < x < 200 and 10 < y < 70:
                tampered_sections.append("❌ PAN Number is tampered.")
            elif 210 < x < 370 and 10 < y < 70:
                tampered_sections.append("❌ Candidate Name is tampered.")
            elif 30 < x < 200 and 80 < y < 130:
                tampered_sections.append("❌ Father's/Mother's Name is tampered.")
            elif 30 < x < 160 and 140 < y < 210:
                tampered_sections.append("❌ Photo is tampered.")
            elif 260 < x < 390 and 140 < y < 210:
                tampered_sections.append("❌ QR Code is tampered.")
            elif 130 < x < 270 and 210 < y < 250:
                tampered_sections.append("❌ Signature is tampered.")

        overall_result = "✅ Valid PAN Card" if score >= 0.7 and not tampered_sections else "❌ Fake PAN Card"
        results.append((test_file.name, result_image, score, overall_result, tampered_sections))
    
    for test_name, result_image, score, overall_result, tampered_sections in results:
        st.image(result_image, caption=f"🔍 Tampered Sections Highlighted ({test_name})", use_column_width=True)
        st.write(f"📊 *SSIM Score:* {score:.4f}")
        st.write(f"🔍 *Overall Result:* {overall_result}")
        
        if tampered_sections:
            st.write("📌 *Tampered Sections:* ")
            for section in tampered_sections:
                st.write(section)
        else:
            st.write("✅ No tampering detected.")
        
        st.write("---")
