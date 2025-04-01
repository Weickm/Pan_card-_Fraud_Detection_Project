import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load and preprocess images
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (400, 200))
    return image

# Compare two images using SSIM
def compare_images(image1, image2):
    score, diff = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

# Detect and highlight tampered regions
def highlight_tampered_sections(reference, test, diff):
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    tampered_areas = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 50:
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
    
    for test_file in test_files:
        test_image = preprocess_image(test_file)
        score, diff = compare_images(reference_image, test_image)
        result_image, tampered_areas = highlight_tampered_sections(reference_image, test_image, diff)

        # Analyzing tampered sections
        tampered_sections = []
        for (x, y, w, h) in tampered_areas:
            if 50 < x < 150 and 20 < y < 60:
                tampered_sections.append("❌ PAN Number is tampered.")
            elif 160 < x < 300 and 20 < y < 60:
                tampered_sections.append("❌ Candidate Name is tampered.")
            elif 50 < x < 150 and 70 < y < 110:
                tampered_sections.append("❌ Father's/Mother's Name is tampered.")
            elif 50 < x < 120 and 120 < y < 180:
                tampered_sections.append("❌ Photo is tampered.")
            elif 280 < x < 380 and 130 < y < 180:
                tampered_sections.append("❌ QR Code is tampered.")
            elif 150 < x < 250 and 180 < y < 200:
                tampered_sections.append("❌ Signature is tampered.")

        overall_result = "✅ Valid PAN Card" if score >= 0.7 and not tampered_sections else "❌ Fake PAN Card"
        
        st.image(result_image, caption=f"🔍 Tampered Sections Highlighted ({test_file.name})", use_column_width=True)
        st.write(f"📊 *SSIM Score:* {score:.4f}")
        st.write(f"🔍 *Overall Result:* {overall_result}")
        
        if tampered_sections:
            st.write("📌 *Tampered Sections:* ")
            for section in tampered_sections:
                st.write(section)
        else:
            st.write("✅ No tampering detected.")
        
        st.write("---")
