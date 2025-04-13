import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="ECG Analyzer", layout="wide")

st.title("ECG Waveform Analyzer (Beta)")
st.markdown("Upload an ECG image (JPEG) and get heart rate + rhythm estimation.")

uploaded_file = st.file_uploader("Upload ECG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Show original image
    st.image(img, caption="Original ECG", use_column_width=True)

    # Crop Lead II (manually tuned — adjust if needed)
    lead_ii = img[250:330, 120:1150]
    lead_ii = cv2.bitwise_not(lead_ii)

    # Extract waveform
    signal = np.mean(lead_ii, axis=0)
    signal = cv2.GaussianBlur(signal, (5, 5), 0)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Detect R-peaks
    peaks, _ = find_peaks(signal, distance=25, height=0.5)

    if len(peaks) > 1:
        rr_intervals = np.diff(peaks)
        avg_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        rr_sec = avg_rr * 0.04
        heart_rate = 60 / rr_sec
        rhythm = "Likely Sinus Rhythm" if std_rr < 10 and 60 <= heart_rate <= 100 else "Possibly Non-Sinus / Irregular Rhythm"

        # Show report
        st.subheader("Analysis Report")
        st.write(f"**Estimated Heart Rate:** {heart_rate:.1f} bpm")
        st.write(f"**Rhythm Interpretation:** {rhythm}")

        # Plot waveform and peaks
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(signal, label='Lead II')
        ax.plot(peaks, signal[peaks], "rx", label='R-peaks')
        ax.set_title(f"Heart Rate: {heart_rate:.1f} bpm | Rhythm: {rhythm}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough R-peaks detected. Try a clearer image or recrop.")