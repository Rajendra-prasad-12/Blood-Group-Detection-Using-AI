# import statements

import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from skimage.feature import local_binary_pattern
from skimage.transform import integral_image
from skimage import filters
from skimage import measure
from scipy.ndimage import label
from skimage.feature import local_binary_pattern
from skimage.morphology import thin, disk
from scipy.ndimage import convolve
import pywt

# Function to load pre-trained model
def load_model():
    # Replace with the path to your saved model
    model_path = r'logistic_regression_model.pkl'
    model = joblib.load(model_path)
    return model

# Preprocessing steps
def crop_borders(image, top=10, bottom=10, left=10, right=10):
    return image[top:image.shape[0]-bottom, left:image.shape[1]-right]

def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def equilize_images(image):
    return cv2.equalizeHist(image)

def preprocess_image(image):
    cropped_image = crop_borders(image)
    normalized_image = normalize_image(equilize_images(cropped_image))
    return normalized_image

# 1. Ridge Endings and Bifurcations
def extract_ridge_endings_bifurcations(image):
    skeleton = thin(image > 127)
    struct_element = disk(1)
    convolved = convolve(skeleton.astype(np.uint8), struct_element)
    ridge_endings_count = np.sum(convolved == 1)
    bifurcations_count = np.sum(convolved == 3)
    return ridge_endings_count, bifurcations_count

# 2. Ridge and Valley Thickness
def extract_thickness(image):
    ridge_thickness = np.mean(image)  # Placeholder for ridge thickness
    valley_thickness = np.mean(255 - image)  # Placeholder for valley thickness
    return ridge_thickness, valley_thickness

# 3. Ridge Frequency
def extract_ridge_frequency(image):
    # Use Fourier Transform to estimate ridge frequency
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)
    magnitude = np.abs(fft_shift)
    ridge_frequency = np.mean(magnitude)  # Placeholder for ridge frequency
    return ridge_frequency

# 4. Local Binary Patterns (LBP)
def extract_lbp(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist / lbp_hist.sum()  # Normalize
    return lbp_hist.tolist()

# 5. Orientation
def extract_orientation(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    orientation_hist = np.histogram(orientation.ravel(), bins=8, range=(-180, 180))[0]
    orientation_hist = orientation_hist / orientation_hist.sum()  # Normalize
    return orientation_hist.tolist()

# 6. Wavelet Transform Features
def extract_wavelet_features(image):
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    features = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            for subband in coeff:
                features.append(np.mean(subband))
                features.append(np.var(subband))
        else:
            features.append(np.mean(coeff))
            features.append(np.var(coeff))
    return features

# Function to extract all features
def extract_features(image):
    features = {}
    # Ridge endings and bifurcations
    ridge_endings_count, bifurcations_count = extract_ridge_endings_bifurcations(image)
    features['ridge_endings'] = ridge_endings_count
    features['bifurcations'] = bifurcations_count
    
    # Ridge and valley thickness
    ridge_thickness, valley_thickness = extract_thickness(image)
    features['ridge_thickness'] = ridge_thickness
    features['valley_thickness'] = valley_thickness
    
    # Ridge frequency
    ridge_frequency = extract_ridge_frequency(image)
    features['ridge_frequency'] = ridge_frequency
    
    # LBP features
    lbp_features = extract_lbp(image)
    for i in range(9):
        features[f"lbp_{i}"] = lbp_features[i]
    
    # Orientation features
    orientation_features = extract_orientation(image)
    for i in range(8):
        features[f"orientation_{i}"] = orientation_features[i]
    
    # Wavelet features
    wavelet_features = extract_wavelet_features(image)
    for i in range(len(wavelet_features)):
        features[f"wavelet_{i}"] = wavelet_features[i]
    
    return features

# Streamlit app title
st.title("Fingerprint Based Blood Group Prediction")

# Upload fingerprint image
st.sidebar.header("Upload Your Fingerprint Image")
uploaded_image = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Read the uploaded image into a NumPy array
    image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # Display the uploaded image
    st.subheader("Uploaded Fingerprint Image")
    st.image(image, caption="Uploaded Fingerprint", width=150, channels="GRAY")

    # Preprocess the uploaded image
    processed_image = preprocess_image(image)

    # Extract features
    features = extract_features(processed_image)

    # Display extracted features
    st.subheader("Extracted Features")
    st.write(features)

    # Load the trained model
    model = load_model()

    # Convert features to a format suitable for prediction (e.g., flatten and reshape)
    feature_vector = np.array(list(features.values())).reshape(1, -1)

    # Predict the blood group
    predicted_blood_group = model.predict(feature_vector)[0]
    # Blood group mappings
    blood_group_mapping = {
        0: "AB+",
        1: "A-",
        2: "O+",
        3: "AB-",
        4: "B+",
        5: "B-",
        6: "A+",
        7: "O-"
    }

    blood_group = blood_group_mapping.get(predicted_blood_group, "Unknown")

    # Display prediction result
    st.subheader(f"Predicted Blood Group: {blood_group}")
