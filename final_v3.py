# final_v3_fixed.py
import cv2
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.base import clone
from skimage.feature import graycomatrix, graycoprops, hog
from skimage import img_as_ubyte
import joblib
from datetime import datetime
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

# ========================
# KONFIGURASI
# ========================
BASE_DIR = "hasil"
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CM_DIR = os.path.join(REPORTS_DIR, "cm")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_DIR = os.path.join(BASE_DIR, "scaler")
IMG_SIZE = (128, 128)

BORDER_PADDING = 50
SAT_THRESHOLD = 30
VALUE_THRESHOLD = 50
MORPH_ITERATIONS = 2
PADDING = 20

for folder in [DEBUG_DIR, REPORTS_DIR, CM_DIR, MODELS_DIR, SCALER_DIR]:
    os.makedirs(folder, exist_ok=True)

FRESH_RAW = "fresh_raw"
ROTTEN_RAW = "rotten_raw"
FRESH_TEST = "fresh_test"
ROTTEN_TEST = "rotten_test"

for f in [FRESH_RAW, ROTTEN_RAW, FRESH_TEST, ROTTEN_TEST]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Folder tidak ada: {f}")

# ========================
# 1. PREPROCESSING (DIPERBAIKI!)
# ========================
def process_and_save_image(image_path, debug_root):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[ERROR] Gagal: {image_path}")
        return None

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    debug_dir = os.path.join(debug_root, name)
    os.makedirs(debug_dir, exist_ok=True)

    def save_step(name, img):
        cv2.imwrite(os.path.join(debug_dir, name), img)

    save_step('00_original.jpg', image)

    border_value = [255, 255, 255, 0] if len(image.shape) == 3 and image.shape[2] == 4 else [255, 255, 255]
    image_padded = cv2.copyMakeBorder(image, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING,
                                      cv2.BORDER_CONSTANT, value=border_value)
    save_step('01_padded.jpg', image_padded)

    bgr = cv2.cvtColor(image_padded, cv2.COLOR_BGRA2BGR) if image_padded.shape[2] == 4 else image_padded
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    _, sat_mask = cv2.threshold(hsv[:, :, 1], SAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(hsv[:, :, 2], VALUE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    save_step('02_sat_mask.jpg', sat_mask)
    save_step('03_val_mask.jpg', val_mask)

    combined_mask = cv2.bitwise_or(sat_mask, val_mask)
    save_step('04_combined_mask.jpg', combined_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=MORPH_ITERATIONS)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=MORPH_ITERATIONS + 1)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    save_step('05_morphology.jpg', combined_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[SKIP] Tidak ada kontur: {image_path}")
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    x = max(0, x - PADDING); y = max(0, y - PADDING)
    w = min(image_padded.shape[1] - x, w + 2 * PADDING)
    h = min(image_padded.shape[0] - y, h + 2 * PADDING)
    roi = image_padded[y:y+h, x:x+w]
    save_step('06_roi_crop.jpg', roi)

    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    circle_mask = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, (255, 255, 255, 255), -1)
    if roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    circular_roi = cv2.bitwise_and(roi, circle_mask)
    save_step('07_circular_roi.png', circular_roi)

    resized = cv2.resize(circular_roi, IMG_SIZE, interpolation=cv2.INTER_AREA)
    final_path = os.path.join(debug_dir, '08_final_128x128.png')
    save_step('08_final_128x128.png', resized)

    print(f"[PROSES] {image_path} -> {final_path}")
    return final_path

# ========================
# 2. PROSES SEMUA
# ========================
print("1. PREPROCESSING...")
train_paths = [(p, 0) for p in glob.glob(f"{FRESH_RAW}/*.*")] + [(p, 1) for p in glob.glob(f"{ROTTEN_RAW}/*.*")]
test_paths  = [(p, 0) for p in glob.glob(f"{FRESH_TEST}/*.*")] + [(p, 1) for p in glob.glob(f"{ROTTEN_TEST}/*.*")]

train_processed = [(process_and_save_image(p, DEBUG_DIR), l) for p, l in train_paths if process_and_save_image(p, DEBUG_DIR)]
test_processed  = [(process_and_save_image(p, DEBUG_DIR), l) for p, l in test_paths  if process_and_save_image(p, DEBUG_DIR)]
train_processed = [(p, l) for p, l in train_processed if p]
test_processed  = [(p, l) for p, l in test_processed  if p]

print(f"Train: {len(train_processed)} | Test: {len(test_processed)}")

# ... (ekstraksi fitur, training, grafik, PDF — sama seperti sebelumnya)