import cv2
import os
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import csv
import random

# ========================
# LOAD MODELS & SCALER
# ========================
print("Loading models dan scaler...")
try:
    svm_model = joblib.load('svm_model.joblib')
    rf_model = joblib.load('rf_model.joblib')
    lr_model = joblib.load('lr_model.joblib')
    scaler = joblib.load('scaler.joblib')
    TARGET_FEATURES = scaler.n_features_in_  # 8874
    print(f"Model dimuat. Scaler mengharapkan {TARGET_FEATURES} fitur.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

# ========================
# FITUR EKSTRAKSI (SAMA PERSIS DENGAN final_v3.py)
# ========================
def extract_features_from_single_roi(roi):
    img = cv2.resize(roi, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Histogram HSV: 256 bins × 3 channel
    hist_h = cv2.calcHist([hsv], [0], None, [256], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])  # 768

    # GLCM: 4 distances × 4 angles × 6 properties
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img_as_ubyte(gray)
    distances = [1, 2, 3, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    texture_features = []

    for d in distances:
        for a in angles:
            glcm = graycomatrix(gray, [d], [a], levels=256, symmetric=True, normed=True)
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in props:
                texture_features.append(graycoprops(glcm, prop)[0, 0])

    texture_features = np.array(texture_features)  # 96
    return np.concatenate([color_features, texture_features])  # 864

def generate_multiple_rois(apple_roi, n_patches=10):
    h, w = apple_roi.shape[:2]
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return []

    indices = random.sample(range(len(ys)), min(n_patches, len(ys)))
    patches = []

    for idx in indices:
        cy, cx = ys[idx], xs[idx]
        y1, y2 = max(0, cy - 64), min(h, cy + 64)
        x1, x2 = max(0, cx - 64), min(w, cx + 64)
        patch = apple_roi[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        if patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
        patch = cv2.resize(patch, (128, 128))
        patches.append(patch)

    return patches

def extract_apple_features(apple_roi, n_patches=10):
    patches = generate_multiple_rois(apple_roi, n_patches)
    if not patches:
        return None

    all_features = [extract_features_from_single_roi(p) for p in patches]
    combined = np.hstack(all_features)  # 864 * n_patches

    # Zero-padding agar cocok dengan scaler
    if len(combined) < TARGET_FEATURES:
        pad_width = TARGET_FEATURES - len(combined)
        combined = np.pad(combined, (0, pad_width), 'constant', constant_values=0)
    combined = combined[:TARGET_FEATURES]  # Pastikan tidak melebihi

    print(f"  Fitur: {len(combined)} (target: {TARGET_FEATURES})")
    return combined

# ========================
# PREPROCESS + PADDING
# ========================
def preprocess_image(image, min_size=128, padding_size=5):
    h, w = image.shape[:2]
    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w
    border_value = [255, 255, 255, 0] if image.ndim == 3 and image.shape[2] == 4 else [255, 255, 255]
    padded = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size,
                                cv2.BORDER_CONSTANT, value=border_value)
    return padded, (padding_size, padding_size)

# ========================
# DETEKSI APEL
# ========================
def detect_all_apples(image_path, min_area=1000, sat_threshold=25, value_threshold=250, morph_iterations=3):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, [], None

    image_padded, padding_offset = preprocess_image(image)
    bgr = cv2.cvtColor(image_padded, cv2.COLOR_BGRA2BGR) if image_padded.shape[2] == 4 else image_padded
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]

    _, sat_mask = cv2.threshold(saturation, sat_threshold, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(value, value_threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_or(sat_mask, val_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=morph_iterations)
    mask = cv2.dilate(mask, kernel, iterations=morph_iterations + 1)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apples = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image_padded[y:y+h, x:x+w]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2

        # Circular ROI utama
        circle_mask = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, (255,255,255,255), -1)
        if roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
        circular_roi = cv2.bitwise_and(roi, circle_mask)
        main_roi = cv2.resize(circular_roi, (128, 128))

        # Ekstrak fitur (multiple ROI + hstack + padding)
        features = extract_apple_features(main_roi, n_patches=10)
        if features is None:
            continue

        center_orig = (x + center[0] - padding_offset[0], y + center[1] - padding_offset[1])
        apples.append({
            'roi': main_roi,
            'center': center_orig,
            'radius': radius,
            'features': features
        })

    apples = sorted(apples, key=lambda a: a['center'][0])
    return image, apples

# ========================
# ANOTASI
# ========================
def annotate_image(original, apples, preds, model_name, dir_path, name):
    img = original.copy()
    for i, apple in enumerate(apples):
        label = preds[i]['label']
        conf = preds[i]['conf']
        color = (255, 0, 0) if label == "FRESH" else (0, 0, 255)
        cv2.circle(img, apple['center'], apple['radius'], color, 6)
        text = f"{label} {conf:.0%}"
        org = (apple['center'][0] - 50, apple['center'][1] + apple['radius'] + 35)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    path = os.path.join(dir_path, f"{name}_ANOTASI_{model_name}.png")
    cv2.imwrite(path, img)
    return path

# ========================
# PREDIKSI + ANOTASI
# ========================
def predict_and_annotate(image_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    anotasi_dir = os.path.join(output_dir, 'anotasi')
    os.makedirs(anotasi_dir, exist_ok=True)

    original, apples = detect_all_apples(image_path)
    if original is None or not apples:
        print(f"Skip: {image_path}")
        return

    print(f"\n{len(apples)} apel terdeteksi: {filename}")
    predictions = []
    svm_preds = []; rf_preds = []; lr_preds = []; final_preds = []

    for i, apple in enumerate(apples):
        features = apple['features']
        if len(features) != TARGET_FEATURES:
            print(f"  Warning: Fitur {len(features)}, skip apel {i+1}")
            continue

        scaled = scaler.transform([features])
        raw = [features]

        # SVM
        p_svm = svm_model.predict_proba(scaled)[0]
        l_svm = "FRESH" if np.argmax(p_svm) == 0 else "ROTTEN"
        c_svm = p_svm[np.argmax(p_svm)]
        svm_preds.append({'label': l_svm, 'conf': c_svm})

        # RF
        p_rf = rf_model.predict_proba(raw)[0]
        l_rf = "FRESH" if np.argmax(p_rf) == 0 else "ROTTEN"
        c_rf = p_rf[np.argmax(p_rf)]
        rf_preds.append({'label': l_rf, 'conf': c_rf})

        # LR
        p_lr = lr_model.predict_proba(scaled)[0]
        l_lr = "FRESH" if np.argmax(p_lr) == 0 else "ROTTEN"
        c_lr = p_lr[np.argmax(p_lr)]
        lr_preds.append({'label': l_lr, 'conf': c_lr})

        # Gabungan
        avg = (p_svm + p_rf + p_lr) / 3
        final = "FRESH" if np.argmax(avg) == 0 else "ROTTEN"
        conf_final = avg[np.argmax(avg)]
        final_preds.append({'label': final, 'conf': conf_final})

        roi_path = os.path.join(output_dir, f"{name}_apel_{i+1}_roi.png")
        cv2.imwrite(roi_path, apple['roi'])

        print(f"  Apel {i+1}: {final} ({conf_final:.1%}) | SVM: {l_svm} {c_svm:.1%} | RF: {l_rf} {c_rf:.1%} | LR: {l_lr} {c_lr:.1%}")

        predictions.append({
            'Apel': i+1, 'SVM_Label': l_svm, 'SVM_Conf': f"{c_svm:.1%}",
            'RF_Label': l_rf, 'RF_Conf': f"{c_rf:.1%}",
            'LR_Label': l_lr, 'LR_Conf': f"{c_lr:.1%}",
            'Final_Label': final, 'Final_Conf': f"{conf_final:.1%}"
        })

    # Simpan anotasi
    annotate_image(original, apples, svm_preds, 'SVM', anotasi_dir, name)
    annotate_image(original, apples, rf_preds, 'RF', anotasi_dir, name)
    annotate_image(original, apples, lr_preds, 'LR', anotasi_dir, name)
    annotate_image(original, apples, final_preds, 'GABUNGAN', anotasi_dir, name)

    # Simpan CSV
    csv_path = os.path.join(output_dir, f"{name}_predictions.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Apel','SVM_Label','SVM_Conf','RF_Label','RF_Conf','LR_Label','LR_Conf','Final_Label','Final_Conf'])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  CSV: {csv_path}")

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    images = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img in sorted(images):
        predict_and_annotate(img, args.output)
    print(f"\nSELESAI! Hasil di: {args.output}/anotasi/")