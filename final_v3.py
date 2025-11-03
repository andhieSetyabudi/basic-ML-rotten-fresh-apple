# final_v3_fixed.py
# LENGKAP: Preprocessing → Fitur → Model → Grafik → PDF → Debug Gambar
# SEMUA DISIMPAN DI DALAM FOLDER "hasil/"
# TIDAK ADA ERROR: ValueError, Path, Unicode, dsb

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
# KONFIGURASI UTAMA
# ========================
BASE_DIR = "hasil"  # SEMUA HASIL DI SINI
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CM_DIR = os.path.join(REPORTS_DIR, "cm")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_DIR = os.path.join(BASE_DIR, "scaler")
IMG_SIZE = (128, 128)

# Preprocessing Parameters
BORDER_PADDING = 50
SAT_THRESHOLD = 30
VALUE_THRESHOLD = 50
MORPH_ITERATIONS = 2
PADDING = 20

# Buat folder di dalam hasil/
for folder in [DEBUG_DIR, REPORTS_DIR, CM_DIR, MODELS_DIR, SCALER_DIR]:
    os.makedirs(folder, exist_ok=True)

# Folder input
FRESH_RAW = "fresh_raw"
ROTTEN_RAW = "rotten_raw"
FRESH_TEST = "fresh_test"
ROTTEN_TEST = "rotten_test"

# Cek folder input
for f in [FRESH_RAW, ROTTEN_RAW, FRESH_TEST, ROTTEN_TEST]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Folder tidak ditemukan: {f}")

print(f"Folder input ditemukan: {FRESH_RAW}, {ROTTEN_RAW}, {FRESH_TEST}, {ROTTEN_TEST}")
print(f"Semua hasil akan disimpan di: {os.path.abspath(BASE_DIR)}")

# ========================
# 1. PREPROCESSING + SIMPAN SEMUA GAMBAR KE hasil/debug/
# ========================
def process_and_save_image(image_path, debug_root):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[ERROR] Gagal membaca gambar: {image_path}")
        return None

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    debug_dir = os.path.join(debug_root, name)
    os.makedirs(debug_dir, exist_ok=True)

    # Helper: simpan gambar
    def save_step(step_name, img):
        path = os.path.join(debug_dir, step_name)
        success = cv2.imwrite(path, img)
        if not success:
            print(f"[GAGAL SIMPAN] {path}")

    # 00: Original
    save_step('00_original.jpg', image)

    # 01: Padding
    border_value = [255, 255, 255, 0] if len(image.shape) == 3 and image.shape[2] == 4 else [255, 255, 255]
    image_padded = cv2.copyMakeBorder(image, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING,
                                      cv2.BORDER_CONSTANT, value=border_value)
    save_step('01_padded.jpg', image_padded)

    # Convert to BGR
    bgr = cv2.cvtColor(image_padded, cv2.COLOR_BGRA2BGR) if image_padded.shape[2] == 4 else image_padded
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 02: Saturation Mask
    _, sat_mask = cv2.threshold(hsv[:, :, 1], SAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    save_step('02_sat_mask.jpg', sat_mask)

    # 03: Value Mask
    _, val_mask = cv2.threshold(hsv[:, :, 2], VALUE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    save_step('03_val_mask.jpg', val_mask)

    # 04: Combined Mask
    combined_mask = cv2.bitwise_or(sat_mask, val_mask)
    save_step('04_combined_mask.jpg', combined_mask)

    # 05: Morphology
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=MORPH_ITERATIONS)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=MORPH_ITERATIONS + 1)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    save_step('05_morphology.jpg', combined_mask)

    # Kontur
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

    # 07: Circular ROI
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    circle_mask = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, (255, 255, 255, 255), -1)
    if roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    circular_roi = cv2.bitwise_and(roi, circle_mask)
    save_step('07_circular_roi.png', circular_roi)

    # 08: Final Resize
    resized = cv2.resize(circular_roi, IMG_SIZE, interpolation=cv2.INTER_AREA)
    final_path = os.path.join(debug_dir, '08_final_128x128.png')
    save_step('08_final_128x128.png', resized)

    print(f"[PROSES] {filename} → {final_path}")
    return final_path

# ========================
# 2. PROSES SEMUA GAMBAR
# ========================
print("\n1. PREPROCESSING SEMUA GAMBAR (TRAIN & TEST)")

train_paths = [(p, 0) for p in glob.glob(f"{FRESH_RAW}/*.*")] + [(p, 1) for p in glob.glob(f"{ROTTEN_RAW}/*.*")]
test_paths  = [(p, 0) for p in glob.glob(f"{FRESH_TEST}/*.*")] + [(p, 1) for p in glob.glob(f"{ROTTEN_TEST}/*.*")]

print(f"   Train: {len(train_paths)} | Test: {len(test_paths)}")

train_processed = []
for path, label in train_paths:
    processed = process_and_save_image(path, DEBUG_DIR)
    if processed:
        train_processed.append((processed, label))

test_processed = []
for path, label in test_paths:
    processed = process_and_save_image(path, DEBUG_DIR)
    if processed:
        test_processed.append((processed, label))

print(f"Preprocessing selesai!")
print(f"   Train: {len(train_processed)} | Test: {len(test_processed)}")
print(f"   Gambar debug: {os.path.abspath(DEBUG_DIR)}")

# ========================
# 3. EKSTRAKSI FITUR
# ========================
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] Gagal baca fitur: {img_path}")
        return None
    if img.shape[2] != 4:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2BGRA)
    img = cv2.resize(img, IMG_SIZE)

    bgr = img[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hists = [cv2.calcHist([hsv], [i], None, [256], [0, 256]).flatten() for i in range(3)]
    color = np.hstack(hists)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = img_as_ubyte(gray)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
    glcm_feat = np.array([graycoprops(glcm, p)[0,0] for p in props])

    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')

    return np.hstack([color, glcm_feat, hog_feat])

print("\n2. EKSTRAKSI FITUR")
X_train = np.array([f for f in [extract_features(p) for p, l in train_processed] if f is not None])
y_train = np.array([l for p, l in train_processed if extract_features(p) is not None])
X_test  = np.array([f for f in [extract_features(p) for p, l in test_processed] if f is not None])
y_test  = np.array([l for p, l in test_processed if extract_features(p) is not None])

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Tidak ada fitur yang diekstrak!")

# ========================
# 4. SCALING
# ========================
print("\n3. SCALING")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(SCALER_DIR, "scaler.joblib"))

# ========================
# 5. MODEL & TRAINING
# ========================
models = [
    ("SVM", SVC(kernel='rbf', C=50, probability=True, random_state=42), True),
    ("RF", RandomForestClassifier(n_estimators=300, random_state=42), False),
    ("LogReg", LogisticRegression(max_iter=2000, C=1.0, random_state=42), True)
]

trained_models = []
results = []

print("\n4. TRAINING MODEL")
for name, model, scale in models:
    print(f"   Training {name}...")
    X_fit = X_train_s if scale else X_train
    model.fit(X_fit, y_train)
    model_path = os.path.join(MODELS_DIR, f"{name.lower()}_model.joblib")
    joblib.dump(model, model_path)
    trained_models.append((name, model, scale))

    X_eval = X_test_s if scale else X_test
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, model.predict_proba(X_eval)) if hasattr(model, 'predict_proba') else np.nan
    results.append({'model': name, 'accuracy': acc, 'loss': loss})

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Fresh','Rotten'], yticklabels=['Fresh','Rotten'])
    plt.title(f'{name} | Acc: {acc:.4f}')
    plt.tight_layout()
    cm_path = os.path.join(CM_DIR, f"cm_{name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

# ========================
# 6. GRAFIK HISTORY DETAIL
# ========================
print("\n5. GRAFIK TRAINING HISTORY")
def plot_detailed_history(name, model, X_tr, y_tr, X_val, y_val, ax_acc, ax_loss):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_tr, y_tr, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42
    )
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

    ax_acc.plot(train_sizes, train_mean, 'o-', color='#1f77b4', label='Train', lw=2)
    ax_acc.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#1f77b4')
    ax_acc.plot(train_sizes, val_mean, 's-', color='#ff7f0e', label='Validation', lw=2)
    ax_acc.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#ff7f0e')
    ax_acc.set_title(f'{name} - Accuracy', fontweight='bold')
    ax_acc.set_xlabel('Training Set Size'); ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(); ax_acc.grid(True, alpha=0.5); ax_acc.set_ylim(0.7, 1.02)

    if hasattr(model, "predict_proba"):
        sizes = np.linspace(0.2, 1.0, 9) * len(X_tr)
        sizes = np.unique(sizes.astype(int))
        if len(sizes) < 9:
            sizes = np.linspace(max(10, int(0.2*len(X_tr))), len(X_tr), 9).astype(int)
        train_loss, val_loss = [], []
        for size in sizes:
            if size < 10: continue
            idx = np.random.choice(len(X_tr), size, replace=False)
            temp_model = clone(model)
            temp_model.fit(X_tr[idx], y_tr[idx])
            train_loss.append(log_loss(y_tr[idx], temp_model.predict_proba(X_tr[idx])))
            val_loss.append(log_loss(y_val, temp_model.predict_proba(X_val)))
        ax_loss.semilogy(sizes, train_loss, 'o-', color='#d62728', label='Train')
        ax_loss.semilogy(sizes, val_loss, 's-', color='#2ca02c', label='Validation')
    else:
        ax_loss.text(0.5, 0.5, 'No Loss\n(RF)', ha='center', va='center', transform=ax_loss.transAxes, fontsize=12)
    ax_loss.set_title(f'{name} - Log Loss'); ax_loss.set_xlabel('Training Size')
    ax_loss.set_ylabel('Log Loss'); ax_loss.legend(); ax_loss.grid(True, which='both')

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Detailed Training History: Accuracy & Loss', fontsize=18, fontweight='bold')
for i, (name, model, scale) in enumerate(trained_models):
    X_tr = X_train_s if scale else X_train
    X_te = X_test_s if scale else X_test
    plot_detailed_history(name, model, X_tr, y_train, X_te, y_test, axes[i, 0], axes[i, 1])
plt.tight_layout(rect=[0, 0, 1, 0.96])
history_path = os.path.join(REPORTS_DIR, "training_history_detailed.png")
plt.savefig(history_path, dpi=400, bbox_inches='tight')
plt.close()
print(f"   → {history_path}")

# ========================
# 7. GRAFIK PERBANDINGAN
# ========================
df_res = pd.DataFrame(results)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(df_res['model'], df_res['accuracy'], color=['#4CAF50','#2196F3','#FF9800'])
ax1.set_title('Test Accuracy'); ax1.set_ylim(0.8, 1)
for i, v in enumerate(df_res['accuracy']): ax1.text(i, v + 0.005, f'{v:.4f}', ha='center')
loss_vals = [v if not np.isnan(v) else 999 for v in df_res['loss']]
colors = ['#F44336' if x < 999 else '#9E9E9E' for x in loss_vals]
ax2.bar(df_res['model'], loss_vals, color=colors)
ax2.set_title('Log Loss')
for i, v in enumerate(loss_vals):
    if v < 999: ax2.text(i, v + 0.005, f'{v:.4f}', ha='center')
    else: ax2.text(i, 0.05, 'N/A', ha='center')
plt.tight_layout()
comp_path = os.path.join(REPORTS_DIR, "model_comparison.png")
plt.savefig(comp_path, dpi=400, bbox_inches='tight')
plt.close()

# ========================
# 8. PDF LAPORAN LENGKAP
# ========================
print("\n6. MEMBUAT PDF LAPORAN")
pdf_path = os.path.join(BASE_DIR, "LAPORAN_APEL.pdf")
with PdfPages(pdf_path) as pdf:
    # Cover
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    ax.text(0.1, 0.9, "LAPORAN KLASIFIKASI APEL SEGAR & BUSUK", fontsize=20, fontweight='bold')
    ax.text(0.1, 0.8, f"{datetime.now().strftime('%d %B %Y, %H:%M')}", fontsize=12)
    ax.text(0.1, 0.7, f"Data: {len(train_processed)} train + {len(test_processed)} test", fontsize=12)
    ax.text(0.1, 0.6, "Model: SVM, RF, Logistic Regression", fontsize=12)
    ax.text(0.1, 0.5, "Fitur: HSV Histogram + GLCM + HOG", fontsize=12)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Contoh Preprocessing
    if train_processed:
        sample_dir = os.path.dirname(train_processed[0][0])
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        steps = [
            ('00_original.jpg', 'Original'), ('01_padded.jpg', 'Padding'),
            ('02_sat_mask.jpg', 'Sat Mask'), ('03_val_mask.jpg', 'Val Mask'),
            ('04_combined_mask.jpg', 'Combined'), ('05_morphology.jpg', 'Morphology'),
            ('06_roi_crop.jpg', 'ROI Crop'), ('07_circular_roi.png', 'Circular ROI'),
            ('08_final_128x128.png', 'Final 128x128')
        ]
        for i, (file, title) in enumerate(steps):
            img_path = os.path.join(sample_dir, file)
            if os.path.exists(img_path):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                ax = axes[i//3, i%3]
                ax.imshow(img); ax.set_title(title, fontsize=10, fontweight='bold'); ax.axis('off')
        plt.suptitle('Contoh Proses Preprocessing (1 Gambar)', fontsize=16, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Grafik
    for img_path in [history_path, comp_path]:
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            fig, ax = plt.subplots(figsize=(8.5, 6)); ax.imshow(img); ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Confusion Matrix
    for cm_file in glob.glob(f"{CM_DIR}/*.png"):
        img = plt.imread(cm_file)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(img); ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

print(f"\nSELESAI 100%! LAPORAN SIAP!")
print(f"   PDF: {os.path.abspath(pdf_path)}")
print(f"   Debug: {os.path.abspath(DEBUG_DIR)}")
print(f"   Model: {os.path.abspath(MODELS_DIR)}")