import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# AYARLAR
# ==============================
BASE_DATASET = r"D:/Surucu_Projesi/processed_dataset"
MODELS_DIR = r"D:/Surucu_Projesi/models"

tasks = ["eye", "yawn"]  # ayrı modeller
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
classes = ["Dikkatli", "Uykulu"]

# ==============================
# TEST VERİSİNİ YÜKLE
# ==============================
test_gen = ImageDataGenerator(rescale=1./255)

test_data = {}
for task in tasks:
    test_dir = os.path.join(BASE_DATASET, task, "test")
    test_gen_flow = test_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    test_data[task] = test_gen_flow

# ==============================
# MODELLERİ YÜKLE VE DEĞERLENDİR
# ==============================
eye_model = load_model(os.path.join(MODELS_DIR, "eye_model.h5"))
yawn_model = load_model(os.path.join(MODELS_DIR, "yawn_model.h5"))

def evaluate_model(model, test_gen):
    preds = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=classes))

print("=== EYE MODEL PERFORMANSI ===")
evaluate_model(eye_model, test_data["eye"])

print("\n=== YAWN MODEL PERFORMANSI ===")
evaluate_model(yawn_model, test_data["yawn"])

# ==============================
# HİBRİT KARAR PERFORMANSI
# ==============================
def hibrit_decision(eye_pred, yawn_pred):
    return np.where((eye_pred==1) | (yawn_pred==1), 1, 0)

print("\n=== HİBRİT KARAR PERFORMANSI ===")
# Eye ve Yawn test tahminleri
eye_preds = np.argmax(eye_model.predict(test_data["eye"], verbose=0), axis=1)
yawn_preds = np.argmax(yawn_model.predict(test_data["yawn"], verbose=0), axis=1)

# En kısa uzunluğu al
min_len = min(len(eye_preds), len(yawn_preds))
eye_preds = eye_preds[:min_len]
yawn_preds = yawn_preds[:min_len]

# Hibrit karar
hybrid_preds = hibrit_decision(eye_preds, yawn_preds)
y_true_hybrid = test_data["eye"].classes[:min_len]  # Eye test setini referans al

print(confusion_matrix(y_true_hybrid, hybrid_preds))
print(classification_report(y_true_hybrid, hybrid_preds, target_names=classes))
