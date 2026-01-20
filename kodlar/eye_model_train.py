import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# =============================
# AYARLAR
# =============================
BASE_DATASET = r"D:/Surucu_Projesi/processed_dataset/eye"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
OUTPUT_MODEL = r"D:/Surucu_Projesi/models/eye_model.h5"

os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)

# =============================
# MODEL OLUŞTUR
# =============================
def build_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# =============================
# DATA GENERATOR
# =============================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True
)
test_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(os.path.join(BASE_DATASET,"train"), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
val   = test_gen.flow_from_directory(os.path.join(BASE_DATASET,"val"), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
test  = test_gen.flow_from_directory(os.path.join(BASE_DATASET,"test"), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

# =============================
# MODEL EĞİT
# =============================
model = build_model(num_classes=train.num_classes)

# Class weight hesapla (dengesiz dataset için)
classes = train.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(classes), y=classes)
class_weight_dict = dict(enumerate(class_weights))

model.fit(train, validation_data=val, epochs=EPOCHS, class_weight=class_weight_dict)

test_loss, test_acc = model.evaluate(test)
print(f"🎯 Eye Model Test Accuracy: {test_acc*100:.2f}%")

# Model kaydet
model.save(OUTPUT_MODEL)
print(f"💾 Eye Model kaydedildi: {OUTPUT_MODEL}")
