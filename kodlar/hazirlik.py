import os
import shutil
from sklearn.model_selection import train_test_split

# ANA DATASET YOLU
BASE_DIR = r"D:/Surucu_Projesi/dataset/Yawn/dataset_new"

# HAZIRLANMIŞ DATASET
OUTPUT_DIR = r"D:/Surucu_Projesi/processed_dataset"

CLASSES = {
    "eye": ["open", "closed"],        # 🔧 küçük harf
    "yawn": ["yawn", "no_yawn"]
}

SPLITS = ["train", "val", "test"]

IMG_EXTENSIONS = (".jpg", ".png", ".jpeg")


def prepare_folders():
    for task in CLASSES:
        for split in SPLITS:
            for cls in CLASSES[task]:
                os.makedirs(os.path.join(OUTPUT_DIR, task, split, cls), exist_ok=True)


def split_and_copy(task, cls):
    src_folder = os.path.join(BASE_DIR, "train", cls)
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(IMG_EXTENSIONS)]

    train_imgs, val_imgs = train_test_split(
        images, test_size=0.2, random_state=42
    )

    for img in train_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(OUTPUT_DIR, task, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(OUTPUT_DIR, task, "val", cls, img)
        )

    # test set zaten hazır
    test_folder = os.path.join(BASE_DIR, "test", cls)
    for img in os.listdir(test_folder):
        if img.lower().endswith(IMG_EXTENSIONS):
            shutil.copy(
                os.path.join(test_folder, img),
                os.path.join(OUTPUT_DIR, task, "test", cls, img)
            )


if __name__ == "__main__":
    prepare_folders()

    # Eye modeli için
    for cls in CLASSES["eye"]:
        split_and_copy("eye", cls)

    # Yawn modeli için
    for cls in CLASSES["yawn"]:
        split_and_copy("yawn", cls)

    print("✅ Dataset başarıyla hazırlandı (open/closed uyumlu).")
