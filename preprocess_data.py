# preprocess_data.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset_from_dir(dir_path, img_size=(64, 64)):
    images, labels = [], []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    class_names = sorted([f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))])
    label_map = {cls: idx for idx, cls in enumerate(class_names)}

    print(f"📁 Classes found: {class_names}")

    for cls in class_names:
        cls_folder = os.path.join(dir_path, cls)
        for img_file in os.listdir(cls_folder):
            if not img_file.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(cls_folder, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Skipped unreadable image: {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label_map[cls])
            except Exception as e:
                print(f"❌ Error loading {img_path}: {e}")
                continue

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    if images.ndim != 4:
        raise ValueError(f"❌ Data shape invalid: Expected 4D, got {images.shape}")

    return images, labels, label_map

def load_data(data_dir='archive/train/asl_alphabet_train', img_size=(64, 64)):
    print("✅ Loading data...")
    X, y, label_map = load_dataset_from_dir(data_dir, img_size)

    # Split into train/test (Kaggle version doesn't separate them)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test), label_map