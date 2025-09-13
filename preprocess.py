import os
import cv2
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_DIR = "dataset"   # path to your dataset
IMG_SIZE = (224, 224)     # resize all images to 224x224
SAVE_PROCESSED = True     # save preprocessed dataset as numpy arrays

# -----------------------------
# FUNCTION TO LOAD AND PREPROCESS IMAGES
# -----------------------------
def load_images_from_folder(folder_path, img_size=(224, 224)):
    X, y = [], []
    class_names = sorted(os.listdir(folder_path))

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Resize & normalize
                img = cv2.resize(img, img_size)
                img = img.astype("float32") / 255.0

                X.append(img)
                y.append(label)

            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return np.array(X), np.array(y), class_names

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Loading dataset...")

    # Load train, test, val sets
    X_train, y_train, classes = load_images_from_folder(os.path.join(DATASET_DIR, "train"), IMG_SIZE)
    X_test, y_test, _ = load_images_from_folder(os.path.join(DATASET_DIR, "test"), IMG_SIZE)
    X_val, y_val, _ = load_images_from_folder(os.path.join(DATASET_DIR, "val"), IMG_SIZE)

    print(f"Classes found: {classes}")
    print(f"Train set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Val set: {X_val.shape[0]} images")
    print(f"Image shape: {X_train.shape[1:]}")

    # Save preprocessed data
    if SAVE_PROCESSED:
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", y_test)
        np.save("X_val.npy", X_val)
        np.save("y_val.npy", y_val)
        print("Preprocessed datasets saved as .npy files")
