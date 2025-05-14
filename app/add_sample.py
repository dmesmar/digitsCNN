import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pathlib import Path

MODEL_PATH = "model.h5"
NEW_DATA_DIR = Path("data/new_samples")
IMG_SIZE = (28, 28)

def load_new_samples():
    X, y = [], []

    for label_dir in NEW_DATA_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        label = int(label_dir.name)
        for img_path in label_dir.glob("*.png"):
            img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0  # Normalizar
            X.append(img_array)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes=10)
    return X, y

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("El modelo no existe. Entrénalo inicialmente primero.")

    X, y = load_new_samples()
    if len(X) == 0:
        raise ValueError("No hay muestras nuevas para reentrenar.")

    print(f"Entrenando con {len(X)} nuevas muestras...")
    model = load_model(MODEL_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=1)
    model.save(MODEL_PATH)
    print("Modelo actualizado y guardado.")
if __name__ == "__main__":
    main()
