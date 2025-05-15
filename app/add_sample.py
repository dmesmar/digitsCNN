import base64
import os
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
import subprocess

MODEL_PATH = "model.h5"
TRAIN_SCRIPT = "app/model.py"
SAMPLES_DIR = Path("data/new_samples")

# Crear carpeta si no existe
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def save_sample(b64_img: str, label: int):
    if not (0 <= label <= 9):
        raise ValueError("Label must be an integer between 0 and 9")

    # Decodificar imagen
    try:
        image_data = base64.b64decode(b64_img)
        img = Image.open(BytesIO(image_data)).convert("L")  # Escala de grises
        img = img.resize((28, 28)) 
    except Exception:
        raise ValueError("Invalid image format")

    label_dir = SAMPLES_DIR / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    img_count = len(list(label_dir.glob("*.png")))
    img_path = label_dir / f"{img_count+1}.png"
    img.save(img_path)

    return img_path

def retrain_model():
    """Ejecuta el script de entrenamiento"""
    try:
        result = subprocess.run(["python", TRAIN_SCRIPT], check=True, capture_output=True)
        return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Retraining failed: {e.stderr.decode('utf-8')}")
