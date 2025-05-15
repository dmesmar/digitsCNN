import os
import io
import base64
import subprocess
import sys

import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "model.h5"
TRAIN_SCRIPT = "app/model.py"

# 1. Cargar o entrenar el modelo
def _load_or_train():
    """
    Devuelve un modelo Keras. Si model.h5 no existe,
    ejecuta model.py para crearlo.
    """
    if not os.path.exists(MODEL_PATH):
        print("model.h5 no encontrado. Entrenando modelo…")
        subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
        print("Entrenamiento finalizado.\n")

    # Ahora sí debería existir
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


_MODEL = _load_or_train()

#2. Preprocesado de la imagen
def _preprocess_image(b64_img: str) -> np.ndarray:
    """
    Convierte una cadena base64 a array (1, 28, 28, 1) float32 [0,1].
    Según cómo genere el usuario la imagen, quizá haga falta invertir
    los colores; mantengo esa línea comentada.
    """
    # Decodificar base64 → bytes
    img_bytes = base64.b64decode(b64_img)

    # Abrir con Pillow
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # 'L' = 8-bit grayscale
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.asarray(img).astype("float32") / 255.0       # 0-1

    arr = 1.0 - arr  

    arr = arr.reshape(1, 28, 28, 1)
    return arr

# 3. Predicción
def predict(b64_img: str) -> dict:
    """
    b64_img : str
        Cadena base64 de la imagen (PNG/JPG).
    return : dict
        {'class': int, 'confidence': float}
    """
    arr = _preprocess_image(b64_img)
    probs = _MODEL.predict(arr, verbose=0)[0]
    print(probs)
    dict_res = {}

    for i, p in enumerate(probs): 
        dict_res[i] = float(round(p, 3))           
        print(f"Número {i}: {p:.4f}")

    return dict_res

if __name__ == "__main__":
    import argparse, pathlib, base64

    parser = argparse.ArgumentParser(description="Predice dígitos MNIST desde imagen.")
    parser.add_argument("image_path", type=str, help="Ruta a la imagen PNG/JPG")
    args = parser.parse_args()

    img_path = pathlib.Path(args.image_path)
    if not img_path.exists():
        sys.exit(f" No se encuentra {img_path}")

    with open(img_path, "rb") as f:
        b64data = base64.b64encode(f.read()).decode()

    result = predict(b64data)
    print(result)

