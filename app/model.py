import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical



# ================================================================
# 1. Callback para parar cuando lleguemos al accuracy objetivo
# ================================================================
class StopAtValAccuracy(Callback):
    def __init__(self, target_acc: float = 0.93):
        super().__init__()
        self.target_acc = target_acc


# ================================================================
# 2. Preparación de datos
# ================================================================
def prepareData(csv_path='app/content/train.csv'):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, 1:].astype("float32").values / 255.0
    y = data.iloc[:, 0].values
    X = X.reshape(-1, 28, 28, 1)
    y = to_categorical(y, num_classes=10)
    return train_test_split(X, y, test_size=0.2,
                            random_state=42, stratify=y)

# ================================================================
# 3. Modelo (más pequeño)
# ================================================================
def createModel():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),

        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

# ================================================================
# 4. Entrenamiento
# ================================================================
def trainModel(model, X_train, y_train, X_val, y_val, epochs=12):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1)
    datagen.fit(X_train)

    # Eliminamos el callback StopAtValAccuracy
    ckpt = ModelCheckpoint('model.h5',
                           monitor='val_accuracy',
                           save_best_only=True,
                           verbose=0)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[ckpt],  # Solo dejamos el checkpoint
                        verbose=1)
    return history


# ================================================================
# 5. Evaluación
# ================================================================
def evaluateModel(model, X_val, y_val, history):
    pass
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nExactitud final (validación): {val_acc*100:.2f}%")

    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.axhline(max(history.history['val_accuracy']),
                ls='--', c='gray', label='Mejor val_acc')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.show()

    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title('Matriz de confusión'); plt.tight_layout(); plt.show()

    print(classification_report(y_true, y_pred))

# ================================================================
# 6. Main
# ================================================================
if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, X_val, y_train, y_val = prepareData()

    model = createModel()
    history = trainModel(model, X_train, y_train, X_val, y_val)  
    evaluateModel(model, X_val, y_val, history)

    print("\nModelo guardado en model.h5")