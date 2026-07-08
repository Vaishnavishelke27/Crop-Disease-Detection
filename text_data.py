import tensorflow as tf
from tensorflow import keras

# Set the path exactly as it is in app.py
MODEL_PATH = 'your_model_files/plant_disease_model.h5'

try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("SUCCESS: Model loaded without error.")
except Exception as e:
    print(f"FAILURE: Model load failed. Error: {e}")

# Run this script: python text_data.py
