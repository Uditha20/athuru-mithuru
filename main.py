from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from PIL import Image
import os
import sys
import tensorflow as tf

app = FastAPI()

# Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# ===== Custom Cast Layer =====
class CastLayer(tf.keras.layers.Layer):
    def __init__(self, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

# ===== Model Loader on Startup =====
@app.on_event("startup")
async def load_ml_model():
    global model
    model_path = "model/dysgraphia_prediction_model.h5"

    print("=" * 50)
    print("TensorFlow version:", tf.__version__)
    print("Model path:", os.path.abspath(model_path))
    print("STARTING MODEL LOADING PROCESS")

    try:
        custom_objects = {
            'Cast': CastLayer,
            'cast': CastLayer,
            'tf.cast': tf.cast,
        }

        print("Trying with custom_objects for Cast layer...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully.")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"✗ Failed to load model: {e}")
        model = None

    print("=" * 50)
    print(f"FINAL MODEL STATUS: {'LOADED' if model is not None else 'FAILED'}")
    if model is None:
        print("RECOMMENDED ACTION: Check model format and custom layers.")
    print("=" * 50)

# ===== Health Check Endpoint =====
@app.get("/")
async def root():
    return {"message": "Dysgraphia Prediction API is running"}

# ===== Reload Model Manually =====
@app.post("/reload-model")
async def reload_model():
    global model
    model_path = "model/dysgraphia_prediction_model.h5"

    try:
        custom_objects = {
            'Cast': CastLayer,
            'cast': CastLayer,
            'tf.cast': tf.cast,
        }

        print("Manual reload requested...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)

        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_loaded": True,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to reload model: {str(e)}",
            "model_loaded": False,
            "error_type": type(e).__name__,
        }

# ===== Prediction Endpoint =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((300, 300))
        image_np = np.array(image).astype('float32') / 255.0
        image_np = np.expand_dims(image_np, axis=0)

        pred = model.predict(image_np)[0][0]
        label = "Dysgraphia" if pred > 0.5 else "No Dysgraphia"

        return {
            "prediction": label,
            "confidence": float(pred),
            "status": "success",
        }

    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "status": "error",
        }

# ===== Debug Info =====
@app.get("/debug")
async def debug_info():
    info = {
        "model_loaded": model is not None,
        "model_path_exists": os.path.exists("model/dysgraphia_prediction_model.h5"),
        "current_directory": os.getcwd(),
        "python_executable": sys.executable,
        "tensorflow_version": tf.__version__,
    }

    try:
        info["model_dir_contents"] = os.listdir("model")
    except:
        info["model_dir_contents"] = []

    if model is not None:
        info["model_info"] = {
            "type": str(type(model)),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }

    return info
