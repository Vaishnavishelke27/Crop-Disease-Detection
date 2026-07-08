import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
import matplotlib.cm as cm
from base64 import b64encode

# Initialize the Flask app
app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'

# 🛑 Keeping the directory 'your_model_files' based on your last input
MODEL_DIR = 'your_model_files' 
MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.h5') 
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.txt')

IMG_HEIGHT = 224
IMG_WIDTH = 224

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True) 

# --- Global Model and Class Names (Loaded Once) ---
model = None
CLASS_NAMES = []
# NOTE: If your model is NOT MobileNetV2, you MUST change this to the name 
# of the final convolutional layer in your model for Grad-CAM to work.
last_conv_layer_name = 'Conv_1' 



def load_model_and_classes():
    """Loads the pre-trained Keras model and class names from files."""
    global model, CLASS_NAMES
    
    print("--- Model Loading Status ---")
    
    abs_model_path = os.path.abspath(MODEL_PATH)
    abs_classes_path = os.path.abspath(CLASS_NAMES_PATH)
    
    print(f"Looking for model at: {abs_model_path}")

    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file NOT FOUND.")
        print(f"Please ensure 'plant_disease_model.h5' and 'class_names.txt' are inside the '{MODEL_DIR}' folder.")
        model = None
        return
    
    try:
        # 🛑 FIX APPLIED HERE: Using compile=False to avoid layer input errors during loading
        model = keras.models.load_model(MODEL_PATH, compile=False)
        # Run a dummy prediction to force model building
        _ = model(tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3)))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from '{MODEL_PATH}'.")
        print(f"Check model integrity and Keras compatibility. Details: {e}")
        model = None
        return

    print(f"Looking for class names at: {abs_classes_path}")

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"FATAL ERROR: Class names file NOT FOUND.")
        print(f"Please ensure 'class_names.txt' is inside the '{MODEL_DIR}' folder.")
        CLASS_NAMES = []
        return
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            CLASS_NAMES = [line.strip() for line in f if line.strip()]
        
        if model and len(CLASS_NAMES) != model.output_shape[1]:
            print(f"WARNING: Class count mismatch! Loaded classes ({len(CLASS_NAMES)}) vs Model outputs ({model.output_shape[1]}).")
        else:
             print(f"Class names loaded successfully ({len(CLASS_NAMES)} classes).")
    except Exception as e:
        print(f"ERROR: Failed to load class names from '{CLASS_NAMES_PATH}'. Details: {e}")
        CLASS_NAMES = []
    print("----------------------------")

# Load model and classes immediately upon app startup
with app.app_context():
    load_model_and_classes()

def preprocess_image(image_bytes):
    """Converts image bytes to a normalized tensor for model prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    # Scale pixel values to [0, 1]
    img_array = img_array / 255.0
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generates a Grad-CAM heatmap to visualize model focus."""
    if model is None:
        return None

    try:
        # Find the base model layer (assuming MobileNetV2 for this example)
        base_model_name = 'mobilenetv2_1.00_224' 
        base_model = model
        try:
            # Try to locate the internal base model if it exists
            base_model = model.get_layer(base_model_name)
            base_model.trainable = True 
        except ValueError:
            pass # Use the entire model if the named base model isn't found

        # NOTE: This layer name must match the name of the last convolutional layer
        # in your specific model architecture for Grad-CAM to work correctly.
        conv_layer = None
        try:
            conv_layer = base_model.get_layer(last_conv_layer_name)
        except ValueError:
            try:
                # If not in the base model, check the full model
                conv_layer = model.get_layer(last_conv_layer_name)
            except ValueError:
                print(f"Grad-CAM Warning: Last conv layer '{last_conv_layer_name}' not found. Skipping Grad-CAM.")
                return None


        # 1. Create a temporary model to compute activations and predictions
        grad_model = tf.keras.models.Model(
            model.inputs, [conv_layer.output, model.output]
        )

        # 2. Compute activations and predictions
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]

        # 3. Compute gradients of the top predicted class with respect to the output of the conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            print("Failed to compute gradients. Check if model layers are trainable.")
            return None
        
        # 4. Compute the pooled gradients (weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 5. Multiply activation map by gradient weights
        heatmap = conv_outputs[0] * pooled_grads[tf.newaxis, tf.newaxis, :]
        heatmap = tf.reduce_sum(heatmap, axis=-1)
        
        # 6. Normalize the heatmap
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            heatmap = heatmap * 0
        else:
            heatmap = tf.maximum(heatmap, 0) / max_val
        
        return heatmap.numpy()

    except Exception as e:
        print(f"Critical Error during Grad-CAM: {e}")
        return None


@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    # Check if the model failed to load at startup and display a warning
    if model is None:
        error_message = "CRITICAL: ML Model failed to load. Please check file paths and model integrity."
    else:
        error_message = request.args.get('error_message')
    
    return render_template('index.html', error_message=error_message)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, prediction, and heatmap generation."""
    # Check if the model is ready before proceeding
    if model is None or not CLASS_NAMES or len(CLASS_NAMES) != model.output_shape[1]:
        return redirect(url_for('index', error_message="Prediction service unavailable. Model or class names not loaded correctly."))
    
    if 'file' not in request.files or not request.files['file'].filename:
        return redirect(url_for('index', error_message="No file selected."))

    try:
        file = request.files['file']
        image_bytes_original = file.read()
        
        # 1. Preprocess and Predict
        img_array = preprocess_image(image_bytes_original)
        predictions = model.predict(img_array)
        score = predictions[0]
        predicted_class_index = np.argmax(score)
        
        predicted_class_name = CLASS_NAMES[predicted_class_index] if predicted_class_index < len(CLASS_NAMES) else "Unknown Disease (Index Mismatch)"
        confidence = np.max(score)

        # 2. Generate Grad-CAM Image
        heatmap_image_base64 = None
        heatmap = make_gradcam_heatmap(img_array, model, predicted_class_index)
        
        if heatmap is not None and np.max(heatmap) > 0:
            original_img_for_heatmap = Image.open(io.BytesIO(image_bytes_original)).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
            cmap = cm.get_cmap("jet")
            jet_colors = cmap(np.arange(256))[:, :3]
            
            # Resize heatmap to match image size and convert to RGB colors
            heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((original_img_for_heatmap.width, original_img_for_heatmap.height), Image.BICUBIC)
            heatmap_resized_np = np.array(heatmap_resized)
            jet_heatmap_np = (jet_colors[heatmap_resized_np] * 255).astype(np.uint8)
            
            # Superimpose heatmap onto original image (adjust weights for visibility)
            superimposed_img_np = (np.array(original_img_for_heatmap) * 0.5 + jet_heatmap_np * 0.5).astype(np.uint8)
            superimposed_img = Image.fromarray(superimposed_img_np)

            # Convert superimposed image to base64 for embedding in HTML
            buffered_heatmap = io.BytesIO()
            superimposed_img.save(buffered_heatmap, format="PNG")
            heatmap_image_base64 = b64encode(buffered_heatmap.getvalue()).decode('utf-8')

        # 3. Prepare Original Image for Display
        original_img_for_display = Image.open(io.BytesIO(image_bytes_original)).convert('RGB')
        buffered_original = io.BytesIO()
        # Resize to fit screen better without distortion, then encode
        original_img_for_display.thumbnail((800, 600)) 
        original_img_for_display.save(buffered_original, format="PNG")
        original_image_base64 = b64encode(buffered_original.getvalue()).decode('utf-8')
        
        # 4. Render Result Page
        return render_template('result.html',
                                predicted_class=predicted_class_name,
                                confidence=f"{confidence * 100:.2f}%",
                                original_image=original_image_base64,
                                heatmap_image=heatmap_image_base64)

    except Exception as e:
        print(f"An exception occurred during prediction: {e}")
        return redirect(url_for('index', error_message=f"Prediction failed due to an internal error: {str(e)}"))



if __name__ == '__main__':
    # Set debug=False for production environments
    app.run(debug=True, host='0.0.0.0', port=5000)
