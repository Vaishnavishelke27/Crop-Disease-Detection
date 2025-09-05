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

# --- CORRECTED PATHS FOR C:\Crop-Disease-Detection ---
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'C:/Crop-Disease-Detection/your_model_files/plant_disease_model.h5'
CLASS_NAMES_PATH = 'C:/Crop-Disease-Detection/your_model_files/class_names.txt'
IMG_HEIGHT = 224
IMG_WIDTH = 224

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global Model and Class Names (Loaded Once) ---
model = None
CLASS_NAMES = []
# The name of the last convolutional layer in MobileNetV2
last_conv_layer_name = 'Conv_1' 

# --- Recommendations Data ---
RECOMMENDATIONS = {
    'Black_rot': {
        'fertilizers': ['Urea (46-0-0)', 'Potassium Sulfate (0-0-50)'],
        'pesticides': ['Mancozeb', 'Captan', 'Myclobutanil']
    },
    'Esca': {
        'fertilizers': ['Balanced NPK (10-10-10)', 'Calcium Nitrate'],
        'pesticides': ['Copper-based fungicides', 'Trichoderma harzianum (biofungicide)']
    },
    'Leaf_blight': {
        'fertilizers': ['Chlorothalonil', 'Azoxystrobin'],
        'pesticides': ['Phosphorus-rich fertilizer (e.g., Triple Superphosphate)', 'Potassium Nitrate']
    },
    'Healthy': {
        'fertilizers': ['General purpose fertilizer', 'Compost'],
        'pesticides': ['No pesticides needed. Focus on preventative care.']
    },
    'Unknown Disease (Index Mismatch)': {
        'fertilizers': ['Consult a local expert for specific recommendations.'],
        'pesticides': ['Consult a local expert for specific recommendations.']
    }
}

# --- Symptoms Data ---
SYMPTOMS = {
    'Black_rot': [
        'Small, tan circular spots with dark borders on leaves.',
        'Lesions on grape berries that turn them into hard, black mummies.',
        'Dark, sunken cankers on the canes.'
    ],
    'Esca': [
        'Tiger-stripe-like discoloration and necrosis on leaves.',
        'Internal wood of the trunk and branches shows dark, spongy decay.',
        'Sudden wilting of the plant (apoplexy).'
    ],
    'Leaf_blight': [
        'Irregular, tan to brown patches on the leaves.',
        'Spots that may be surrounded by a purple or dark brown halo.',
        'Defoliation, leading to reduced photosynthesis and weakened vines.'
    ],
    'Healthy': [
        'Leaves are a uniform green color.',
        'No visible spots, discoloration, or lesions.',
        'Grape berries are firm and free of decay.'
    ],
    'Unknown Disease (Index Mismatch)': [
        'Please consult a local expert for a proper diagnosis.'
    ]
}

def load_model_and_classes():
    """Loads the pre-trained Keras model and class names from files."""
    global model, CLASS_NAMES
    try:
        abs_model_path = os.path.abspath(MODEL_PATH)
        abs_class_names_path = os.path.abspath(CLASS_NAMES_PATH)
    except Exception as e:
        print(f"ERROR: Could not resolve absolute paths for model/class names. Details: {e}")
        return

    print(f"Attempting to load model from: {abs_model_path}")
    if not os.path.exists(abs_model_path):
        print(f"FATAL ERROR: Model file NOT FOUND at: {abs_model_path}")
        model = None
        return
    try:
        model = keras.models.load_model(abs_model_path)
        _ = model(tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3)))
        print("Model loaded successfully and built with dummy input.")
    except Exception as e:
        print(f"ERROR: Failed to load model from '{abs_model_path}'. Details: {e}")
        model = None
        return

    print(f"Attempting to load class names from: {abs_class_names_path}")
    if not os.path.exists(abs_class_names_path):
        print(f"ERROR: Class names file NOT FOUND at: {abs_class_names_path}")
        return
    try:
        with open(abs_class_names_path, 'r') as f:
            CLASS_NAMES = [line.strip() for line in f if line.strip()]
        if not CLASS_NAMES:
            print(f"WARNING: Class names file '{abs_class_names_path}' is empty or contains no valid names after stripping.")
        print(f"Class names loaded successfully: {CLASS_NAMES}")
        if model and len(CLASS_NAMES) != model.output_shape[1]:
            print(f"WARNING: Mismatch between number of loaded class names ({len(CLASS_NAMES)}) and model output classes ({model.output_shape[1]}).")
        print(f"Number of class names loaded: {len(CLASS_NAMES)}")
    except Exception as e:
        print(f"ERROR: Failed to load class names from '{abs_class_names_path}'. Details: {e}")
        CLASS_NAMES = []

with app.app_context():
    load_model_and_classes()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def make_gradcam_heatmap(img_array, model, pred_index=None):
    if model is None:
        print("Model not loaded, cannot create heatmap.")
        return None

    try:
        base_model = model.get_layer('mobilenetv2_1.00_224')
        # IMPORTANT FIX: Set the base model as trainable for gradient calculation
        base_model.trainable = True
    except ValueError:
        print(f"Could not find layer 'mobilenetv2_1.00_224'. Grad-CAM failed.")
        return None
    
    # Check if the last convolutional layer exists in the base model
    if not any(layer.name == last_conv_layer_name for layer in base_model.layers):
        print(f"Last convolutional layer '{last_conv_layer_name}' not found in base model.")
        return None

    try:
        conv_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=base_model.get_layer(last_conv_layer_name).output
        )
    except Exception as e:
        print(f"Error creating convolutional model for Grad-CAM: {e}")
        return None

    classifier_input = tf.keras.Input(shape=conv_model.output_shape[1:])
    x = classifier_input
    for layer in model.layers[1:]: 
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = conv_model(img_array)
        tape.watch(last_conv_layer_output)
        
        predictions = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        print("Failed to compute gradients. Check if the model's layers are trainable.")
        return None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = last_conv_layer_output[0] * pooled_grads[tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_sum(heatmap, axis=-1)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        heatmap = heatmap * 0
    else:
        heatmap = tf.maximum(heatmap, 0) / max_val
    
    return heatmap.numpy()


@app.route('/', methods=['GET'])
def index():
    error_message = request.args.get('error_message')
    return render_template('index.html', error_message=error_message)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not CLASS_NAMES or len(CLASS_NAMES) != model.output_shape[1]:
        return redirect(url_for('index', error_message="Model or class names not loaded correctly."))
    if 'file' not in request.files or not request.files['file'].filename:
        return redirect(url_for('index', error_message="No file selected."))

    try:
        file = request.files['file']
        image_bytes_original = file.read()
        img_array = preprocess_image(image_bytes_original)
        
        predictions = model.predict(img_array)
        score = predictions[0]
        predicted_class_index = np.argmax(score)
        
        predicted_class_name = CLASS_NAMES[predicted_class_index] if predicted_class_index < len(CLASS_NAMES) else "Unknown Disease (Index Mismatch)"
        confidence = np.max(score)

        symptoms = SYMPTOMS.get(predicted_class_name, SYMPTOMS['Unknown Disease (Index Mismatch)'])

        heatmap_image_base64 = None
        heatmap = make_gradcam_heatmap(img_array, model, predicted_class_index)
        if heatmap is not None and np.max(heatmap) > 0:
            original_img_for_heatmap = Image.open(io.BytesIO(image_bytes_original)).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
            cmap = cm.get_cmap("jet")
            jet_colors = cmap(np.arange(256))[:, :3]
            
            heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((original_img_for_heatmap.width, original_img_for_heatmap.height), Image.BICUBIC)
            heatmap_resized_np = np.array(heatmap_resized)
            jet_heatmap_np = (jet_colors[heatmap_resized_np] * 255).astype(np.uint8)
            
            superimposed_img_np = (np.array(original_img_for_heatmap) * 0.6 + jet_heatmap_np * 0.4).astype(np.uint8)
            superimposed_img = Image.fromarray(superimposed_img_np)

            buffered_heatmap = io.BytesIO()
            superimposed_img.save(buffered_heatmap, format="PNG")
            heatmap_image_base64 = b64encode(buffered_heatmap.getvalue()).decode('utf-8')

        original_img_for_display = Image.open(io.BytesIO(image_bytes_original)).convert('RGB')
        buffered_original = io.BytesIO()
        original_img_for_display.save(buffered_original, format="PNG")
        original_image_base64 = b64encode(buffered_original.getvalue()).decode('utf-8')
        
        return render_template('result.html', 
                               predicted_class=predicted_class_name,
                               confidence=f"{confidence * 100:.2f}%",
                               original_image=original_image_base64,
                               heatmap_image=heatmap_image_base64,
                               symptoms=symptoms)

    except Exception as e:
        print(f"An exception occurred: {e}")
        return redirect(url_for('index', error_message=str(e)))

@app.route('/treatment/<disease_name>')
def show_treatment(disease_name):
    recommendations = RECOMMENDATIONS.get(disease_name, RECOMMENDATIONS['Unknown Disease (Index Mismatch)'])
    return render_template('treatment.html', 
                           disease_name=disease_name.replace('_', ' '),
                           recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)