import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
import numpy as np

# --- REQUIRED DEPENDENCIES ---
# Run: pip install tensorflow keras numpy Pillow scikit-image scipy

# --- RELATIVE PATHS ---
TRAIN_DIR = 'dataset/train'
VALIDATION_DIR = 'dataset/validation'
MODEL_SAVE_PATH = 'your_model_files/plant_disease_model.h5'
CLASS_NAMES_PATH = 'your_model_files/class_names.txt'

# --- VERIFY PATHS BEFORE TRAINING ---
if not os.path.exists(TRAIN_DIR):
    print(f"FATAL ERROR: Training directory not found at '{TRAIN_DIR}'")
    exit()
if not os.path.exists(VALIDATION_DIR):
    print(f"FATAL ERROR: Validation directory not found at '{VALIDATION_DIR}'")
    exit()

# --- TRAINING CONFIGURATION ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Load class names from the file
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Class names loaded: {CLASS_NAMES}")
else:
    print(f"WARNING: Class names file not found. Assuming 4 classes.")
    NUM_CLASSES = 4

# --- DATA GENERATION AND AUGMENTATION ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
except Exception as e:
    print(f"Error creating data generators. Please check your dataset paths and structure.")
    print(f"Details: {e}")
    exit()

if train_generator.n == 0:
    print(f"FATAL ERROR: Training generator found 0 images in '{TRAIN_DIR}'. Please check your folder structure.")
    exit()
if validation_generator.n == 0:
    print(f"FATAL ERROR: Validation generator found 0 images in '{VALIDATION_DIR}'. Please check your folder structure.")
    exit()

# --- MODEL BUILDING AND TRAINING ---
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Build model using tf.keras.Model to avoid Sequential loading issues
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', name='classification_output')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Starting initial training of top layers...")
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

print("\nStarting fine-tuning of the entire model...")
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine_tune = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator
)

# Save class names after training to ensure consistency
class_names_from_generator = list(train_generator.class_indices.keys())
with open(CLASS_NAMES_PATH, 'w') as f:
    for class_name in class_names_from_generator:
        f.write(f"{class_name}\n")
print(f"Class names saved to: {CLASS_NAMES_PATH}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
print(f"\nTraining complete. Saving model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
