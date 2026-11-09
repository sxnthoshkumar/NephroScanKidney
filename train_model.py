import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# Define paths
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "kidney_model.keras")

DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ==============================
# Data augmentation
# ==============================

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# ==============================
# Load training data
# ==============================

train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Train"),
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# ==============================
# Load validation data
# ==============================

val_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Train"),
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# ==============================
# Build CNN Model
# ==============================

model = keras.Sequential([
    
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(150, 150, 3)
    ),
    
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        len(train_generator.class_indices),
        activation="softmax"
    )
])

# ==============================
# Compile model
# ==============================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Display model structure
# ==============================

model.summary()

# ==============================
# Train model
# ==============================

EPOCHS = 10

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ==============================
# Save model and class mapping
# ==============================

model.save(MODEL_PATH)

import json
mapping_path = os.path.join(MODEL_DIR, "class_indices.json")
mapping_data = {
    "raw_indices": train_generator.class_indices,
    "classes": {
        str(v): {
            "folder_name": k,
            "display_name": "Cystine Kidney Stone" if k == "Cyst" else ("Normal (Healthy)" if k == "Normal" else "Renal Tumor"),
            "sub_title": "Cystine Calculus / Renal Cyst" if k == "Cyst" else ("Normal Renal Architecture" if k == "Normal" else "Suspicious Renal Neoplasm"),
            "category": "Cystine Stone / Cyst" if k == "Cyst" else ("Normal" if k == "Normal" else "Tumor"),
            "severity": "Moderate" if k == "Cyst" else ("Low" if k == "Normal" else "Critical"),
            "color": "#F59E0B" if k == "Cyst" else ("#10B981" if k == "Normal" else "#EF4444"),
            "status_badge": "Pathological Finding Detected (Cystine Stone)" if k == "Cyst" else ("Normal Scan - No Abnormalities" if k == "Normal" else "Critical Finding - Urgent Follow-Up")
        }
        for k, v in train_generator.class_indices.items()
    }
}
with open(mapping_path, "w") as f:
    json.dump(mapping_data, f, indent=2)

print("====================================")
print("Model training completed!")
print("Model saved at:", MODEL_PATH)
print("Class Indices mapped:", train_generator.class_indices)
print("Mapping saved at:", mapping_path)
print("====================================")