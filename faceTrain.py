import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "images"

# Load metadata (not directly needed for training but useful for mapping)
metadata = {}
for label in os.listdir(DATA_DIR):
    meta_path = os.path.join(DATA_DIR, label, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata[label] = json.load(f)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

# Data generators
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # Use categorical since we have two classes
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# Class indices mapping
class_indices = train_generator.class_indices
print(f"Class Mapping: {class_indices}")

# Load MobileNetV2 base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers initially

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(class_indices), activation="softmax")(x)  # Softmax for multi-class classification

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train first phase (only top layers)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("face_recognition_model.h5", save_best_only=True)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS // 2,  # Train first half with frozen base model
    callbacks=callbacks
)

# Unfreeze some layers of MobileNetV2 for fine-tuning
for layer in base_model.layers[-50:]:  # Unfreeze last 50 layers
    layer.trainable = True

# Compile again with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train second phase (fine-tuning)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS // 2,
    callbacks=callbacks
)

print("Training complete! Model saved as 'face_recognition_model.h5'")
