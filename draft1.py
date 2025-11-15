#---------------------------------------------------------------------------------Model that gave 5% accuraacy----------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset
from PIL import Image
import numpy as np

# --- 1. Configuration ---
IMG_SIZE = 64  # We used a smaller 64x64 size for the custom model
BATCH_SIZE = 32
EPOCHS = 30    # We ran for 30 epochs

# --- 2. Load and Prepare the Dataset ---
print("Loading dataset...")
# Load the dataset
dataset = load_dataset("kannanwisen/Indian-Traffic-Sign-Classification")

# Split the 'train' data into new 'train' and 'validation' splits (80/20)
train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_data = train_val_split['train']
val_data = train_val_split['test']

# Get class names and number of classes
class_names = train_data.features['label'].names
num_classes = len(class_names)

print(f"Dataset loaded: {len(train_data)} training samples, {len(val_data)} validation samples.")
print(f"Number of classes: {num_classes}")

# --- 3. Preprocessing ---
def preprocess_image(examples):
    """Resize images to the model's expected input size."""
    examples['image'] = [img.resize((IMG_SIZE, IMG_SIZE)) for img in examples['image']]
    return examples

print("Applying preprocessing (resizing images)...")
train_data = train_data.map(preprocess_image, batched=True)
val_data = val_data.map(preprocess_image, batched=True)

# Set the format to TensorFlow for the model
train_data.set_format(type='tensorflow', columns=['image', 'label'])
val_data.set_format(type='tensorflow', columns=['image', 'label'])

# --- 4. Data Augmentation ---
# These layers help prevent overfitting by creating varied images
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="data_augmentation")

# --- 5. THE FAILED MODEL: Simple 3-Block CNN ---
# This is the model that was too simple to learn the task.
print("Building the simple 3-block CNN model...")
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Normalize pixel values
    layers.Rescaling(1./255),
    
    # Apply augmentation
    data_augmentation,
    
    # --- Convolutional Block 1 ---
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # --- Convolutional Block 2 ---
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # --- Convolutional Block 3 ---
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # --- Classifier Head ---
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout to help prevent overfitting
    layers.Dense(num_classes, activation='softmax') # Output layer for 85 classes
], name="simple_cnn_v1")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use this for integer labels
    metrics=['accuracy']
)

model.summary()

# --- 6. Prepare tf.data.Dataset for Training ---
# This function correctly formats the data for Keras
def format_for_keras(example_dict):
    image = example_dict['image']
    label = example_dict['label']
    return tf.cast(image, tf.float32), label

print("Preparing datasets for training...")
train_ds = train_data.to_tf_dataset(
    columns=['image', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE
).map(format_for_keras).prefetch(tf.data.AUTOTUNE)

val_ds = val_data.to_tf_dataset(
    columns=['image', 'label'],
    shuffle=False, # No need to shuffle validation data
    batch_size=BATCH_SIZE
).map(format_for_keras).prefetch(tf.data.AUTOTUNE)

# --- 7. Train the Model ---
print(f"Starting model training for {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("Training complete.")

# --- 8. Evaluate the Model (and see the low score) ---
print("\nEvaluating model on the validation set:")
results = model.evaluate(val_ds)
print(f"\nFinal Test Loss: {results[0]}")
print(f"Final Test Accuracy: {results[1] * 100:.2f}%")

# This script will output the low accuracy score (around 5%)
# that demonstrated this model architecture was not sufficient.
