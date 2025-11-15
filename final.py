import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from datasets import load_dataset
from PIL import Image
import numpy as np

# --- 1. Configuration ---
IMG_SIZE = 128     
BATCH_SIZE = 32
INITIAL_EPOCHS = 50 # Max epochs for phase 1 (will be stopped early)
FINE_TUNE_EPOCHS = 100 # Max epochs for phase 2 (will be stopped early)

# --- 2. Load and Prepare the Dataset ---
print("Loading dataset...")
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

# --- 3. Preprocessing (for ResNet) ---
def preprocess_image(examples):
    """Resize images to the model's expected input size."""
    examples['image'] = [img.resize((IMG_SIZE, IMG_SIZE)) for img in examples['image']]
    return examples

print("Applying preprocessing (resizing images to 128x128)...")
train_data = train_data.map(preprocess_image, batched=True)
val_data = val_data.map(preprocess_image, batched=True)

# Set the format to TensorFlow for the model
train_data.set_format(type='tensorflow', columns=['image', 'label'])
val_data.set_format(type='tensorflow', columns=['image', 'label'])

# --- 4. Prepare tf.data.Dataset for Training ---
# This function applies the ResNet-specific normalization
def format_for_keras(example_dict):
    image = tf.cast(example_dict['image'], tf.float32)
    label = example_dict['label']
    # Apply the ResNet-specific preprocessing
    image = preprocess_input(image) 
    return image, label

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

# --- 5. Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# --- 6. Build the Model (Transfer Learning) ---
print("Building ResNet50 model...")
# Load pre-trained model, without its final classifier layer
base_model = ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Do NOT include the final 1000-class layer
    weights='imagenet'  # Use the pre-trained weights
)

# Freeze the base model. We will only train our new "head"
base_model.trainable = False

# Build our new model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    base_model, # Add the frozen ResNet base
    
    # --- This is our new "Head" ---
    layers.GlobalAveragePooling2D(), 
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax') # Our final output layer
], name="resnet50_transfer_learning")

# --- 7. Define Callbacks ---
# These tools automatically manage training.
print("Defining callbacks...")
# This will reduce the learning rate when validation loss stops improving
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, # new_lr = old_lr * 0.2
    patience=3, # Wait 3 epochs before reducing
    min_lr=1e-6,
    verbose=1
)

# This will stop training early if there's no improvement
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=7, # Wait 7 epochs after the loss stops dropping
    restore_best_weights=True, # <-- Restores the best model weights
    verbose=1
)

# --- 8. Phase 1: Train the Head ---
print("\n--- PHASE 1: TRAINING THE HEAD ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS, 
    callbacks=[lr_scheduler, early_stopper]
)
print("--- PHASE 1 COMPLETE. BEST WEIGHTS RESTORED. ---")

# --- 9. Phase 2: Fine-Tuning ---
print("\n--- PHASE 2: FINE-TUNING ---")

# Unfreeze the top layers of the base model
base_model.trainable = True

# We unfreeze the last 20 layers for fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compile the model with a VERY LOW learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 100x smaller!
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training from where we left off
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS, 
    initial_epoch=len(history_phase1.history['loss']), # Start from last epoch
    callbacks=[lr_scheduler, early_stopper]
)
print("--- FINE-TUNING COMPLETE. BEST WEIGHTS RESTORED. ---")

# --- 10. Save and Evaluate the Final Model ---
model.save("resnet_finetuned_model.h5")
print("Final model saved as 'resnet_finetuned_model.h5'")

print("\nEvaluating final, fine-tuned model:")
results = model.evaluate(val_ds)
print(f"\nFinal Test Loss: {results[0]}")
print(f"Final Test Accuracy: {results[1] * 100:.2f}%")

