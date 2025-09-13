import tensorflow as tf
from keras import layers, models

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = (224, 224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

# -----------------------------
# Dataset Loader
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode='binary',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode='binary'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode='binary'
)

# -----------------------------
# Fix Image Channels
# -----------------------------
def fix_channels(image, label):
    # Ensure dtype is float32
    image = tf.image.convert_image_dtype(image, tf.float32)

    channels = tf.shape(image)[-1]

    # Case 1: More than 3 channels → take first 3
    image = tf.cond(channels > 3,
                    lambda: image[..., :3],
                    lambda: image)

    # Case 2: Only 1 channel → tile to 3 channels
    image = tf.cond(tf.equal(channels, 1),
                    lambda: tf.tile(image, [1, 1, 3]),
                    lambda: image)

    return image, label



train_ds = train_ds.map(fix_channels, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(fix_channels, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_ds.map(fix_channels, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# -----------------------------
# Data Augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# -----------------------------
# Model
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze for transfer learning

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# -----------------------------
# Training Stage 1
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# -----------------------------
# Fine-tuning
# -----------------------------
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100
)
train_ds = train_ds.map(fix_channels)
val_ds = val_ds.map(fix_channels)
test_ds = test_ds.map(fix_channels)

# -----------------------------
# Evaluation
# -----------------------------
loss, acc, prec, rec = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.2f}")
print(f"Test Precision: {prec:.2f}")
print(f"Test Recall: {rec:.2f}")

# -----------------------------
# Save Model
# -----------------------------
model.save("currency_detector.h5")
import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
