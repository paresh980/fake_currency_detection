# =========================
# train_model.py - Full evaluation
# =========================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from keras.utils import to_categorical
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns

# -----------------------------
# 1️⃣ Load preprocessed dataset
# -----------------------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")
X_val   = np.load("X_val.npy")
y_val   = np.load("y_val.npy")

# -----------------------------
# 2️⃣ One-hot encode labels
# -----------------------------
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)
y_val_cat   = to_categorical(y_val, num_classes)

class_names = ["real", "fake"]  # Make sure this matches your dataset

# -----------------------------
# 3️⃣ Build MobileNetV2 model
# -----------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 4️⃣ Callbacks
# -----------------------------
checkpoint = ModelCheckpoint("currency_model_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# -----------------------------
# 5️⃣ Train the model
# -----------------------------
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint, earlystop]
)

# -----------------------------
# 6️⃣ Plot training history
# -----------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# -----------------------------
# 7️⃣ Evaluate on Test Set
# -----------------------------
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

test_acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test F1 Score: {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# -----------------------------
# 8️⃣ Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 9️⃣ Save final model
# -----------------------------
model.save("currency_detector_final.h5")
print("Model saved as currency_detector_final.h5")
