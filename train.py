# ==========================================================
# Malaria Detection - Production Training Script
# MobileNetV2 transfer learning + fine-tuning
# ==========================================================
# Folder layout expected (matches your project):
#   Dataset/Train/Parasite, Dataset/Train/Uninfected
#   Dataset/Test/Parasite,  Dataset/Test/Uninfected
#
# Outputs (app.py works with these unchanged):
#   malaria_model.keras   -> final model
#   class_indices.json    -> label mapping
#   training_curves.png, confusion_matrix.png, roc_curve.png
# ==========================================================

import os
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# ==========================================================
# SETTINGS
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "Dataset" / "Train"   # capital D/T matches your folders
TEST_DIR = BASE_DIR / "Dataset" / "Test"

# IMPORTANT: app.py uses target_size=(128, 128). Keep these equal.
IMG_SIZE = 128
BATCH_SIZE = 32
VAL_SPLIT = 0.2

EPOCHS_HEAD = 8          # phase 1: train only the new classifier head
EPOCHS_FINE = 20         # phase 2: fine-tune top of MobileNetV2
FINE_TUNE_LAYERS = 40    # how many top layers of the base to unfreeze

MODEL_PATH = BASE_DIR / "malaria_model.keras"
CLASS_INDEX_PATH = BASE_DIR / "class_indices.json"

for p in (TRAIN_DIR, TEST_DIR):
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {p}")

# ==========================================================
# DATA GENERATORS
# Augmentation ONLY on training data. Validation/test data are
# only preprocessed, so metrics are honest.
# ==========================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=(0.85, 1.15),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
    validation_split=VAL_SPLIT,
)

eval_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VAL_SPLIT,
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

flow_args = dict(
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    seed=SEED,
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, subset="training", shuffle=True, **flow_args
)
# Same seed + same split => validation images never overlap training images
val_gen = eval_datagen.flow_from_directory(
    TRAIN_DIR, subset="validation", shuffle=False, **flow_args
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, shuffle=False, **flow_args
)

print("Class indices:", train_gen.class_indices)
assert train_gen.class_indices == test_gen.class_indices, \
    "Train and Test folders have different class names!"

with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

class_names = [None] * len(train_gen.class_indices)
for name, idx in train_gen.class_indices.items():
    class_names[idx] = name

# ==========================================================
# CLASS WEIGHTS (handles any imbalance)
# ==========================================================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weight = {i: w for i, w in enumerate(weights)}
print("Class weights:", class_weight)

# ==========================================================
# BUILD MODEL
# ==========================================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False   # phase 1: fully frozen

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)   # keeps BatchNorm in inference mode
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

metrics = [
    "accuracy",
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]


def make_callbacks(patience_stop, patience_lr):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ==========================================================
# PHASE 1 - TRAIN CLASSIFIER HEAD
# ==========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=metrics,
)
model.summary()

print("\n=== Phase 1: training classifier head ===")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weight,
    callbacks=make_callbacks(patience_stop=3, patience_lr=2),
)

# ==========================================================
# PHASE 2 - FINE-TUNE TOP LAYERS (low learning rate)
# BatchNorm layers stay frozen - this is critical for stable
# fine-tuning with small batches.
# ==========================================================
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=metrics,
)

print("\n=== Phase 2: fine-tuning ===")
checkpoint = ModelCheckpoint(
    str(MODEL_PATH),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD + EPOCHS_FINE,
    initial_epoch=len(history_head.history["loss"]),
    class_weight=class_weight,
    callbacks=make_callbacks(patience_stop=6, patience_lr=3) + [checkpoint],
)

# Load the best checkpoint (falls back to current weights if none saved)
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model.save(MODEL_PATH)
print(f"\nBest model saved to: {MODEL_PATH}")

# ==========================================================
# TRAINING CURVES
# ==========================================================
def merge(key):
    return history_head.history.get(key, []) + history_fine.history.get(key, [])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
    ax.plot(merge(key), label="Train")
    ax.plot(merge(f"val_{key}"), label="Validation")
    ax.axvline(len(history_head.history["loss"]) - 1, color="gray",
               linestyle="--", label="Fine-tune start")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "training_curves.png", dpi=150)
plt.close()

# ==========================================================
# FINAL EVALUATION ON HELD-OUT TEST SET
# ==========================================================
print("\n=== Evaluation on Test set ===")
test_gen.reset()
probs = model.predict(test_gen, verbose=1).ravel()
y_true = test_gen.classes
y_pred = (probs >= 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
auc = roc_auc_score(y_true, probs)
print(f"ROC-AUC: {auc:.4f}")

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# Confusion matrix plot
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Test)")
plt.xticks(range(len(class_names)), class_names, rotation=20)
plt.yticks(range(len(class_names)), class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# ROC curve plot
fpr, tpr, _ = roc_curve(y_true, probs)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "roc_curve.png", dpi=150)
plt.close()

print("\nDone. Saved: malaria_model.keras, class_indices.json, "
      "training_curves.png, confusion_matrix.png, roc_curve.png")