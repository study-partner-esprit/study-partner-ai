"""
Focus CNN Transfer Learning Training Script
Uses MobileNetV2 (ImageNet) as backbone with fine-tuning for focus detection.
Classes: Focused (0), Drifting (1), Lost (2)
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3
LABEL_MAP = {"Focused": 0, "Drifting": 1, "Lost": 2}
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
EPOCHS_FROZEN = 10
EPOCHS_FINE_TUNE = 20
FINE_TUNE_AT = -20  # Unfreeze last 20 layers


def build_model(num_classes: int = NUM_CLASSES) -> Model:
    """Build MobileNetV2 transfer-learning model."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False  # Freeze backbone initially

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    return model


def get_generators(labels_csv: str, batch_size: int = BATCH_SIZE):
    """Create train/val/test generators from labels CSV."""
    df = pd.read_csv(labels_csv)
    df["label_str"] = df["label_idx"].map({v: k for k, v in LABEL_MAP.items()})

    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df["label_idx"], random_state=42
    )
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, stratify=test_df["label_idx"], random_state=42
    )

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    def flow(gen, dataframe):
        return gen.flow_from_dataframe(
            dataframe,
            x_col="image_path",
            y_col="label_str",
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode="categorical",
            classes=list(LABEL_MAP.keys()),
            shuffle=True,
        )

    return flow(train_gen, train_df), flow(val_gen, val_df), flow(val_gen, test_df)


def train(labels_csv: str, output_dir: str):
    """Full training pipeline: frozen → fine-tune → save."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_model_path = str(output_path / "focus_model_transfer.h5")

    print(f"[1/4] Loading data from {labels_csv}")
    train_gen, val_gen, test_gen = get_generators(labels_csv)

    print("[2/4] Building MobileNetV2 model (frozen backbone)")
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
    ]

    print(f"[3/4] Phase 1 — Training top layers ({EPOCHS_FROZEN} epochs)")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks,
    )

    # Fine-tune: unfreeze last N layers of backbone
    base_model = (
        model.layers[1] if hasattr(model.layers[1], "layers") else model.layers[0]
    )
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(
        f"[3/4] Phase 2 — Fine-tuning last {abs(FINE_TUNE_AT)} layers ({EPOCHS_FINE_TUNE} epochs)"
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )

    print("[4/4] Evaluating on test set")
    loss, acc = model.evaluate(test_gen)
    print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

    model.save(best_model_path)
    print(f"\nModel saved to {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train focus CNN with transfer learning"
    )
    parser.add_argument(
        "--labels",
        default="data/labels_balanced.csv",
        help="Path to balanced labels CSV",
    )
    parser.add_argument(
        "--output",
        default="outputs/models",
        help="Directory to save trained model",
    )
    args = parser.parse_args()

    train(args.labels, args.output)
