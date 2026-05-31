import os
import json
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_loader import load_all_datasets,load_combined_dataset
from sklearn.preprocessing import LabelEncoder
from model import build_model


# --- Utility function to pad variable-length sequences ---
def pad_sequences_to_max(X):
    """Pad variable-length 2D arrays (seq_len, feature_dim) to max length."""
    max_len = max(x.shape[0] for x in X)
    feature_dim = X[0].shape[1]
    padded = np.zeros((len(X), max_len, feature_dim))
    for i, x in enumerate(X):
        padded[i, :x.shape[0], :] = x  # pad with zeros
    return padded


# --- Main training function ---
def train_all(base_path, save_models_dir="models", epochs=100, batch_size=64, combine=False):
    os.makedirs(save_models_dir, exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    if combine:
        print("🔗 Loading combined dataset...")
        X_train, y_train, X_test, y_test = load_combined_dataset(base_path)
        datasets = [("combined", X_train, y_train, X_test, y_test)]
    else:
        datasets = load_all_datasets(base_path)

    for subset, X_train, y_train, X_test, y_test in datasets:
        print(f"\n🚀 Training model for subset: {subset}")

        # Debug info
        print(f"Subset: {subset}")
        print("X_train type:", type(X_train))
        print("X_train dtype:", X_train.dtype)
        print("Example shape of first element:",
              X_train[0].shape if isinstance(X_train[0], np.ndarray) else type(X_train[0]))

        # --- Handle object arrays ---
        if X_train.dtype == object:
            try:
                # Try stacking if all shapes are equal
                X_train = np.stack(X_train, axis=0)
                X_test = np.stack(X_test, axis=0)
            except ValueError:
                # Fall back to padding if variable-length sequences
                print("⚠️ Variable-length sequences detected — padding to max length.")
                X_train = pad_sequences_to_max(X_train)
                X_test = pad_sequences_to_max(X_test)

        # --- Normalize ---
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

        # --- Add channel dimension if missing (for CNNs) ---
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

        print("✅ X_train final shape:", X_train.shape)

        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y_train))

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # --- Save label map ---
        label_map = {i: label for i, label in enumerate(le.classes_)}
        with open(f"labels/{subset}_labels.json", "w") as f:
            json.dump(label_map, f)

        # --- Build and compile model ---
        model = build_model(input_shape, num_classes)

        # --- Callbacks ---
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(save_models_dir, f"{subset}_cnn_model.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )

        # --- Train ---
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[checkpoint, reduce_lr],
            verbose=2
        )

        print(f"✅ Training finished for {subset}. Best model saved in {save_models_dir}")
