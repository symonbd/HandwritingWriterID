import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_all_datasets


def pad_sequences_to_max(X, max_len):
    """Pad sequences to the given max_len"""
    feature_dim = X[0].shape[1]
    padded = np.zeros((len(X), max_len, feature_dim))
    for i, x in enumerate(X):
        length = min(x.shape[0], max_len)
        padded[i, :length, :] = x[:length]
    return padded


def evaluate_all(base_path, models_dir="models", labels_dir="labels"):
    datasets = load_all_datasets(base_path)

    for subset, X_train, y_train, X_test, y_test in datasets:
        model_path = os.path.join(models_dir, f"{subset}_cnn_model.h5")
        label_map_path = os.path.join(labels_dir, f"{subset}_labels.json")
        seq_len_path = os.path.join(labels_dir, f"{subset}_seq_len.json")

        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {subset}, skipping...")
            continue
        if not os.path.exists(label_map_path) or not os.path.exists(seq_len_path):
            print(f"⚠️ Labels or sequence length info missing for {subset}, skipping...")
            continue

        print(f"\n📊 Evaluating CNN model for {subset}")
        model = load_model(model_path)

        # Load max sequence length
        with open(seq_len_path, "r") as f:
            max_len = json.load(f)["max_seq_len"]

        # Pad X_test
        if X_test.dtype == object:
            X_test = pad_sequences_to_max(X_test, max_len)
        X_test = X_test.astype("float32")

        # Remove channel for 1D CNN
        if len(X_test.shape) == 4 and X_test.shape[-1] == 1:
            X_test = X_test.squeeze(-1)

        # Load label map
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        # Encode labels
        y_is_string = y_test.dtype.kind in {'U', 'S', 'O'}
        if y_is_string:
            y_test_encoded = np.array([list(label_map.values()).index(str(label)) for label in y_test])
        else:
            y_test_encoded = y_test.astype(int)

        # Predict
        y_pred_probs = model.predict(X_test)
        y_pred_encoded = np.argmax(y_pred_probs, axis=1)

        # Convert back to original labels
        y_test_labels = [label_map[str(i)] for i in y_test_encoded]
        y_pred_labels = [label_map[str(i)] for i in y_pred_encoded]

        # Metrics
        print(classification_report(y_test_labels, y_pred_labels))
        acc = accuracy_score(y_test_labels, y_pred_labels)
        print(f"Accuracy for {subset}: {acc:.4f}")


# --- Example usage ---
if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "Data", "onhw-chars_2021-06-30")
    evaluate_all(
        base_path,
        models_dir=os.path.join(os.path.dirname(__file__), "models"),
        labels_dir=os.path.join(os.path.dirname(__file__), "labels")
    )
