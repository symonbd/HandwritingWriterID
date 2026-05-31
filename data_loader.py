import numpy as np
import os

def load_all_datasets(base_path):
    """Loads all .npy files from every subfolder under base_path
    Returns list of tuples: (subset_name, X_train, y_train, X_test, y_test)
    """
    all_data = []


    for subset in sorted(os.listdir(base_path)):
        subset_path = os.path.join(base_path, subset)
        if not os.path.isdir(subset_path):
            continue


    required_files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]

    if all(os.path.exists(os.path.join(subset_path, f)) for f in required_files):
        print(f"📂 Loading subset: {subset}")

        X_train = np.load(os.path.join(subset_path, "X_train.npy"), allow_pickle=True)
        y_train = np.load(os.path.join(subset_path, "y_train.npy"), allow_pickle=True)
        X_test = np.load(os.path.join(subset_path, "X_test.npy"), allow_pickle=True)
        y_test = np.load(os.path.join(subset_path, "y_test.npy"), allow_pickle=True)

        all_data.append((subset, X_train, y_train, X_test, y_test))
    else:
        print(f"⚠️ Skipping {subset}: Missing files")


    return all_data




def load_combined_dataset(base_path):
    """Concatenate all subsets into one combined dataset."""
    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []


    for subset in sorted(os.listdir(base_path)):
        subset_path = os.path.join(base_path, subset)
        if not os.path.isdir(subset_path):
            continue


        try:
            X_train = np.load(os.path.join(subset_path, "X_train.npy"), allow_pickle=True)
            y_train = np.load(os.path.join(subset_path, "y_train.npy"), allow_pickle=True)
            X_test = np.load(os.path.join(subset_path, "X_test.npy"), allow_pickle=True)
            y_test = np.load(os.path.join(subset_path, "y_test.npy"), allow_pickle=True)

            X_train_all.append(X_train)
            y_train_all.append(y_train)
            X_test_all.append(X_test)
            y_test_all.append(y_test)


        except FileNotFoundError:
            continue


    if not X_train_all:
        raise ValueError(f"No datasets found in {base_path}")


    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)


    print(f"✅ Combined data: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test