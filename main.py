import os
from train import train_all
# from data_loader import load_all_datasets

def main():
    # Get the absolute path of the current folder
    base_path = os.path.dirname(__file__)

    # Point to your dataset folder (relative to this script)
    dataset_path = os.path.join(base_path, "Data", "onhw-chars_2021-06-30")

    # ✅ Call your training function
    train_all(
        base_path=dataset_path,
        save_models_dir=os.path.join(base_path, "models"),
        epochs=20,
        batch_size=64,
        combine=False
    )

if __name__ == "__main__":
    main()
