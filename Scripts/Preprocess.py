import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
input_file = "../CTR_data/criteo-uplift-v2.1.csv"  # Path to your CSV file
output_dir = "./Data/processed/"                 # Directory to save the processed splits
npy_dir = "./Data/npy/"                               # Directory to save the numpy files
train_ratio = 0.8                                 # Proportion of training data
val_ratio = 0.1                                   # Proportion of validation data
test_ratio = 0.1                                  # Proportion of test data

# Seed for reproducibility
random_seed = 42                                  # Set a fixed random seed for reproducibility

# Ensure ratios sum to 1
assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)

# File paths for processed .csv files
processed_files = {
    "train": os.path.join(output_dir, "train.csv"),
    "val": os.path.join(output_dir, "val.csv"),
    "test": os.path.join(output_dir, "test.csv"),
}

# Check if processed CSV files already exist
if all(os.path.exists(file) for file in processed_files.values()):
    print("Processed .csv files detected. Skipping dataset segmentation...")
else:
    # Load dataset
    print("Loading dataset...")
    with tqdm(total=1, desc="Loading data") as pbar:
        data = pd.read_csv(input_file)
        pbar.update(1)

    # Shuffle and split data into train and temp (val + test)
    print("Splitting dataset...")
    train_data, temp_data = train_test_split(
        data, test_size=(1 - train_ratio), random_state=random_seed
    )
    # Split temp data into validation and test sets
    val_data, test_data = train_test_split(
        temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed
    )

    # Verify splits
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    # Save the splits
    print("Saving datasets...")
    for dataset, name in tqdm([(train_data, "train"), (val_data, "val"), (test_data, "test")], desc="Saving Data"):
        dataset.to_csv(processed_files[name], index=False)

print("Converting .csv files to .npy format...")

# Convert .csv files to .npy files
for name, file_path in tqdm(processed_files.items(), desc="Converting to .npy"):
    data = pd.read_csv(file_path)
    npy_file_path = os.path.join(npy_dir, f"{name}.npy")
    np.save(npy_file_path, data.values)
    print(f"Saved {name}.npy to {npy_dir}")

print("Data preprocessing and conversion complete!")
