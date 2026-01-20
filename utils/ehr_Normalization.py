import numpy as np
import pickle
import json
import os
import csv
from pathlib import Path

class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            dct = pickle.load(load_file)
            self._means = dct['means']
            self._stds = dct['stds']

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret

def identify_numerical_columns(header):
    """Identify which columns are numerical (not one-hot encoded or masks)"""
    numerical_indices = []
    for i, col_name in enumerate(header):
        # Skip one-hot encoded columns (contain '->')
        if '->' in col_name:
            continue
        # Skip mask columns
        if col_name.startswith('mask->'):
            continue
        numerical_indices.append(i)
    return numerical_indices

def load_csv_data(file_path):
    """Load CSV file and return data and header"""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            # Convert string values to float, handling empty strings
            processed_row = []
            for val in row:
                if val == '':
                    processed_row.append(0.0)
                else:
                    try:
                        processed_row.append(float(val))
                    except ValueError:
                        processed_row.append(0.0)  # Fallback for unexpected values
            data.append(processed_row)
        return np.array(data), header

def save_normalized_data(data, header, output_path):
    """Save normalized data as CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            # Format to avoid scientific notation
            formatted_row = []
            for val in row:
                if isinstance(val, (int, np.integer)):
                    formatted_row.append(str(val))
                elif isinstance(val, (float, np.floating)):
                    if abs(val) < 1e-6:  # Very small number
                        formatted_row.append('0.0')
                    elif val == int(val):
                        formatted_row.append(str(int(val)))
                    else:
                        formatted_row.append(f"{val:.8f}".rstrip('0').rstrip('.'))
                else:
                    formatted_row.append(str(val))
            writer.writerow(formatted_row)

def get_time_series_files_from_listfile(listfile_path, data_dir):
    
    time_series_files = []
    with open(listfile_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) > 2:  # Make sure there are enough columns
                time_series_filename = row[2]  # time_series column
                file_path = Path(data_dir) / time_series_filename
                if file_path.exists():
                    time_series_files.append(file_path)
                else:
                    print(f"Warning: File not found: {file_path}")
    return time_series_files

def normalize_datasets(train_listfile, val_listfile, test_listfile, data_dir, output_base_dir):
    
    data_path = Path(data_dir)
    output_path = Path(output_base_dir)
    
    # Create output directories
    output_train_dir = output_path / "train"
    output_val_dir = output_path / "val"
    output_test_dir = output_path / "test"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_val_dir.mkdir(parents=True, exist_ok=True)
    output_test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all files from listfiles
    train_files = get_time_series_files_from_listfile(train_listfile, data_dir)
    val_files = get_time_series_files_from_listfile(val_listfile, data_dir)
    test_files = get_time_series_files_from_listfile(test_listfile, data_dir)
    
    if not train_files:
        print(f"No CSV files found from {train_listfile}")
        return
    
    # Read first training file to get header and identify numerical columns
    first_data, header = load_csv_data(train_files[0])
    numerical_indices = identify_numerical_columns(header)
    
    print(f"Found {len(numerical_indices)} numerical columns to normalize")
    print(f"Numerical columns: {[header[i] for i in numerical_indices]}")
    
    # PHASE 1: Calculate normalization parameters from TRAINING data only
    print("\n=== PHASE 1: Calculating normalization parameters from TRAINING data ===")
    normalizer = Normalizer(fields=numerical_indices)
    
    for csv_file in train_files:
        print(f"Processing {csv_file.name} for statistics...")
        data, _ = load_csv_data(csv_file)
        normalizer._feed_data(data)
    
    # Save normalization parameters
    params_path = output_path / "normalizer_params.pkl"
    normalizer._save_params(params_path)
    print(f"Saved normalization parameters to {params_path}")
    
    # PHASE 2: Normalize TRAINING data
    print("\n=== PHASE 2: Normalizing TRAINING data ===")
    for csv_file in train_files:
        print(f"Normalizing {csv_file.name}...")
        data, header = load_csv_data(csv_file)
        normalized_data = normalizer.transform(data)
        output_file = output_train_dir / f"{csv_file.name}"
        save_normalized_data(normalized_data, header, output_file)
        print(f"Saved normalized training data to {output_file}")
    
    # PHASE 3: Normalize VALIDATION data using training parameters
    print("\n=== PHASE 3: Normalizing VALIDATION data using training parameters ===")
    val_normalizer = Normalizer(fields=numerical_indices)
    val_normalizer.load_params(params_path)
    
    for csv_file in val_files:
        print(f"Normalizing {csv_file.name}...")
        data, header = load_csv_data(csv_file)
        normalized_data = val_normalizer.transform(data)
        output_file = output_val_dir / f"{csv_file.name}"
        save_normalized_data(normalized_data, header, output_file)
        print(f"Saved normalized validation data to {output_file}")
    
    # PHASE 4: Normalize TEST data using training parameters
    print("\n=== PHASE 4: Normalizing TEST data using training parameters ===")
    test_normalizer = Normalizer(fields=numerical_indices)
    test_normalizer.load_params(params_path)
    
    for csv_file in test_files:
        print(f"Normalizing {csv_file.name}...")
        data, header = load_csv_data(csv_file)
        normalized_data = test_normalizer.transform(data)
        output_file = output_test_dir / f"{csv_file.name}"
        save_normalized_data(normalized_data, header, output_file)
        print(f"Saved normalized test data to {output_file}")
    
    # Print summary
    print(f"\n=== NORMALIZATION COMPLETE ===")
    print(f"Processed {len(train_files)} training files")
    print(f"Processed {len(val_files)} validation files")
    print(f"Processed {len(test_files)} test files")
    print(f"Normalized training files saved to: {output_train_dir}")
    print(f"Normalized validation files saved to: {output_val_dir}")
    print(f"Normalized test files saved to: {output_test_dir}")
    print(f"Normalization parameters saved to: {params_path}")

if __name__ == "__main__":
    # Configuration
    train_listfile = ""  #train metadata path
    val_listfile = ""   # val metadata path
    test_listfile = ""   # test metadata path
    data_dir = ""  # preprocessed ehr data directory
    output_base_dir = ""
    
    normalize_datasets(train_listfile, val_listfile, test_listfile, data_dir, output_base_dir)


