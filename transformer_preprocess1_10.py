#1秒目から10秒目を使って11秒目を当てる際の前処理
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class CFDVelocityDataset(Dataset):
    def __init__(self, base_dir, chunk_size=5000, max_value=1e3, test=False):
        """
        Initialize the dataset by reading and processing the data in chunks to avoid memory overflow.

        Args:
            base_dir (str): The base directory containing Time=1, ..., Time=5 folders.
            chunk_size (int): Number of files to process in one batch.
            max_value (float): Maximum allowable absolute value for velocity.
            test (bool): If True, only process the first chunk for testing purposes.
        """
        self.base_dir = base_dir
        self.time_dirs = [os.path.join(base_dir, f"Time={t}") for t in range(1, 11)]
        self.chunk_size = chunk_size
        self.max_value = max_value
        self.test = test
        self.data = self._collect_data_in_chunks()

        if len(self.data) == 0:
            raise ValueError("No valid data found in the dataset. Check for NaN values or excessively large velocities.")

    def _parse_filename(self, filename):
        """Extract velocity_x, velocity_y, and time from the filename."""
        try:
            basename = os.path.basename(filename).strip()  # Remove any surrounding spaces
            if not basename.startswith("velocity_x"):
                raise ValueError("Filename does not start with 'velocity_x'")

            parts = basename.split("_")
            if len(parts) < 3:
                raise ValueError("Filename does not contain enough parts")

            vx = float(parts[1].split("=")[1])
            vy = float(parts[3].split("=")[1])
            time_val = int(parts[4].split("=")[1].replace(".csv", ""))
            return vx, vy, time_val
        except (IndexError, ValueError) as e:
            # Log invalid filenames for debugging if needed
            print(f"Invalid filename format: {filename} | Error: {e}")
            return None

    def _read_velocity_csv(self, filepath):
        """Read a CSV file and reshape it to (2, 32, 32)."""
        df = pd.read_csv(filepath)
        arr = df.values.reshape(32, 32, 2)
        arr = np.transpose(arr, (2, 0, 1))  # (2, 32, 32)
        if np.isnan(arr).any() or np.abs(arr).max() > self.max_value:
            return None
        return arr

    def _process_file_group(self, f1):
        """
        Process a group of files corresponding to the same (vx, vy) across Time=1 to Time=5.
        
        Args:
            f1 (str): The file path for Time=1.
        
        Returns:
            tuple: A tuple containing (input_data, label_data), or None if files are missing or invalid.
        """
        parsed = self._parse_filename(f1)
        if parsed is None:
            return None

        vx, vy, _ = parsed

        pattern_t2 = os.path.join(self.time_dirs[1], f"velocity_x={vx}_velocity_y={vy}_time=2.csv")
        pattern_t3 = os.path.join(self.time_dirs[2], f"velocity_x={vx}_velocity_y={vy}_time=3.csv")
        pattern_t4 = os.path.join(self.time_dirs[3], f"velocity_x={vx}_velocity_y={vy}_time=4.csv")
        pattern_t5 = os.path.join(self.time_dirs[4], f"velocity_x={vx}_velocity_y={vy}_time=5.csv")
        pattern_t6 = os.path.join(self.time_dirs[5], f"velocity_x={vx}_velocity_y={vy}_time=6.csv")
        pattern_t7 = os.path.join(self.time_dirs[6], f"velocity_x={vx}_velocity_y={vy}_time=7.csv")
        pattern_t8 = os.path.join(self.time_dirs[7], f"velocity_x={vx}_velocity_y={vy}_time=8.csv")
        pattern_t9 = os.path.join(self.time_dirs[8], f"velocity_x={vx}_velocity_y={vy}_time=9.csv")
        pattern_t10 = os.path.join(self.time_dirs[9], f"velocity_x={vx}_velocity_y={vy}_time=10.csv")
        pattern_t11 = os.path.join(self.time_dirs[10], f"velocity_x={vx}_velocity_y={vy}_time=11.csv")

        if (os.path.exists(pattern_t2) and
            os.path.exists(pattern_t3) and
            os.path.exists(pattern_t4) and
            os.path.exists(pattern_t5) and
            os.path.exists(pattern_t6) and
            os.path.exists(pattern_t7) and
            os.path.exists(pattern_t8) and
            os.path.exists(pattern_t9) and
            os.path.exists(pattern_t10) and
            os.path.exists(pattern_t11)):

            arr1 = self._read_velocity_csv(f1)
            arr2 = self._read_velocity_csv(pattern_t2)
            arr3 = self._read_velocity_csv(pattern_t3)
            arr4 = self._read_velocity_csv(pattern_t4)
            arr5 = self._read_velocity_csv(pattern_t5)
            arr6 = self._read_velocity_csv(pattern_t6)
            arr7 = self._read_velocity_csv(pattern_t7)
            arr8 = self._read_velocity_csv(pattern_t8)
            arr9 = self._read_velocity_csv(pattern_t9)
            arr10 = self._read_velocity_csv(pattern_t10)
            arr11 = self._read_velocity_csv(pattern_t11)

            if any(arr is None for arr in [arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10, arr11]):
                # Skip processing if any file contains NaN or invalid values
                return None

            input_data = np.stack([arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10], axis=1)  # (2, 10, 32, 32)
            label_data = arr11[:, :, :]  # (2, 1, 32, 32)
            label_filename = os.path.basename(pattern_t11)

            return (input_data, label_data, label_filename)

        return None

    def _collect_data_in_chunks(self):
        """
        Collect and process data in chunks to avoid memory overflow.

        Returns:
            list of tuples: Each tuple contains (input_data, label_data).
        """
        time1_files = glob.glob(os.path.join(self.time_dirs[0], "*.csv"))
        total_files = len(time1_files)
        data = []

        for start in range(0, total_files, self.chunk_size):
            chunk_files = time1_files[start:start + self.chunk_size]

            with ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(self._process_file_group, chunk_files), total=len(chunk_files), desc=f"Processing chunk {start // self.chunk_size + 1}/{(total_files + self.chunk_size - 1) // self.chunk_size}"))

            for result in results:
                if result is not None:
                    data.append(result)

            # Stop after processing the first chunk if in test mode
            if self.test:
                print("Test mode: Stopping after the first chunk.")
                break

        return data

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        input_data, label_data, label_filename = self.data[idx]
        return (torch.tensor(input_data, dtype=torch.float32),
                torch.tensor(label_data, dtype=torch.float32),
                label_filename)

if __name__ == "__main__":
    base_directory = "/mnt/data1/tony/train_data_ver11"  # Replace with the actual base directory path
    try:
        dataset = CFDVelocityDataset(base_directory, chunk_size=5000, max_value=1e3, test=True)

        # DataLoader for batching and shuffling
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Example of iterating through the DataLoader
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}: Inputs shape {inputs.shape}, Labels shape {labels.shape}")
            break
            # Inputs shape: (batch_size, 2, 4, 32, 32)
            # Labels shape: (batch_size, 2, 1, 32, 32)
    except ValueError as e:
        print(e)

