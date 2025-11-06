import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from .visualizer import load_eeprom_from_json, build_temperature_frames_single_subpage, load_thermal_subpages

def load_json_file(file_path):
    """Load a JSON file containing thermal data."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_thermal_data_lopo(base_folder, train_users, test_users, random_state=42):
    """
    Load thermal data for LOPO cross-validation.
    Args:
        base_folder (str): Path to the base folder containing person directories
        train_users (list): List of person folder names to use for training
        test_users (list): List of person folder names to use for testing
        random_state (int): Random seed for reproducibility
    Returns:
        X_train, X_test, y_train, y_test, class_names
    """
    sequences = []
    letters = []
    person_ids = []
    user_list = []
    eeprom_list = load_eeprom_from_json("thermal_eeprom.json")
    # Only include folders in train_users or test_users
    person_dirs = sorted([d for d in os.listdir(base_folder)
                         if os.path.isdir(os.path.join(base_folder, d)) and d != 'processed'])

    for person in person_dirs:
        if person not in train_users and person not in test_users:
            continue
        person_folder = os.path.join(base_folder, person)
        letter_dirs = sorted([d for d in os.listdir(person_folder)
                             if os.path.isdir(os.path.join(person_folder, d))])

        for letter in letter_dirs:
            letter_folder = os.path.join(person_folder, letter)
            files = [f for f in os.listdir(letter_folder) if f.endswith('.json')]

            for file in files:
                file_path = os.path.join(letter_folder, file)
                subpages = load_thermal_subpages(file_path)
                data = build_temperature_frames_single_subpage(subpages, eeprom_list)

                if data is not None:
                    thermal_frames = []
                    for frame in data:
                        # Handle new format: [timestamp, [raw_sensor_data]]
                        if not frame or len(frame) < 2:  # Skip empty or malformed frames
                            continue
                        
                        timestamp = frame[0]
                        raw_data = frame[1]
                        
                        # Skip if raw_data is empty
                        if not raw_data:
                            continue
                        
                        thermal_array = np.array(raw_data, dtype=np.float32)
                        thermal_array = thermal_array.reshape((24, 32))  # Reshape to 24x32
                        
                        thermal_frames.append(thermal_array)
                    
                    # Only add sequences with at least 3 frames
                    if len(thermal_frames) >= 3:
                        sequences.append(thermal_frames)
                        letters.append(letter)
                        person_ids.append(person)
                        user_list.append(person)

    class_names = sorted(list(set(letters)))
    label_map = {class_name: i for i, class_name in enumerate(class_names)}
    y_numeric = np.array([label_map[label] for label in letters])

    # Split into train and test sets based on person_ids
    train_indices = [i for i, p in enumerate(person_ids) if p in train_users]
    test_indices = [i for i, p in enumerate(person_ids) if p in test_users]

    X_train = [sequences[i] for i in train_indices]
    X_test = [sequences[i] for i in test_indices]
    y_train = y_numeric[train_indices]
    y_test = y_numeric[test_indices]

    print(f"LOPO Data loaded: {len(X_train)} training sequences, {len(X_test)} test sequences")
    print(f"Classes: {class_names}")

    return X_train, X_test, y_train, y_test, class_names

def pad_sequences(sequences, max_length, pad_value=0):
    """Pad sequences to the same length for batch processing."""
    padded_sequences = []
    mask = []
    
    for seq in sequences:
        seq_length = len(seq)
        if seq_length > max_length:
            # Truncate if sequence is too long
            padded_seq = seq[:max_length]
            seq_mask = [1] * max_length
        else:
            # Pad if sequence is too short
            padding = [np.zeros_like(seq[0]) for _ in range(max_length - seq_length)]
            padded_seq = seq + padding
            seq_mask = [1] * seq_length + [0] * (max_length - seq_length)
        
        padded_sequences.append(np.array(padded_seq))
        mask.append(seq_mask)
    
    return np.array(padded_sequences), np.array(mask)
