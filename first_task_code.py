import os
import sys
import subprocess

# -----------------------------
# ENVIRONMENT SETTINGS
# -----------------------------
# 1. Force protobuf to use python implementation (helps stability in some envs)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# 2. Suppress oneDNN custom operations logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -----------------------------
# IMPORTS WITH DIAGNOSTICS
# -----------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("\nCRITICAL: TensorFlow is not installed.")
    print("Run: pip install tensorflow")
    sys.exit(1)
except TypeError as e:
    # Catch the specific protobuf/tensorflow compatibility error (unhashable type: list)
    import traceback
    trace = traceback.format_exc()
    if "unhashable type: 'list'" in trace:
        print("\n" + "="*60)
        print("CRITICAL ENVIRONMENT ERROR: Dependency Conflict")
        print("="*60)
        print("Your TensorFlow and Protobuf versions are incompatible.")
        print("The previous repair installed too new of a version (6.x).")
        print("We need to force install Protobuf < 4.21.0.")
        
        try:
            choice = input("\nWould you like this script to auto-repair the environment? (y/n): ").strip().lower()
            if choice == 'y':
                print("\n[1/2] Uninstalling conflicting libraries...")
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "onnx", "onnx2tf", "sng4onnx", "protobuf"])
                
                print("\n[2/2] Installing TensorFlow with compatible Protobuf...")
                # We specifically pin protobuf<4.21.0 to fix the unhashable type error
                subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf<4.21.0", "tensorflow"])
                
                print("\n" + "="*60)
                print("REPAIR COMPLETE. Please run this script again.")
                print("="*60 + "\n")
                sys.exit(0)
            else:
                print("\nManual Fix Required:")
                print("pip uninstall -y protobuf && pip install \"protobuf<4.21.0\" tensorflow")
                sys.exit(1)
        except Exception as fix_err:
            print(f"\nAuto-fix failed: {fix_err}")
            print("Please run manually: pip uninstall -y protobuf && pip install \"protobuf<4.21.0\" tensorflow")
            sys.exit(1)
            
    raise e

import json
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
PERSON_LABEL_FILES = {
    'person1': 'aadarsh_thermal_first.txt',
    'person2': 'amay_thermal_first.txt',
    'person3': 'samarth_thermal_first.txt',
    'person4': 'kshitij_thermal_first.txt',
    'person5': 'harsh_thermal_first.txt',
    'person6': 'swarnali_thermal_first.txt',
    'person7': 'aishwarya_thermal_first.txt',
}

# ---------------- Data Loading Helper ----------------
def get_data_paths_combined(base_dir, json_name='thermal.json'):
    """
    Reads text files and JSONs to create a list of (data_array, label) tuples.
    """
    data_by_person = {}
    for person, txt_file in PERSON_LABEL_FILES.items():
        person_dir = os.path.join(base_dir, person)
        json_path = os.path.join(person_dir, json_name)
        txt_path = os.path.join(person_dir, txt_file)
        
        if not (os.path.isdir(person_dir) and os.path.isfile(json_path) and os.path.isfile(txt_path)):
            print(f"[SKIP] missing for {person}")
            continue

        # Load Labels
        label_map = {}
        with open(txt_path, 'r') as f:
            for line in f:
                if ',' not in line: continue
                parts = line.strip().split(',', 1)
                if len(parts) < 2: continue
                lbl_str, ts_str = parts
                try:
                    label_map[int(ts_str)] = int(lbl_str)
                except ValueError:
                    continue

        # Load JSON Data
        try:
            with open(json_path, 'r') as f:
                json_entries = json.load(f)
        except Exception as e:
            print(f"[ERROR] failed to load JSON for {person}: {e}")
            continue

        # Match Timestamps
        json_map = {}
        for entry in json_entries:
            if isinstance(entry, list) and len(entry) == 2:
                ts = entry[0]
                vals = entry[1]
                arr = np.array(vals, dtype=np.float32)
                if arr.size > 0 and not np.isnan(arr).any():
                    json_map[int(ts)] = arr

        paired = []
        for ts, lbl in label_map.items():
            if ts in json_map:
                paired.append((json_map[ts], lbl))

        if paired:
            data_by_person[person] = paired

    all_entries = []
    for lst in data_by_person.values():
        all_entries.extend(lst)
    return all_entries

# ---------------- TF Data Pipeline ----------------
def preprocess_sample(sample_arr):
    """
    Converts input to (Height, Width, Channels) for TensorFlow.
    Performs standardization.
    """
    arr = np.array(sample_arr, dtype=np.float32)
    
    # Normalization (Mean=0, Std=1 per sample)
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 0:
        arr = (arr - mean) / std
    else:
        arr = arr - mean

    # Reshape logic to match TF 'channels_last' (H, W, C)
    # Case 1: (C, W) -> Transpose to (W, C), expand to (1, W, C)
    if arr.ndim == 2:
        arr = np.transpose(arr, (1, 0)) # Now (W, C)
        arr = np.expand_dims(arr, axis=0) # Now (1, W, C)
        
    # Case 2: (C, H, W) -> Transpose to (H, W, C)
    elif arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected shape: {arr.shape}")
        
    return arr

def create_tf_dataset(entries, batch_size=32, shuffle=True):
    X_raw = [x[0] for x in entries]
    y_raw = [x[1] for x in entries]
    
    print("Preprocessing data for TensorFlow...")
    X_processed = np.array([preprocess_sample(x) for x in X_raw])
    y_processed = np.array(y_raw, dtype=np.int32)
    
    print(f"Processed Data Shape: {X_processed.shape}")
    
    ds = tf.data.Dataset.from_tensor_slices((X_processed, y_processed))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(entries))
    ds = ds.batch(batch_size)
    return ds, X_processed.shape[1:] 

# ---------------- Model Definition ----------------
def create_thermal_cnn(input_shape, num_classes=2):
    """
    Replicates the PyTorch architecture in Keras.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1: Conv -> BN -> ReLU -> Pool
        layers.Conv2D(8, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(1, 2)),
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        layers.Conv2D(16, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(1, 2)),
        
        # Flatten & Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ---------------- Main Pipeline ----------------
def train_and_save_h5(base_dir='lopocv_first',
                      out_dir='deploy_output',
                      json_name='thermal.json',
                      epochs=20,          # Set max to 20, let EarlyStopping decide
                      batch_size=64,      # Increased to 64 for stability
                      lr=1e-3):
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    all_entries = get_data_paths_combined(base_dir, json_name=json_name)
    if not all_entries:
        raise RuntimeError("No data found.")
    print(f"Total samples: {len(all_entries)}")
    
    # 2. Prepare Datasets
    split_idx = int(0.8 * len(all_entries))
    train_entries = all_entries[:split_idx]
    val_entries = all_entries[split_idx:]
    
    # Create datasets with new batch size
    train_ds, input_shape = create_tf_dataset(train_entries, batch_size=batch_size, shuffle=True)
    val_ds, _ = create_tf_dataset(val_entries, batch_size=batch_size, shuffle=False)
    
    print(f"Input Shape (H, W, C): {input_shape}")
    
    # 3. Build Model
    model = create_thermal_cnn(input_shape, num_classes=2)
    
    # 4. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # ---------------------------------------------------------
    # NEW: Callbacks for Early Stopping
    # ---------------------------------------------------------
    callbacks = [
        # Stop if val_loss doesn't improve for 4 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True, # CRITICAL: This ensures we save the version from Epoch 2, not Epoch 20
            verbose=1
        ),
        # Reduce Learning Rate if we get stuck (helps squeeze out extra accuracy)
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    
    # 5. Train
    print("\nStarting TensorFlow training (with Early Stopping)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks, # Add the callbacks here
        verbose=1
    )
    
    # 6. Save directly to H5
    h5_path = os.path.join(out_dir, "final_model.h5")
    model.save(h5_path, save_format='h5')
    print(f"\nSUCCESS: Best model saved to: {h5_path}")
    
    return h5_path

if __name__ == "__main__":
    try:
        # Using optimized settings
        h5_file = train_and_save_h5(
            base_dir='lopocv_first',
            out_dir='deploy_output',
            json_name='thermal.json',
            epochs=20,       # Max cap
            batch_size= 10   # Optimized
        )
        print(f"Done. File: {h5_file}")
    except Exception as e:
        print(f"Script failed: {e}")