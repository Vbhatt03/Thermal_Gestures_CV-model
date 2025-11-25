import os
import sys
import subprocess

# -----------------------------
# ENVIRONMENT SETTINGS
# -----------------------------
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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
    # Catch protobuf conflict
    import traceback
    trace = traceback.format_exc()
    if "unhashable type: 'list'" in trace:
        print("\n" + "="*60)
        print("CRITICAL ENVIRONMENT ERROR: Dependency Conflict")
        print("="*60)
        print("Run this manually to fix:")
        print("pip uninstall -y protobuf && pip install \"protobuf<4.21.0\" tensorflow")
        sys.exit(1)
    raise e

import json
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
PERSON_LABEL_FILES = {
    'person1': 'aadarsh_thermal_2.txt',
    'person2': 'amay_thermal_2.txt',
    'person3': 'samarth_thermal_2.txt',
    'person4': 'kshitij_thermal_2.txt',
    'person5': 'swarnali_thermal_2.txt',
    'person6': 'aishwarya_thermal_2.txt',
}

# ---------------- Data Loading Helper ----------------
def get_data_paths_combined(base_dir, json_name='thermal.json'):
    data_by_person = {}
    for person, txt_file in PERSON_LABEL_FILES.items():
        person_dir = os.path.join(base_dir, person)
        json_path = os.path.join(person_dir, json_name)
        txt_path = os.path.join(person_dir, txt_file)
        
        if not (os.path.isdir(person_dir) and os.path.isfile(json_path) and os.path.isfile(txt_path)):
            print(f"[SKIP] missing for {person}")
            continue

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

        try:
            with open(json_path, 'r') as f:
                json_entries = json.load(f)
        except Exception as e:
            print(f"[ERROR] failed to load JSON for {person}: {e}")
            continue

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
    arr = np.array(sample_arr, dtype=np.float32)
    
    # Normalization
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 0:
        arr = (arr - mean) / std
    else:
        arr = arr - mean

    # Reshape logic (H, W, C)
    if arr.ndim == 2:
        arr = np.transpose(arr, (1, 0))
        arr = np.expand_dims(arr, axis=0)
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
    
    # Check max label to warn user if mismatch occurs
    max_label = np.max(y_processed)
    print(f"Max label found in data: {max_label} (Implies at least {max_label+1} classes)")

    ds = tf.data.Dataset.from_tensor_slices((X_processed, y_processed))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(entries))
    ds = ds.batch(batch_size)
    return ds, X_processed.shape[1:] 

# ---------------- Model Definition ----------------
def create_thermal_cnn(input_shape, num_classes=5):
    """
    Updated for 5 classes (0, 1, 2, 3, 4).
    Includes Improved Augmentation & L2 Regularization.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # --- IMPROVED AUGMENTATION ---
        # 1. Flip horizontal only (Vertical flip confuses thermal people detection)
        layers.RandomFlip("horizontal"),
        # 2. Reduced rotation slightly
        layers.RandomRotation(0.1),
        # 3. Add Translation (Shifting image slightly helps generalization)
        layers.RandomTranslation(0.1, 0.1),
        # 4. Add Zoom
        layers.RandomZoom(0.1),
        
        # Conv Block 1
        layers.Conv2D(8, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(1, 2)),
        
        # Conv Block 2
        layers.Conv2D(16, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(1, 2)),
        
        # Dense Layers
        layers.Flatten(),
        # Add L2 Regularization to penalize overfitting
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        
        # Output Layer: num_classes=5 for labels 0-4
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ---------------- Main Pipeline ----------------
def train_and_save_h5(base_dir='lopocv_second',
                      out_dir='deploy_output',
                      json_name='thermal.json',
                      epochs=50,          # Increased to 50
                      batch_size=32,      # Standard batch size for better gradients
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
    
    train_ds, input_shape = create_tf_dataset(train_entries, batch_size=batch_size, shuffle=True)
    val_ds, _ = create_tf_dataset(val_entries, batch_size=batch_size, shuffle=False)
    
    print(f"Input Shape (H, W, C): {input_shape}")
    
    # 3. Build Model (Fixed for 5 classes)
    model = create_thermal_cnn(input_shape, num_classes=5)
    model.summary()
    
    # 4. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,                  # Increased patience slightly
            restore_best_weights=True, 
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]
    
    # 5. Train
    print("\nStarting TensorFlow training (5 Classes)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks, 
        verbose=1
    )
    
    # 6. Save H5
    h5_path = os.path.join(out_dir, "final_model_5class.h5")
    model.save(h5_path, save_format='h5')
    print(f"\nSUCCESS: Best model saved to: {h5_path}")
    
    return h5_path

if __name__ == "__main__":
    try:
        h5_file = train_and_save_h5(
            base_dir='lopocv_second',
            out_dir='deploy_output',
            json_name='thermal.json',
            epochs=50,
            batch_size=8
        )
        print(f"Done. File: {h5_file}")
    except Exception as e:
        print(f"Script failed: {e}")