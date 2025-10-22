# Thermal Gestures CV Model

A deep learning model for recognizing hand gestures using thermal imaging. This project implements a CNN-LSTM architecture to classify thermal gesture sequences in a Leave-One-Person-Out (LOPO) cross-validation framework.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Configuration](#configuration)
- [Training](#training)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## 🎯 Project Overview

This project processes thermal camera data to recognize and classify hand gesture letters. The model uses a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to capture both spatial and temporal features from thermal sequences.

**Key Features:**
- Thermal sensor data processing (24×32 pixel grids)
- Sequential gesture recognition (5-frame sequences)
- Leave-One-Person-Out cross-validation for robust evaluation
- Data augmentation for improved generalization
- Multi-person gesture classification (5 letter classes)

---

## 🏗️ Architecture

### Primary Model: CNN-LSTM

The main model combines spatial feature extraction with temporal modeling:

#### Input Specifications
| Property | Value |
|----------|-------|
| Thermal Grid Size | 24×32 pixels |
| Sequence Length | 30 frames (via sliding window) |
| Input Channels | 2 (original + motion difference) |
| Input Shape | `(30, 6, 8, 2)` after pooling |
| Max Gesture Length | 110 frames (supported) |

#### CNN Component (TimeDistributed)
Processes each frame independently to extract spatial features:

```
Block 1:
  ├─ Conv2D (32 filters, 3×3 kernel, ReLU)
  ├─ Conv2D (32 filters, 3×3 kernel, ReLU)
  └─ MaxPooling2D (2×2)

Block 2:
  ├─ Conv2D (64 filters, 3×3 kernel, ReLU)
  ├─ Conv2D (64 filters, 3×3 kernel, ReLU)
  └─ MaxPooling2D (2×2)

Flatten → Feature Maps
```

#### LSTM Component
Captures temporal patterns across the sequence:

```
LSTM Layer 1: 128 units (returns sequences)
           ↓
LSTM Layer 2: 64 units (returns final output)
           ↓
Classification Head
```

#### Classification Head
```
Dense (128 units, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (50%)
    ↓
Dense (num_classes, Softmax)
```

#### Compilation Settings
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch Normalization:** Applied after convolutional and first dense layers

### Lightweight Model: Lightweight CNN-LSTM

A reduced model for resource-constrained environments:

| Component | Specification |
|-----------|----------------|
| Conv2D Block 1 | 16 filters, 3×3 kernel, 2×2 MaxPool |
| Conv2D Block 2 | 32 filters, 3×3 kernel, 2×2 MaxPool |
| LSTM | 64 units |
| Dense | 64 units (ReLU) + Dropout (30%) |
| Output | num_classes (Softmax) |

---

## 🔄 Data Pipeline

### Phase 1: Data Loading (`src/data/loader.py`)

**Input Format:**
- JSON files containing thermal frame sequences
- Each frame includes: timestamp and raw sensor values (768 values)
- Organized by: `Person > Letter > JSON files`

**Processing:**
```
Raw JSON Data
    ↓
Frame Validation (≥ 768 sensor values)
    ↓
Reshape to 24×32 grid
    ↓
Normalize (0-65535 → 0-1 range)
    ↓
Sequence Filtering (≥ 3 frames per sequence)
    ↓
LOPO Split (person-based stratification)
```

**Output:**
- Training sequences: Multiple persons' data
- Test sequences: One holdout person
- Class names: Alphabetically sorted gesture labels

---

### Phase 2: Preprocessing (`src/data/preprocessing.py`)

Each sequence undergoes the following preprocessing pipeline:

#### Step 1: Sequence Normalization
- **Method:** Min-max scaling across entire sequence
- **Purpose:** Consistent thermal value scaling independent of sequence
- **Output Range:** [0, 1]

#### Step 2: Per-Frame Normalization (if disabled above)
- **Method:** Individual frame min-max scaling
- **Fallback:** When sequence normalization is disabled

#### Step 3: Noise Reduction
- **Filter:** Gaussian filter
- **Sigma Parameter:** 0.5
- **Purpose:** Smooth thermal noise while preserving gesture boundaries

#### Step 4: Edge Enhancement
- **Method:** Sobel operator (X and Y gradients)
- **Edge Weight:** 0.3 (30%)
- **Formula:** `Enhanced = (0.7 × Original) + (0.3 × Edges)`
- **Purpose:** Emphasize gesture contours and boundaries

#### Step 5: Motion Differencing
- **Calculation:** Frame differences between consecutive frames
- **Purpose:** Capture temporal motion patterns
- **Zero-frame:** First frame difference set to zero (maintains length)

#### Step 6: Channel Stacking
- **Output:** 2-channel tensors
- **Channel 1:** Original (enhanced) thermal frame
- **Channel 2:** Motion difference frame

#### Step 7: Sliding Window Processing
- **Purpose:** Handle long gesture sequences (>30 frames, up to 110 frames max)
- **Window Size:** 30 frames (SEQUENCE_LENGTH)
- **Stride:** 10 frames (WINDOW_STRIDE) - creates 33% overlap
- **Method:** Creates multiple overlapping windows from each long sequence
- **Example:** 110-frame gesture → 9 overlapping 30-frame windows
  
```
Original sequence:     [F0  F1  F2  ... F109]  (110 frames total)
                       ⟵───────────────────⟶
                        Sliding Window of 30 frames
                       
Window 1:  [F0-F29]    (frames 0-29)
Window 2:  [F10-F39]   (frames 10-39)
Window 3:  [F20-F49]   (frames 20-49)
...
Window 9:  [F80-F109]  (frames 80-109)
```

- **Benefits:**
  - Captures full gesture motion across multiple samples
  - Significantly increases training data without additional recording
  - Preserves temporal continuity via overlapping windows
  - Enables LSTM to learn complete gesture dynamics

#### Step 8: Sequence Padding/Truncation
- **Target Length:** 30 frames (SEQUENCE_LENGTH)
- **Padding:** Sequences < 30 frames padded with zero tensors
- **Padding Value:** 0 (black/cold regions)
- **No truncation:** Sliding windows ensure all sequences are ≤30 frames

---

### Phase 3: Data Augmentation (Training Only)

Augmentation applied **exclusively to training data** to improve model generalization:

#### Rotation Augmentation
- **Angles:** -10°, +10°
- **Method:** `scipy.ndimage.rotate` with `reshape=False`
- **Applied to:** Each frame in sequence

#### Spatial Shifts
- **Shift Amounts:** ±2 pixels in both X and Y directions
- **Configurations:** (2,0), (-2,0), (0,2), (0,-2)
- **Method:** `scipy.ndimage.shift`
- **Purpose:** Model robustness to small hand position variations

#### Gaussian Noise Injection
- **Noise Scale:** 0.03 (3% standard deviation)
- **Distribution:** Normal distribution N(0, 0.03)
- **Clipping:** Output clipped to [0, 1] range
- **Purpose:** Simulate thermal sensor noise variations

#### Data Expansion

**Combined Effect of Sliding Windows + Augmentation:**
- **Sliding Window Factor:** Each long gesture (e.g., 110 frames) → ~9 overlapping windows
- **Augmentation Factor:** Each window → 4 variants (original + 3 augmented)
- **Total Expansion:** 9 windows × 4 augmentations = 36× per original long gesture
- **Practical Example:** 10 original 110-frame training sequences → 360 training samples
- **Training Labels:** Automatically expanded to match processed sequences
- **Test Data:** Sliding windows applied, but no augmentation

**Data Expansion Formula:**
$$\text{Processed Samples} = \text{Original Samples} \times \text{Windows per Gesture} \times \text{Augmentations per Window}$$

For your dataset:
- Maximum gesture length: 110 frames
- Window size: 30 frames  
- Window stride: 10 frames
- Windows per 110-frame gesture: ~9
- Training augmentations: 4× per window
- **Effective expansion: 9-36× depending on gesture length**

---

## ⚙️ Configuration

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 32 | Samples per gradient update |
| **Epochs** | 20 | Maximum training iterations |
| **Sequence Length** | 30 | Frames per thermal sequence (via sliding window) |
| **Window Stride** | 10 | Frame offset between sliding windows |
| **Max Gesture Frames** | 110 | Maximum gesture duration supported |
| **Number of Classes** | 5 | Gesture letters to classify |
| **Random Seed** | 42 | Reproducibility seed |
| **Augmentation** | Enabled | Data augmentation for training |
| **Sliding Window** | Enabled | For sequences > 30 frames |

### Optimizer Settings

| Setting | Value |
|---------|-------|
| Algorithm | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |

### Callbacks Configuration

#### Early Stopping
- **Monitor:** Validation accuracy
- **Patience:** 15 epochs
- **Action:** Stop training if no improvement
- **Recovery:** Restore best weights

#### Learning Rate Reduction
- **Monitor:** Validation loss
- **Patience:** 8 epochs
- **Reduction Factor:** 0.2 (reduces LR by 80%)
- **Minimum LR:** 1e-6

#### TensorBoard Logging
- **Histogram Frequency:** 1 (log every epoch)
- **Log Directory:** `model_dir/logs/`

#### Custom Test Accuracy Tracker
- Tracks test accuracy during training
- Monitors model generalization in real-time

python
# 1. Load data with LOPO split
X_train, X_test, y_train, y_test, class_names = load_thermal_data_lopo(
    DATA_DIR,
    train_users=train_users,
    test_users=[test_user],
    random_state=42
)

# 2. Preprocess with augmentation
X_train_processed, X_test_processed, y_train, y_test = prepare_data_for_training(
    X_train, X_test, y_train, y_test,
    batch_size=32,
    max_sequence_length=5,
    num_classes=5,
    random_state=42,
    use_augmentation=True
)

# 3. Create and train model
model = create_cnn_lstm_model(input_shape, num_classes)
history = model.fit(
    X_train_processed, y_train,
    batch_size=32,
    epochs=20,
    callbacks=[early_stopping, reduce_lr, test_tracker, tensorboard],
    verbose=1
)

# 4. Evaluate and save
y_pred, cm = evaluate_model(model, X_test_processed, y_test, class_names)
save_model(model, model_path)
```



---


---

## 📁 Project Structure

```
Thermal_Gestures_CV-model/
├── README.md                          # This file
├── train.py                           # Main training script
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Data loading & LOPO splitting
│   │   └── preprocessing.py           # Preprocessing & augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm.py                # Model architectures
│   │   ├── utils.py                   # Training utilities & callbacks
│   │   └── thermal_letter_model_LOPO_{Person}_{Timestamp}/
│   │       ├── *_final.h5             # Final models
│   │       ├── class_names.npy        # Class mappings
│   │       └── logs/                  # TensorBoard logs
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py                 # Visualization utilities
│
├── Data_anotation/
│   └── JSON_creater.py                # Annotation tools
│
└── training_history.png               # Training visualizations
```

---



### Configuration in train.py

```python
if __name__ == "__main__":
    RANDOM_SEED = 42
    DATA_DIR = "D:\\Data_collecn\\micro-gestures\\data\\Labelled_data\\"
    MODEL_DIR = "src\\models"
    EPOCHS = 20
    SEQUENCE_LENGTH = 30              # Increased from 5 to handle long gestures (up to 110 frames)
    BATCH_SIZE = 32
    NUM_CLASSES = 5
    WINDOW_STRIDE = 10                # Sliding window stride (10-frame overlap creates 33% overlap)
    use_augmentation = True
```

#### Tuning Sliding Window Parameters

**For your use case (gestures up to 110 frames):**

| Scenario | SEQUENCE_LENGTH | WINDOW_STRIDE | Windows per 110-frame | Data Expansion |
|----------|-----------------|---------------|----------------------|-----------------|
| Conservative (small data) | 50 | 15 | 5 | 20× |
| **Balanced (recommended)** | **30** | **10** | **9** | **36×** |
| Aggressive (large data) | 20 | 5 | 18 | 72× |

**Considerations:**
- **Smaller windows** → more training samples, but less temporal context per sample
- **Larger windows** → fewer samples, but richer temporal patterns
- **Smaller stride** → more overlap, smoother transitions, but more correlated data
- **Larger stride** → less overlap, more diverse windows, but potential gaps

To adjust, modify in `train.py`:
```python
SEQUENCE_LENGTH = 30    # Change window size (frames)
WINDOW_STRIDE = 10      # Change stride (frames)
```

---






