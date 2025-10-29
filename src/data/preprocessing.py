import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.ndimage import rotate, shift, binary_dilation

def temporal_downsample_to_target(sequence, target_length=100):
    """
    Resample sequence to exact target_length.
    Preserves temporal structure regardless of original length.
    
    Args:
        sequence: List of frames
        target_length: Target number of frames (default 100)
    
    Returns:
        List of target_length frames
    """
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    # Linear index interpolation to sample exact frames
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return [sequence[int(i)] for i in indices]


def subtract_background(sequence, background_percentile=20):
    """
    Subtract background thermal signature to isolate hand.
    
    The hand holding a marker is the HOTTEST part.
    By subtracting cool regions (background), we focus on the hand.
    
    Args:
        sequence: List of raw thermal frames
        background_percentile: Percentile for background estimation (lower = cooler regions)
    
    Returns:
        List of background-subtracted frames
    """
    # Estimate background (coolest regions across entire sequence)
    all_values = np.concatenate([f.flatten() for f in sequence])
    background_temp = np.percentile(all_values, background_percentile)
    
    # Subtract background from each frame
    bg_subtracted = []
    for frame in sequence:
        frame = frame.astype(np.float32)
        # Subtract background
        diff = frame - background_temp
        # Clip to prevent negative values (keep only positive differences)
        diff = np.maximum(diff, 0)
        bg_subtracted.append(diff)
    
    return bg_subtracted


def extract_hand_region(thermal_frame, percentile_threshold=70):
    """
    Extract hand/marker region by detecting hot spots.
    
    Assumes hand (hottest region) is the target.
    Masks out background to focus model attention.
    
    Args:
        thermal_frame: Raw or processed thermal frame
        percentile_threshold: Percentile for hot region detection (higher = hotter regions only)
    
    Returns:
        Masked frame with hand region highlighted, background suppressed
    """
    thermal_frame = thermal_frame.astype(np.float32)
    
    # Find hot threshold (hand is hottest)
    threshold = np.percentile(thermal_frame, percentile_threshold)
    
    # Create binary mask of hot regions
    mask = thermal_frame > threshold
    
    # Dilate mask slightly to include margins around hand
    mask = binary_dilation(mask, iterations=2)
    
    # Get background value (coolest part)
    background_value = np.percentile(thermal_frame, 10)
    
    # Apply mask: keep hand region, suppress background
    masked_frame = np.where(mask, thermal_frame, background_value)
    
    return masked_frame.astype(np.float32)


def normalize_per_frame_percentile(sequence):
    """
    Normalize each frame independently using robust percentiles.
    BEST approach for thermal gesture recognition.
    
    Preserves thermal gradients and is robust to outliers.
    
    Args:
        sequence: List of thermal frames (24x32 arrays)
    
    Returns:
        List of normalized frames [0, 1]
    """
    normalized_sequence = []
    
    for frame in sequence:
        frame = frame.astype(np.float32)
        
        # Robust percentile-based normalization per frame
        p5 = np.percentile(frame, 5)    # Background temperature
        p95 = np.percentile(frame, 95)  # Hand/gesture temperature
        
        # Normalize
        if p95 <= p5:  # Edge case: uniform frame
            normalized_frame = np.ones_like(frame, dtype=np.float32) * 0.5
        else:
            normalized_frame = (frame - p5) / (p95 - p5)
            normalized_frame = np.clip(normalized_frame, 0, 1)
        
        normalized_sequence.append(normalized_frame.astype(np.float32))
    
    return normalized_sequence

def normalize_thermal_data(thermal_array):
    """Normalize thermal data to [0, 1] range (legacy - kept for compatibility)."""
    thermal_array = thermal_array.astype(np.float32)
    min_val = np.min(thermal_array)
    max_val = np.max(thermal_array)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(thermal_array, dtype=np.float32)
    
    normalized = (thermal_array - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1).astype(np.float32)

def normalize_per_sequence(sequence):
    """Normalize an entire sequence for consistent scaling (legacy - NOT RECOMMENDED)."""
    flat_seq = np.concatenate([frame.flatten() for frame in sequence])
    min_val = np.min(flat_seq)
    max_val = np.max(flat_seq)
    
    # Avoid division by zero
    if max_val == min_val:
        return [np.zeros_like(frame) for frame in sequence]
    
    return [(frame - min_val) / (max_val - min_val) for frame in sequence]

def calculate_motion_features_raw(sequence):
    """
    Calculate motion features from RAW thermal data BEFORE normalization.
    
    This is the CORRECT approach - motion is calculated from raw values
    to preserve motion magnitudes and prevent scale collapse.
    
    Args:
        sequence: List of raw thermal frames (not normalized yet)
    
    Returns:
        List of frames with 2 channels: [raw_thermal, temporal_gradient]
        Shape of each frame: (24, 32, 2)
    """
    motion_frames = []
    
    for i, frame in enumerate(sequence):
        frame = frame.astype(np.float32)
        
        if i == 0:
            # First frame: no previous frame, use zero gradient
            gradient = np.zeros_like(frame, dtype=np.float32)
        else:
            # Calculate gradient: current - previous
            # This preserves motion magnitude and direction
            prev_frame = sequence[i-1].astype(np.float32)
            gradient = frame - prev_frame
        
        # Stack as channels: [raw_thermal, motion_gradient]
        stacked = np.stack([frame, gradient], axis=-1)
        motion_frames.append(stacked)
    
    return motion_frames

def normalize_motion_frames(motion_sequence):
    """
    Normalize motion frames with DIFFERENT strategies per channel.
    
    Channel 0 (Thermal): Non-zero-centered → Percentile-based normalization
    Channel 1 (Gradient): Zero-centered → Robust z-score normalization
    
    Why different methods?
    - Thermal values cluster around a non-zero center (65xxx range raw, or peaks after BG subtract)
      Percentiles work well here
    - Gradient is mostly zeros with occasional spikes (motion events)
      Percentiles fail (tiny denominator); z-score with MAD is robust
    
    Args:
        motion_sequence: List of frames with shape (24, 32, 2)
                        Channel 0: raw thermal
                        Channel 1: temporal gradient (frame - prev_frame)
    
    Returns:
        List of properly normalized frames with shape (24, 32, 2)
    """
    normalized = []
    
    # PASS 1: Compute global statistics for gradient channel (zero-centered)
    all_gradients = []
    for frame in motion_sequence:
        if frame.shape[-1] >= 2:
            all_gradients.append(frame[..., 1].flatten())
    
    if all_gradients:
        all_gradients = np.concatenate(all_gradients)
        # Use MAD (Median Absolute Deviation) for robust std estimation
        # MAD is more stable than percentiles for zero-centered signals
        gradient_median = np.median(all_gradients)
        gradient_mad = np.median(np.abs(all_gradients - gradient_median))
        # Convert MAD to std equivalent: std ≈ 1.4826 * MAD
        gradient_std = max(1.4826 * gradient_mad, 1e-6)
    else:
        gradient_std = 1e-6
    
    # PASS 2: Normalize each frame using channel-specific methods
    for frame in motion_sequence:
        normalized_frame = np.zeros_like(frame, dtype=np.float32)
        
        # ═══════════════════════════════════════════════════════════
        # CHANNEL 0: Thermal (Non-zero-centered)
        # Method: Percentile-based normalization
        # ═══════════════════════════════════════════════════════════
        if frame.shape[-1] >= 1:
            thermal_channel = frame[..., 0]
            p5 = np.percentile(thermal_channel, 5)
            p95 = np.percentile(thermal_channel, 95)
            
            if p95 > p5:
                # Scale [p5, p95] → [0, 1]
                normalized_thermal = (thermal_channel - p5) / (p95 - p5)
                normalized_thermal = np.clip(normalized_thermal, 0, 1)
            else:
                # Edge case: uniform frame (all same thermal value)
                normalized_thermal = np.ones_like(thermal_channel) * 0.5
            
            normalized_frame[..., 0] = normalized_thermal
        
        # ═══════════════════════════════════════════════════════════
        # CHANNEL 1: Gradient (Zero-centered)
        # Method: Robust z-score normalization (using MAD)
        # ═══════════════════════════════════════════════════════════
        if frame.shape[-1] >= 2:
            gradient_channel = frame[..., 1]
            
            # Z-score: (x - median) / MAD_std
            # Median and MAD are more robust than mean/std for this signal
            normalized_gradient = (gradient_channel - gradient_median) / gradient_std
            
            # Soft clip to [-3, 3] standard deviations (natural bounds)
            # Prevents extreme outliers from dominating CNN attention
            normalized_gradient = np.clip(normalized_gradient, -3, 3)
            
            # Scale to approximately [-0.5, 0.5]
            # Balances with thermal channel [0, 1] so neither dominates
            normalized_gradient = normalized_gradient / 6.0
            
            normalized_frame[..., 1] = normalized_gradient
        
        normalized.append(normalized_frame)
    
    return normalized

def calculate_frame_differences(sequence):
    """Calculate frame differences to highlight motion (LEGACY - NOT RECOMMENDED)."""
    diffs = []
    for i in range(1, len(sequence)):
        diff = sequence[i] - sequence[i-1]
        diffs.append(diff)
    
    # Add a zero diff for first frame to maintain sequence length
    diffs = [np.zeros_like(sequence[0])] + diffs
    return diffs

def normalize_per_user(thermal_array):
    """Normalize considering user-specific temperature ranges."""
    # Find likely background temperature (10th percentile)
    background_temp = np.percentile(thermal_array, 10)
    
    # Find likely hand temperature (90th percentile)
    hand_temp = np.percentile(thermal_array, 90)
    
    # Normalize relative to temperature difference
    if hand_temp == background_temp:
        return np.zeros_like(thermal_array)
    
    normalized = (thermal_array - background_temp) / (hand_temp - background_temp)
    normalized = np.clip(normalized, 0, 1)
    
    return normalized

# def reduce_noise(thermal_array, sigma=0.2):
#     """Apply Gaussian filter to reduce noise."""
#     return ndimage.gaussian_filter(thermal_array, sigma=sigma)

def enhance_edges(thermal_array, weight=0.3):
    """Enhance edges in thermal image."""
    edges_x = ndimage.sobel(thermal_array, axis=0)
    edges_y = ndimage.sobel(thermal_array, axis=1)
    edges = np.hypot(edges_x, edges_y)
    edges = edges / edges.max() if edges.max() > 0 else edges
    
    # Combine original with edges
    enhanced = (1 - weight) * thermal_array + weight * edges
    return enhanced

def preprocess_sequence(sequence, normalize_sequence=True, 
                       target_length=100, hand_focused=True):
    """
    Apply preprocessing to a sequence of thermal frames.
    THERMAL ONLY - No gradient channel
    
    Preprocessing order:
    1. Temporal resampling to 100 frames
    2. Background subtraction (percentile=20)
    3. Hand region extraction (percentile_threshold=70)
    4. Per-frame percentile normalization
    
    Args:
        sequence: List of raw thermal frames
        normalize_sequence: Whether to normalize (recommended: True)
        target_length: Target sequence length (default: 100)
        hand_focused: Whether to focus on hand/marker region (recommended: True)
    
    Returns:
        List of preprocessed thermal frames with shape (100, 24, 32)
        Values: Normalized to 0-1 range
    """
    
    # STEP 1: Temporal resampling to consistent length
    sequence = temporal_downsample_to_target(sequence, target_length=target_length)
    
    # STEP 2: Hand-focused preprocessing
    if hand_focused:
        # Subtract background to remove static thermal variations
        sequence = subtract_background(sequence, background_percentile=20)
        
        # Extract hand region (mask out background)
        sequence = [extract_hand_region(frame, percentile_threshold=70) for frame in sequence]
    
    # STEP 3: Normalize thermal data
    if normalize_sequence:
        processed_sequence = normalize_per_frame_percentile(sequence)
    else:
        # Still clip and convert to float32
        processed_sequence = []
        for frame in sequence:
            frame = np.clip(frame.astype(np.float32), 0, 1)
            processed_sequence.append(frame)
    
    return processed_sequence

def augment_sequence(sequence, max_augmentations=3):
    """Apply data augmentation to a sequence."""
    augmented_sequences = [sequence]  # Start with original sequence
    
    # Apply random rotation
    for angle in [-10, 10]:
        aug_seq = [rotate(frame, angle, reshape=False) for frame in sequence]
        augmented_sequences.append(aug_seq)
    
    # Apply random shifts (2D: [dy, dx] for 24x32 frames)
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        aug_seq = [shift(frame, [dy, dx]) for frame in sequence]
        augmented_sequences.append(aug_seq)
    
    # Add slight gaussian noise
    noise_scale = 0.03
    noise_seq = [np.clip(frame + np.random.normal(0, noise_scale, frame.shape), 0, 1) 
                for frame in sequence]
    augmented_sequences.append(noise_seq)
    
    # Return original + augmented sequences (limited by max_augmentations)
    return augmented_sequences[:max_augmentations + 1]

def preprocess_dataset(sequences, max_length, normalize_sequence=True, 
                      use_augmentation=False, hand_focused=True):
    """
    Preprocess an entire dataset (THERMAL ONLY - no gradient channel).
    
    Args:
        sequences: List of thermal frame sequences
        max_length: Target sequence length
        normalize_sequence: Whether to normalize
        use_augmentation: Whether to augment training data
        hand_focused: Whether to focus on hand/marker region (recommended: True)
    """
    processed_sequences = []
    
    for sequence in sequences:
        # Apply preprocessing (including downsampling to max_length)
        # THERMAL ONLY - no gradient channel
        processed_seq = preprocess_sequence(
            sequence, 
            normalize_sequence=normalize_sequence,
            target_length=max_length,  # Use max_length as target
            hand_focused=hand_focused  # Enable hand-focused preprocessing
        )
        
        # Apply augmentation if specified
        if use_augmentation:
            aug_sequences = augment_sequence(processed_seq)
            processed_sequences.extend(aug_sequences)
        else:
            processed_sequences.append(processed_seq)
    
    # NO PADDING NEEDED ANYMORE! All sequences are already max_length
    padded_sequences = []
    for seq in processed_sequences:
        # Sequence should already be exactly max_length, but add safety check
        if len(seq) != max_length:
            print(f"Warning: sequence length {len(seq)} != {max_length}")
        padded_sequences.append(np.array(seq))
    
    return np.array(padded_sequences)

def prepare_data_for_training(X_train, X_test, y_train, y_test, batch_size,
                              max_sequence_length, num_classes, random_state,
                              use_augmentation):
    """
    Prepare data for training with hand-focused preprocessing (THERMAL ONLY).
    """
    # Preprocess training data (with hand-focused preprocessing, THERMAL ONLY)
    X_train_processed = preprocess_dataset(
        X_train, 
        max_sequence_length,
        normalize_sequence=True,
        use_augmentation=use_augmentation,  # Use the parameter, not hardcoded True
        hand_focused=True  # Enable hand-focused preprocessing
    )
    
    # Preprocess test data (no augmentation for test data, but hand-focused)
    X_test_processed = preprocess_dataset(
        X_test, 
        max_sequence_length,
        normalize_sequence=True,
        use_augmentation=False,
        hand_focused=True  # Enable hand-focused preprocessing
    )
    
    # If augmentation was used, we need to expand y_train
    if use_augmentation:
        aug_factor = 4  # Original + 3 augmentations
        y_train_expanded = np.repeat(y_train, aug_factor)
        y_train = y_train_expanded
    
    return X_train_processed, X_test_processed, y_train, y_test
