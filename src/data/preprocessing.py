import numpy as np
from scipy import ndimage
from scipy.ndimage import rotate, shift

def create_sliding_windows(sequence, window_size, stride):
    """
    Create sliding windows from a long sequence.
    
    Args:
        sequence (list): List of thermal frames
        window_size (int): Number of frames in each window
        stride (int): Number of frames to move between windows
    
    Returns:
        list: List of sliding window sequences
    
    Example:
        >>> seq = [f0, f1, ..., f109]  # 110 frames
        >>> windows = create_sliding_windows(seq, window_size=30, stride=10)
        >>> # Returns 9 windows: [f0-f29], [f10-f39], [f20-f49], ...
    """
    if len(sequence) < window_size:
        # If sequence is shorter than window, return single sequence (will be padded later)
        return [sequence]
    
    windows = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[start:start + window_size]
        windows.append(window)
    
    # Always include the last window if not already covered
    if (len(sequence) - window_size) % stride != 0:
        last_window = sequence[-window_size:]
        windows.append(last_window)
    
    return windows

def normalize_thermal_data(thermal_array):
    """Normalize thermal data to [0, 1] range."""
    thermal_array = thermal_array.astype(np.float32)
    min_val = np.min(thermal_array)
    max_val = np.max(thermal_array)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(thermal_array, dtype=np.float32)
    
    normalized = (thermal_array - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1).astype(np.float32)

def normalize_per_sequence(sequence):
    """Normalize an entire sequence for consistent scaling."""
    flat_seq = np.concatenate([frame.flatten() for frame in sequence])
    min_val = np.min(flat_seq)
    max_val = np.max(flat_seq)
    
    # Avoid division by zero
    if max_val == min_val:
        return [np.zeros_like(frame) for frame in sequence]
    
    return [(frame - min_val) / (max_val - min_val) for frame in sequence]

def calculate_frame_differences(sequence):
    """Calculate frame differences to highlight motion."""
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

def reduce_noise(thermal_array, sigma=0.5):
    """Apply Gaussian filter to reduce noise."""
    return ndimage.gaussian_filter(thermal_array, sigma=sigma)

def enhance_edges(thermal_array, weight=0.3):
    """Enhance edges in thermal image."""
    edges_x = ndimage.sobel(thermal_array, axis=0)
    edges_y = ndimage.sobel(thermal_array, axis=1)
    edges = np.hypot(edges_x, edges_y)
    edges = edges / edges.max() if edges.max() > 0 else edges
    
    # Combine original with edges
    enhanced = (1 - weight) * thermal_array + weight * edges
    return enhanced

def preprocess_sequence(sequence, use_frame_differencing=True, normalize_sequence=True):
    """Apply preprocessing to a sequence of thermal frames."""
    processed_sequence = []
    
    # Step 1: Normalize the entire sequence if specified
    if normalize_sequence:
        sequence = normalize_per_sequence(sequence)
    
    # Process each frame
    for i, frame in enumerate(sequence):
        # Step 2: If not normalized as sequence, normalize individual frame
        if not normalize_sequence:
            frame = normalize_thermal_data(frame)
        
        # Step 3: Apply noise reduction
        frame = reduce_noise(frame)
        
        # Step 4: Enhance edges
        frame = enhance_edges(frame)
        
        processed_sequence.append(frame)
    
    # Step 5: Calculate frame differences if specified
    if use_frame_differencing and len(sequence) > 1:
        diff_sequence = calculate_frame_differences(processed_sequence)
        
        # Step 6: Combine original and difference frames
        # Stack as channels
        final_sequence = [np.stack([orig, diff], axis=-1) 
                         for orig, diff in zip(processed_sequence, diff_sequence)]
    else:
        # Add channel dimension
        final_sequence = [np.expand_dims(frame, axis=-1) for frame in processed_sequence]
    
    return final_sequence

def augment_sequence(sequence, max_augmentations=3):
    """Apply data augmentation to a sequence."""
    augmented_sequences = [sequence]  # Start with original sequence
    
    # Apply random rotation
    for angle in [-10, 10]:
        aug_seq = [rotate(frame, angle, reshape=False) for frame in sequence]
        augmented_sequences.append(aug_seq)
    
    # Apply random shifts
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        aug_seq = [shift(frame, [dy, dx, 0]) for frame in sequence]
        augmented_sequences.append(aug_seq)
    
    # Add slight gaussian noise
    noise_scale = 0.03
    noise_seq = [np.clip(frame + np.random.normal(0, noise_scale, frame.shape), 0, 1) 
                for frame in sequence]
    augmented_sequences.append(noise_seq)
    
    # Return original + augmented sequences (limited by max_augmentations)
    return augmented_sequences[:max_augmentations + 1]

def preprocess_dataset(sequences, max_length, use_frame_differencing=True, 
                      normalize_sequence=True, use_augmentation=False, 
                      use_sliding_window=False, window_stride=None):
    """
    Preprocess an entire dataset with optional sliding window support.
    
    Args:
        sequences (list): List of thermal frame sequences
        max_length (int): Maximum sequence length for padding/truncation
        use_frame_differencing (bool): Whether to compute frame differences
        normalize_sequence (bool): Whether to normalize entire sequence at once
        use_augmentation (bool): Whether to apply data augmentation
        use_sliding_window (bool): Whether to use sliding windows for long sequences
        window_stride (int): Stride for sliding window (if None, defaults to max_length//3)
    
    Returns:
        np.ndarray: Preprocessed sequences of shape (num_sequences, max_length, height, width, channels)
    """
    processed_sequences = []
    
    # Default stride is 1/3 of window size for good overlap
    if window_stride is None:
        window_stride = max(1, max_length // 3)
    
    for sequence in sequences:
        # Apply sliding windows if enabled and sequence is longer than max_length
        if use_sliding_window and len(sequence) > max_length:
            windowed_sequences = create_sliding_windows(sequence, max_length, window_stride)
        else:
            windowed_sequences = [sequence]
        
        # Process each window
        for windowed_seq in windowed_sequences:
            # Apply preprocessing
            processed_seq = preprocess_sequence(
                windowed_seq, 
                use_frame_differencing=use_frame_differencing,
                normalize_sequence=normalize_sequence
            )
            
            # Apply augmentation if specified
            if use_augmentation:
                aug_sequences = augment_sequence(processed_seq)
                processed_sequences.extend(aug_sequences)
            else:
                processed_sequences.append(processed_seq)
    
    # Pad sequences to the same length
    padded_sequences = []
    for seq in processed_sequences:
        if len(seq) > max_length:
            # Truncate if sequence is too long (shouldn't happen with sliding window)
            padded_seq = seq[:max_length]
        else:
            # Pad with zeros if sequence is too short
            padding = [np.zeros_like(seq[0]) for _ in range(max_length - len(seq))]
            padded_seq = seq + padding
        
        padded_sequences.append(np.array(padded_seq))
    
    return np.array(padded_sequences)

def prepare_data_for_training(X_train, X_test, y_train, y_test, batch_size,
                              max_sequence_length, num_classes, random_state,
                              use_augmentation, use_sliding_window=True, window_stride=None):
    """
    Prepare data for training with sliding window support.
    
    Args:
        X_train, X_test: Training and test sequences
        y_train, y_test: Training and test labels
        batch_size: Batch size for training
        max_sequence_length: Maximum sequence length
        num_classes: Number of classes
        random_state: Random seed
        use_augmentation: Whether to augment training data
        use_sliding_window: Whether to use sliding windows for sequences > max_sequence_length
        window_stride: Stride for sliding window (if None, auto-calculated)
    
    Returns:
        X_train_processed, X_test_processed, y_train, y_test
    """
    if window_stride is None:
        window_stride = max(1, max_sequence_length // 3)
    
    print(f"Preprocessing training data (sliding_window={use_sliding_window}, stride={window_stride})...")
    # Preprocess training data
    X_train_processed = preprocess_dataset(
        X_train, 
        max_sequence_length,
        use_frame_differencing=True,
        normalize_sequence=True,
        use_augmentation=True,
        use_sliding_window=use_sliding_window,
        window_stride=window_stride
    )
    
    print(f"Preprocessing test data (sliding_window={use_sliding_window}, stride={window_stride})...")
    # Preprocess test data (no augmentation for test data)
    X_test_processed = preprocess_dataset(
        X_test, 
        max_sequence_length,
        use_frame_differencing=True,
        normalize_sequence=True,
        use_augmentation=False,
        use_sliding_window=use_sliding_window,
        window_stride=window_stride
    )
    
    # If augmentation or sliding windows were used, we need to expand y_train
    # Calculate expansion factor
    expansion_factor = len(X_train_processed) / len(X_train)
    
    if use_sliding_window or use_augmentation:
        # Repeat labels based on sliding windows and augmentation
        y_train_expanded = np.repeat(y_train, int(expansion_factor))
        
        # Handle any rounding differences
        if len(y_train_expanded) != len(X_train_processed):
            y_train_expanded = np.repeat(y_train, int(np.ceil(len(X_train_processed) / len(y_train))))
            y_train_expanded = y_train_expanded[:len(X_train_processed)]
        
        y_train = y_train_expanded
    
    # Same for test data
    test_expansion_factor = len(X_test_processed) / len(X_test)
    if use_sliding_window:
        y_test_expanded = np.repeat(y_test, int(test_expansion_factor))
        
        # Handle any rounding differences
        if len(y_test_expanded) != len(X_test_processed):
            y_test_expanded = np.repeat(y_test, int(np.ceil(len(X_test_processed) / len(y_test))))
            y_test_expanded = y_test_expanded[:len(X_test_processed)]
        
        y_test = y_test_expanded
    
    print(f"Training samples after processing: {len(X_train_processed)} (expansion: {expansion_factor:.1f}x)")
    print(f"Test samples after processing: {len(X_test_processed)} (expansion: {test_expansion_factor:.1f}x)")
    
    return X_train_processed, X_test_processed, y_train, y_test
