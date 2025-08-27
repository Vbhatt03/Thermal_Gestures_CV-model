import numpy as np
from scipy import ndimage
from scipy.ndimage import rotate, shift

def normalize_thermal_data(thermal_array):
    """Normalize thermal data to [0, 1] range."""
    min_val = np.min(thermal_array)
    max_val = np.max(thermal_array)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(thermal_array)
    
    return (thermal_array - min_val) / (max_val - min_val)

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
                      normalize_sequence=True, use_augmentation=False):
    """Preprocess an entire dataset."""
    processed_sequences = []
    
    for sequence in sequences:
        # Apply preprocessing
        processed_seq = preprocess_sequence(
            sequence, 
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
            # Truncate if sequence is too long
            padded_seq = seq[:max_length]
        else:
            # Pad with zeros if sequence is too short
            padding = [np.zeros_like(seq[0]) for _ in range(max_length - len(seq))]
            padded_seq = seq + padding
        
        padded_sequences.append(np.array(padded_seq))
    
    return np.array(padded_sequences)

def prepare_data_for_training(X_train, X_test, y_train, y_test, batch_size,
                              max_sequence_length, num_classes, random_state,
                              use_augmentation):
    """Prepare data for training."""
    # Preprocess training data
    X_train_processed = preprocess_dataset(
        X_train, 
        max_sequence_length,
        use_frame_differencing=True,
        normalize_sequence=True,
        use_augmentation=True
    )
    
    # Preprocess test data (no augmentation for test data)
    X_test_processed = preprocess_dataset(
        X_test, 
        max_sequence_length,
        use_frame_differencing=True,
        normalize_sequence=True,
        use_augmentation=False
    )
    
    # If augmentation was used, we need to expand y_train
    if use_augmentation:
        aug_factor = 4  # Original + 3 augmentations
        y_train_expanded = np.repeat(y_train, aug_factor)
        y_train = y_train_expanded
    
    return X_train_processed, X_test_processed, y_train, y_test
