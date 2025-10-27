"""
10-Fold Cross-Validation Training Script
Trains on all 10 users' data using k-fold cross validation (not LOPO).
This allows mixing of users in train/test sets for each fold.

Usage:
    python train_10fold_cv.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold

from config import *
from src.data.loader import load_json_file
from src.data.preprocessing import preprocess_sequence
from src.models.cnn_lstm import create_cnn_lstm_model
from src.models.utils import (
    save_model, create_model_checkpoint, create_early_stopping, 
    create_reduce_lr_callback, evaluate_model
)
from src.visualization.plotter import plot_confusion_matrix, plot_training_history


def load_all_thermal_data(base_folder):
    """
    Load all thermal data from all users and letters.
    
    Args:
        base_folder: Path to base folder containing user directories
        
    Returns:
        sequences: List of thermal frame sequences
        labels: List of class labels (letter names)
        class_names: Sorted list of unique class names
    """
    sequences = []
    labels = []
    
    # Get all user folders
    user_dirs = sorted([d for d in os.listdir(base_folder)
                       if os.path.isdir(os.path.join(base_folder, d)) and d != 'processed'])
    
    print(f"Found {len(user_dirs)} user folders: {user_dirs}\n")
    
    for user_idx, user in enumerate(user_dirs):
        user_folder = os.path.join(base_folder, user)
        
        # Get all letter folders for this user
        letter_dirs = sorted([d for d in os.listdir(user_folder)
                             if os.path.isdir(os.path.join(user_folder, d))])
        
        for letter in letter_dirs:
            letter_folder = os.path.join(user_folder, letter)
            
            # Get all JSON files for this letter
            files = [f for f in os.listdir(letter_folder) if f.endswith('.json')]
            
            for file in files:
                file_path = os.path.join(letter_folder, file)
                data = load_json_file(file_path)
                
                if data and len(data) > 0:
                    # Extract raw thermal frames from JSON
                    raw_frames = []
                    for entry in data:
                        if len(entry) >= 2:
                            timestamp, raw_data = entry[0], entry[1]
                            if raw_data and len(raw_data) >= 768:
                                # Reshape 768 values to 24x32 frame
                                frame = np.array(raw_data[:768], dtype=np.float32).reshape(24, 32)
                                raw_frames.append(frame)
                    
                    if raw_frames:
                        sequences.append(raw_frames)
                        labels.append(letter)
    
    # Get unique class names (sorted for consistency)
    class_names = sorted(list(set(labels)))
    num_samples = len(sequences)
    num_classes = len(class_names)
    
    print(f"Loaded {num_samples} sequences")
    print(f"Classes: {num_classes} ({class_names})\n")
    
    return sequences, labels, class_names


def preprocess_data(sequences, labels, class_names, sequence_length=100):
    """
    Preprocess all sequences.
    
    Args:
        sequences: List of raw thermal sequences
        labels: List of labels
        class_names: List of unique class names
        sequence_length: Target length for sequences
        
    Returns:
        X: Preprocessed sequences (num_samples, sequence_length, 24, 32)
        y: Label indices (num_samples,)
    """
    X = []
    y = []
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    print(f"Preprocessing {len(sequences)} sequences...")
    for idx, (sequence, label) in enumerate(zip(sequences, labels)):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(sequences)}")
        
        # Preprocess sequence (thermal only, no gradient)
        preprocessed = preprocess_sequence(
            sequence,
            normalize_sequence=True,
            target_length=sequence_length,
            hand_focused=True
        )
        
        X.append(np.array(preprocessed))
        y.append(label_to_idx[label])
    
    X = np.array(X)
    y = np.array(y)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension: (N, 100, 24, 32) → (N, 100, 24, 32, 1)
    print(f"Preprocessing complete!")
    print(f"X shape: {X.shape} (samples, frames, height, width)")
    print(f"y shape: {y.shape}\n")
    
    return X, y


def train_fold(fold_num, X_train, X_test, y_train, y_test, class_names, total_folds=10):
    """
    Train and evaluate model for a single fold.
    
    Args:
        fold_num: Current fold number (1-indexed)
        X_train, X_test: Training and test data
        y_train, y_test: Training and test labels
        class_names: List of class names
        total_folds: Total number of folds
        
    Returns:
        history: Training history
        y_pred: Predictions on test set
        cm: Confusion matrix
        test_accuracy: Final test accuracy
    """
    print(f"\n{'='*80}")
    print(f"FOLD {fold_num}/{total_folds}")
    print(f"{'='*80}")
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train samples per class: {np.bincount(y_train)}")
    
    input_shape = X_train.shape[1:]
    num_classes = len(class_names)
    
    # Create timestamp for this fold
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"thermal_letter_model_10fold_fold{fold_num}_{timestamp}"
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nCreating model... Input shape: {input_shape}")
    model = create_cnn_lstm_model(input_shape, num_classes)
    model.summary()
    
    # Create callbacks
    callbacks = [
        create_early_stopping(patience=15),
        create_reduce_lr_callback(patience=8),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        validation_split=0.2
    )
    
    # Save model
    final_model_path = os.path.join(model_dir, f"{model_name}_final.h5")
    save_model(model, final_model_path)
    print(f"Model saved: {final_model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred, cm = evaluate_model(model, X_test, y_test, class_names)
    
    # Calculate test accuracy
    test_accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot results for this fold
    try:
        plot_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot training history
        plot_training_history(history, save_path=os.path.join(plot_dir, 'training_history.png'))
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, 
                            save_path=os.path.join(plot_dir, 'confusion_matrix.png'))
        
        print(f"Plots saved to: {plot_dir}")
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    # Clean up to free memory
    del model
    tf.keras.backend.clear_session()
    
    return history, y_pred, cm, test_accuracy


def train_10fold_cv():
    """
    Main function: Train using 10-Fold Cross Validation.
    """
    print("\n" + "="*80)
    print("10-FOLD CROSS VALIDATION TRAINING")
    print("="*80 + "\n")
    
    # Load all data
    sequences, labels, class_names = load_all_thermal_data(DATA_DIR)
    
    # Preprocess all data
    X, y = preprocess_data(sequences, labels, class_names, sequence_length=SEQUENCE_LENGTH)
    
    # Initialize 10-Fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    fold_accuracies = []
    
    # Train on each fold
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        history, y_pred, cm, test_accuracy = train_fold(
            fold_num, X_train, X_test, y_train, y_test, 
            class_names, total_folds=10
        )
        
        fold_results.append({
            'fold': fold_num,
            'history': history,
            'y_pred': y_pred,
            'y_test': y_test,
            'cm': cm,
            'accuracy': test_accuracy
        })
        
        fold_accuracies.append(test_accuracy)
    
    # Print summary
    print("\n" + "="*80)
    print("10-FOLD CROSS VALIDATION RESULTS SUMMARY")
    print("="*80 + "\n")
    
    for result in fold_results:
        print(f"Fold {result['fold']:2d}: Accuracy = {result['accuracy']:.4f}")
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\n{'Mean Accuracy':30s}: {mean_accuracy:.4f}")
    print(f"{'Std Accuracy':30s}: {std_accuracy:.4f}")
    print(f"{'Min Accuracy':30s}: {np.min(fold_accuracies):.4f}")
    print(f"{'Max Accuracy':30s}: {np.max(fold_accuracies):.4f}")
    
    # Save summary
    summary_file = os.path.join(MODEL_DIR, 
                               f"10fold_cv_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write("10-FOLD CROSS VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']:2d}: Accuracy = {result['accuracy']:.4f}\n")
        f.write(f"\n{'Mean Accuracy':30s}: {mean_accuracy:.4f}\n")
        f.write(f"{'Std Accuracy':30s}: {std_accuracy:.4f}\n")
        f.write(f"{'Min Accuracy':30s}: {np.min(fold_accuracies):.4f}\n")
        f.write(f"{'Max Accuracy':30s}: {np.max(fold_accuracies):.4f}\n")
    
    print(f"\nSummary saved to: {summary_file}\n")
    
    return fold_results, mean_accuracy, std_accuracy


if __name__ == "__main__":
    RANDOM_SEED = 42
    DATA_DIR = "D:\\Data_collecn\\micro-gestures\\data\\Labelled_data\\"
    MODEL_DIR = "src\\models"
    EPOCHS = 20
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 32
    NUM_CLASSES = 5
    use_augmentation = True
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    fold_results, mean_acc, std_acc = train_10fold_cv()
