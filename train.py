import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# class Config:
#     def __init__(self, config_dict):
#         for k, v in config_dict.items():
#             setattr(self, k, v)

from config import *
from src.data.loader import load_thermal_data_lopo
from src.data.preprocessing import prepare_data_for_training
from src.models.cnn_lstm import create_cnn_lstm_model, create_lightweight_cnn_lstm
from src.models.utils import (
    save_model, create_model_checkpoint, create_early_stopping, 
    create_reduce_lr_callback, evaluate_model
)
from src.visualization.plotter import plot_confusion_matrix, plot_training_history, plot_lopo_results, TestAccuracyTracker

# ...existing code...

def get_user_folders(data_dir):
    """Return a list of user folder names in the dataset directory."""
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

def train_lopo():
    print("Starting LOPO cross-validation...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    user_folders = get_user_folders(DATA_DIR)
    results = {}

    for test_user in user_folders:
        print(f"\n=== LOPO: Using '{test_user}' as test user ===")
        train_users = [u for u in user_folders if u != test_user]

        X_train, X_test, y_train, y_test, class_names = load_thermal_data_lopo(
            DATA_DIR,
            train_users=train_users,
            test_users=[test_user],
            random_state=RANDOM_SEED
        )
        X_train = np.expand_dims(X_train, axis=-1)  # (N, 100, 24, 32) → (N, 100, 24, 32, 1)
        X_test = np.expand_dims(X_test, axis=-1)

        print(f"X_train shape: {X_train.shape}")  # Should print: (N, 100, 24, 32, 1)
        print(f"X_test shape: {X_test.shape}")
        print("Preprocessing data...")
        
        X_train_processed, X_test_processed, y_train, y_test = prepare_data_for_training(
            X_train, X_test, y_train, y_test, BATCH_SIZE,
            SEQUENCE_LENGTH, NUM_CLASSES, RANDOM_SEED, use_augmentation
        )

        print(f"X_train_processed shape: {X_train_processed.shape}")
        print(f"X_test_processed shape: {X_test_processed.shape}")
        
        input_shape = X_train_processed.shape[1:]
        num_classes = len(class_names)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"thermal_letter_model_LOPO_{test_user}_{timestamp}"
        model_dir = os.path.join(MODEL_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        print("Creating model...")
        model = create_cnn_lstm_model(input_shape, num_classes)

        # Callback to track test accuracy during training
        test_acc_tracker = TestAccuracyTracker(X_test_processed, y_test)

        callbacks = [
            test_acc_tracker,
            create_early_stopping(patience=15),
            create_reduce_lr_callback(patience=8),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, 'logs'),
                histogram_freq=1
            )
        ]

        print("Training model...")
        history = model.fit(
            X_train_processed, y_train,
            batch_size=32,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        final_model_path = os.path.join(model_dir, f"{model_name}_final.h5")
        save_model(model, final_model_path)

        print("Evaluating model on test user...")
        y_pred, cm = evaluate_model(model, X_test_processed, y_test, class_names)

        # Plot comprehensive LOPO results (loss, accuracy, confusion matrix)
        try:
            plot_lopo_results(
                history, 
                test_accuracies=test_acc_tracker.test_accuracies,
                cm=cm,
                class_names=class_names,
                user_name=test_user,
                model_dir=model_dir
            )
            print(f"✓ Comprehensive LOPO results plot created for {test_user}")
        except Exception as e:
            print(f"⚠️  Could not plot LOPO results: {e}")

        # Also save individual plots for reference
        try:
            plot_training_history(history, test_accuracies=test_acc_tracker.test_accuracies)
            # Move to model directory
            if os.path.exists('training_history.png'):
                import shutil
                shutil.move('training_history.png', os.path.join(model_dir, f'training_history_{test_user}.png'))
        except Exception as e:
            print(f"⚠️  Could not plot training history: {e}")

        try:
            plot_confusion_matrix(cm, class_names, title=f'Confusion Matrix - {test_user}')
            # Move to model directory
            if os.path.exists('confusion_matrix.png'):
                import shutil
                shutil.move('confusion_matrix.png', os.path.join(model_dir, f'confusion_matrix_{test_user}.png'))
        except Exception as e:
            print(f"⚠️  Could not plot confusion matrix: {e}")

        np.save(os.path.join(model_dir, 'class_names.npy'), class_names)

        results[test_user] = {
            "model_dir": model_dir,
            "confusion_matrix": cm,
            "test_accuracy": np.mean(y_pred == y_test)
        }

        print(f"LOPO for '{test_user}' completed.")
        print(f"Final Test Accuracy: {np.mean(y_pred == y_test):.2%}")

    print("\n" + "="*50)
    print("LOPO Cross-Validation Results Summary:")
    print("="*50)
    for user, metrics in results.items():
        print(f"{user}: {metrics['test_accuracy']:.2%}")
    
    avg_accuracy = np.mean([r['test_accuracy'] for r in results.values()])
    print(f"\nAverage Accuracy: {avg_accuracy:.2%}")
    print("="*50)
    
    return results

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

    train_lopo()