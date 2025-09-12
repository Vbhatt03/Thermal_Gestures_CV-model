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
from src.visualization.plotter import plot_confusion_matrix, plot_training_history

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

        # Load data: train on all users except test_user, test on test_user
        X_train, X_test, y_train, y_test, class_names = load_thermal_data_lopo(
            DATA_DIR,
            train_users=train_users,
            test_users=[test_user],
            random_state=RANDOM_SEED
        )

        #config_obj = Config(globals())

        print("Preprocessing data...")
        X_train_processed, X_test_processed, y_train, y_test = prepare_data_for_training(
            X_train, X_test, y_train, y_test, BATCH_SIZE,
            SEQUENCE_LENGTH, NUM_CLASSES, RANDOM_SEED, use_augmentation
        )

        input_shape = X_train_processed.shape[1:]
        num_classes = len(class_names)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"thermal_letter_model_LOPO_{test_user}_{timestamp}"
        model_dir = os.path.join(MODEL_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        print("Creating model...")
        model = create_cnn_lstm_model(input_shape, num_classes)

        callbacks = [
            create_model_checkpoint(model_dir),
            #create_early_stopping(patience=10),
            #create_reduce_lr_callback(patience=5),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, 'logs'),
                histogram_freq=1
            )
        ]

        print("Training model...")
        history = model.fit(
            X_train_processed, y_train,
            validation_split=0.1,
            batch_size=32,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        final_model_path = os.path.join(model_dir, f"{model_name}_final.h5")
        save_model(model, final_model_path)

        print("Evaluating model...")
        y_pred, cm = evaluate_model(model, X_test_processed, y_test, class_names)

        plot_training_history(history)
        plot_confusion_matrix(cm, class_names)

        np.save(os.path.join(model_dir, 'class_names.npy'), class_names)

        results[test_user] = {
            "model_dir": model_dir,
            "confusion_matrix": cm,
            # Add more metrics as needed
        }

        print(f"LOPO for '{test_user}' completed. Model saved to {final_model_path}")

    print("LOPO cross-validation completed.")
    return results

if __name__ == "__main__":
    RANDOM_SEED = 42
    DATA_DIR = "D:\Vowels_only\\"  # Update this path
    MODEL_DIR = "src\models"  # Update this path
    EPOCHS = 16
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 32
    NUM_CLASSES = 26
    use_augmentation = True
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # train_model()  # Comment out the old call
    train_lopo()
# ...existing code...
