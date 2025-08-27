import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def save_model(model, model_path):
    """Save model to disk."""
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(model_path):
    """Load model from disk."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_model_checkpoint(checkpoint_dir, monitor='val_accuracy', mode='max'):
    """Create a ModelCheckpoint callback."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_accuracy:.4f}.h5'),
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        verbose=1
    )

def create_early_stopping(patience=10, monitor='val_accuracy', mode='max'):
    """Create an EarlyStopping callback."""
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=True,
        verbose=1
    )

def create_reduce_lr_callback(patience=5, factor=0.2, min_lr=1e-6):
    """Create a ReduceLROnPlateau callback."""
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=1
    )

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model and print detailed metrics."""
    # Predict classes for test data
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate and print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return y_pred, cm

def predict_letter(model, thermal_sequence, class_names, preprocess_fn):
    # Preprocess the thermal sequence
    processed_seq = preprocess_fn([thermal_sequence])
    
    # Make prediction
    prediction = model.predict(processed_seq)[0]
    
    # Get predicted class and confidence
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    
    return class_names[predicted_idx], confidence
