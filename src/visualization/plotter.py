import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
import tensorflow as tf

class TestAccuracyTracker(tf.keras.callbacks.Callback):
    """
    Callback to track test accuracy during training with early stopping.
    Useful for LOPO where we don't have validation set.
    
    Features:
    - Tracks test accuracy every epoch
    - Early stopping based on test accuracy plateau
    - Displays epoch-wise test accuracy
    - Saves best model based on test accuracy
    """
    def __init__(self, test_data, test_labels, patience, verbose=True):
        super().__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_accuracies = []
        self.patience = patience
        self.verbose = verbose
        self.best_test_acc = 0.0
        self.wait_count = 0
        self.best_epoch = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        """Evaluate on test set after each epoch."""
        test_loss, test_acc = self.model.evaluate(
            self.test_data, self.test_labels, verbose=0
        )
        self.test_accuracies.append(test_acc)
        
        # Display epoch-wise test accuracy
        train_acc = logs.get('accuracy', 0.0) if logs else 0.0
        train_loss = logs.get('loss', 0.0) if logs else 0.0
        
        if self.verbose:
            print(f"  ├─ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%) | Test Loss: {test_loss:.4f}")
        
        # Early stopping logic based on test accuracy improvement
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.wait_count = 0
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            
            if self.verbose:
                print(f"  └─ ✓ New best test accuracy! (was {self.test_accuracies[self.best_epoch-2]:.4f})")
        else:
            self.wait_count += 1
            if self.verbose and self.wait_count % 5 == 0:
                print(f"  └─ ⚠ No improvement for {self.wait_count}/{self.patience} epochs")
            
            # Early stopping trigger
            if self.wait_count >= self.patience:
                if self.verbose:
                    print(f"\n✓ Early stopping triggered!")
                    print(f"  Best test accuracy: {self.best_test_acc:.4f} at epoch {self.best_epoch}")
                    print(f"  Stopping training...")
                self.model.stop_training = True
                
                # Restore best weights
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)


def plot_thermal_frame(thermal_frame, title=None, cmap='inferno'):
    """
    Plot a single thermal frame.
    
    Args:
        thermal_frame: 2D thermal array
        title: Title for the plot
        cmap: Colormap to use
    """
    plt.figure(figsize=(8, 6))
    
    # Remove channel dimension if present
    if thermal_frame.ndim == 3 and thermal_frame.shape[2] == 1:
        thermal_frame = thermal_frame[:, :, 0]
    
    plt.imshow(thermal_frame, cmap=cmap)
    plt.colorbar(label='Temperature')
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()

def plot_thermal_sequence(sequence, interval=200, title=None, cmap='inferno'):
    """
    Plot a thermal sequence as an animation.
    
    Args:
        sequence: List of thermal frames
        interval: Animation interval in milliseconds
        title: Title for the animation
        cmap: Colormap to use
    """
    # Remove channel dimension if present
    if sequence[0].ndim == 3 and sequence[0].shape[2] == 1:
        sequence = [frame[:, :, 0] for frame in sequence]
    
    fig, ax = plt.figure(figsize=(8, 6)), plt.subplot()
    
    # Find global min and max for consistent colormapping
    vmin = min(frame.min() for frame in sequence)
    vmax = max(frame.max() for frame in sequence)
    
    img = ax.imshow(sequence[0], cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=ax, label='Temperature')
    
    if title:
        ax.set_title(f"{title} (Frame 1/{len(sequence)})")
    else:
        ax.set_title(f"Frame 1/{len(sequence)}")
    
    def update(frame_idx):
        img.set_array(sequence[frame_idx])
        if title:
            ax.set_title(f"{title} (Frame {frame_idx+1}/{len(sequence)})")
        else:
            ax.set_title(f"Frame {frame_idx+1}/{len(sequence)}")
        return [img]
    
    ani = FuncAnimation(fig, update, frames=len(sequence), interval=interval, blit=True)
    plt.tight_layout()
    plt.show()
    
    return ani

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Confusion matrix plot saved as 'confusion_matrix.png'")
    plt.close()

def plot_training_history(history, test_accuracies=None):
    """
    Plot training history.
    
    Args:
        history: Keras training history
        test_accuracies: Optional list of test accuracies for each epoch
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', marker='o', linewidth=2)
    
    # Plot test accuracy if provided
    if test_accuracies is not None and len(test_accuracies) > 0:
        plt.plot(test_accuracies, label='Test', marker='s', linewidth=2)
    # Otherwise check if validation data exists
    elif 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation', marker='s', linewidth=2)
    
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', marker='o', linewidth=2)
    
    # Plot validation loss if it exists
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation', marker='s', linewidth=2)
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Training history plot saved as 'training_history.png'")
    plt.close()

def plot_lopo_results(history, test_accuracies, cm, class_names, user_name, model_dir):
    """
    Plot comprehensive LOPO results: loss, test accuracy, and confusion matrix.
    
    Args:
        history: Keras training history
        test_accuracies: List of test accuracies for each epoch
        cm: Confusion matrix
        class_names: List of class names
        user_name: Name of test user
        model_dir: Directory to save plots
    """
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Loss (Train)
    ax1 = plt.subplot(1, 3, 1)
    epochs = range(1, len(history.history['loss']) + 1)
    ax1.plot(epochs, history.history['loss'], 'b-o', linewidth=2, markersize=6, label='Train Loss')
    ax1.set_title(f'Loss - Test User: {user_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Accuracy (Train + Test)
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(epochs, history.history['accuracy'], 'g-o', linewidth=2, markersize=6, label='Train Accuracy')
    if test_accuracies is not None and len(test_accuracies) > 0:
        test_epochs = range(1, len(test_accuracies) + 1)
        ax2.plot(test_epochs, test_accuracies, 'r-s', linewidth=2, markersize=6, label='Test Accuracy')
    ax2.set_title(f'Accuracy - Test User: {user_name}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Confusion Matrix
    ax3 = plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax3, cbar_kws={'label': 'Count'})
    ax3.set_title(f'Confusion Matrix - Test User: {user_name}', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=11)
    ax3.set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(model_dir, f'lopo_results_{user_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ LOPO results plot saved: {plot_path}")
    plt.close()

def visualize_model_predictions(model, X_test, y_test, class_names, num_examples=5):
    """
    Visualize model predictions on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        num_examples: Number of examples to visualize
    """
    # Get predictions
    y_pred_probs = model.predict(X_test[:num_examples])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Plot examples
    for i in range(min(num_examples, len(X_test))):
        # Get the middle frame of the sequence for visualization
        middle_idx = len(X_test[i]) // 2
        frame = X_test[i][middle_idx]
        
        # Remove channel dimension if needed
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
        
        true_label = class_names[y_test[i]]
        pred_label = class_names[y_pred[i]]
        confidence = y_pred_probs[i][y_pred[i]] * 100
        
        title = f"True: {true_label}, Pred: {pred_label} ({confidence:.1f}%)"
        
        plt.figure(figsize=(6, 5))
        plt.imshow(frame, cmap='inferno')
        plt.colorbar(label='Temperature')
        plt.title(title)
        plt.tight_layout()
        plt.show()
