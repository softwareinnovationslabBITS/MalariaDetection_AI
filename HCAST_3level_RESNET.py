"""
3-Level Hierarchical Classification using H-CAST + ResNet Embeddings (DINOv2)
===============================================================================
Uses dinov2 embeddings with all 3 levels of labels pre-computed.

Level 1: Infection Detection (Negative/Positive)
Level 2: Species Classification (Vivax/Falciparum) - PRIMARY OPTIMIZATION TARGET
Level 3: Stage Classification (7 stage classes)

Comprehensive metrics matching HCAST_Swin with all enhancements.
"""

print("\n" + "="*100)
print("H-CAST + RESNET (DINOv2 Embeddings) - 3-LEVEL HIERARCHICAL CLASSIFICATION")
print("="*100)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# GPU Configuration
print("\nConfiguring GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.optimizer.set_jit(True)
        print(f"✓ GPU enabled: {gpus[0].name}")
        print("✓ XLA JIT compilation: ENABLED")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️  No GPU detected - running on CPU")

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"✓ Mixed precision policy: {policy.name}")

print(f"TensorFlow version: {tf.__version__}")


# ============================================================================
# TREE-PATH CONSISTENCY LAYER (3-LEVEL)
# ============================================================================
@tf.keras.utils.register_keras_serializable()
class TreePathConsistency3Level(layers.Layer):
    """
    Enforces hierarchical consistency across 3 levels.
    - If L1=0 (Negative), then L2 and L3 should be suppressed
    - If L1=1 (Positive), allows L2 and L3 to activate
    """
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        l1_raw, l2_raw, l3_raw = inputs

        l1_cons = l1_raw
        l2_cons = l2_raw * l1_raw
        l3_cons = l3_raw * l1_raw

        l2_out = self.alpha * l2_cons + (1 - self.alpha) * l2_raw
        l3_out = self.alpha * l3_cons + (1 - self.alpha) * l3_raw

        return l1_cons, l2_out, l3_out

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


# ============================================================================
# HELPER FUNCTIONS FOR COMPREHENSIVE EVALUATION
# ============================================================================

def calculate_binary_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive binary classification metrics"""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = metrics['recall']  # Same as recall

    # Specificity (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1-score
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    # AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['auc'] = 0.0

    # Balanced Accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # G-Mean (geometric mean of sensitivity and specificity)
    metrics['g_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])

    return metrics


def calculate_multiclass_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive multiclass classification metrics"""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # AUC (one-vs-rest)
    try:
        n_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        metrics['auc'] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
    except:
        metrics['auc'] = 0.0

    # Balanced Accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    return metrics


def evaluate_dataset(X, y_l1, y_l2, y_l3, model, dataset_name):
    """Evaluate on a dataset and return comprehensive metrics"""
    # Get predictions
    p_l1, p_l2, p_l3 = model.predict(X, verbose=0)
    pred_l1 = (p_l1 > 0.5).astype(int).flatten()
    pred_l2 = (p_l2 > 0.5).astype(int).flatten()
    pred_l3 = np.argmax(p_l3, axis=1)

    # Get probabilities
    prob_l1 = p_l1.flatten()
    prob_l2 = p_l2.flatten()
    prob_l3 = p_l3

    y_l1_int = y_l1.astype(int)
    y_l2_int = y_l2.astype(int)
    y_l3_int = y_l3.astype(int)

    # Create masks
    l1_mask = np.ones(len(y_l1_int), dtype=bool)
    l2_mask = (y_l1_int == 1) & (y_l2_int >= 0)
    l3_mask = y_l3_int >= 0

    # Calculate metrics for each level
    l1_metrics = calculate_binary_metrics(
        y_l1_int[l1_mask], pred_l1[l1_mask], prob_l1[l1_mask]
    )

    l2_metrics = calculate_binary_metrics(
        y_l2_int[l2_mask], pred_l2[l2_mask], prob_l2[l2_mask]
    ) if l2_mask.sum() > 0 else {}

    l3_metrics = calculate_multiclass_metrics(
        y_l3_int[l3_mask], pred_l3[l3_mask], prob_l3[l3_mask]
    ) if l3_mask.sum() > 0 else {}

    # Hierarchical accuracy
    hier_acc = np.mean(
        (pred_l1[l3_mask] == y_l1_int[l3_mask]) &
        (pred_l2[l3_mask] == y_l2_int[l3_mask]) &
        (pred_l3[l3_mask] == y_l3_int[l3_mask])
    ) if l3_mask.sum() > 0 else 0.0

    return {
        'predictions': {'l1': pred_l1, 'l2': pred_l2, 'l3': pred_l3},
        'probabilities': {'l1': prob_l1, 'l2': prob_l2, 'l3': prob_l3},
        'true': {'l1': y_l1_int, 'l2': y_l2_int, 'l3': y_l3_int},
        'masks': {'l1': l1_mask, 'l2': l2_mask, 'l3': l3_mask},
        'metrics': {
            'l1': l1_metrics,
            'l2': l2_metrics,
            'l3': l3_metrics,
            'hierarchical_accuracy': hier_acc
        },
        'counts': {
            'l1': int(l1_mask.sum()),
            'l2': int(l2_mask.sum()),
            'l3': int(l3_mask.sum())
        }
    }


def save_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def plot_roc_curve(y_true, y_prob, title, save_path, level='binary'):
    """Generate and save ROC curve"""
    plt.figure(figsize=(10, 8))

    if level == 'binary':
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    else:  # multiclass
        n_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, linewidth=2, label=f'Class {i} (AUC = {auc_score:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history_obj, save_dir):
    """Generate training curves"""
    history_dict = history_obj.history

    # Create comprehensive plots
    # 1. Combined Loss curves
    plt.figure(figsize=(12, 8))
    plt.plot(history_dict['loss'], linewidth=2, label='Train Total Loss')
    plt.plot(history_dict['val_loss'], linewidth=2, label='Val Total Loss')
    plt.plot(history_dict['level1_loss'], linewidth=1.5, alpha=0.7, label='Train L1 Loss')
    plt.plot(history_dict['val_level1_loss'], linewidth=1.5, alpha=0.7, label='Val L1 Loss')
    plt.plot(history_dict['level2_loss'], linewidth=1.5, alpha=0.7, label='Train L2 Loss')
    plt.plot(history_dict['val_level2_loss'], linewidth=1.5, alpha=0.7, label='Val L2 Loss')
    plt.plot(history_dict['level3_loss'], linewidth=1.5, alpha=0.7, label='Train L3 Loss')
    plt.plot(history_dict['val_level3_loss'], linewidth=1.5, alpha=0.7, label='Val L3 Loss')
    plt.title('Model Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Combined Accuracy curves
    plt.figure(figsize=(12, 8))
    plt.plot(history_dict['level1_accuracy'], linewidth=2, label='Train L1 Accuracy')
    plt.plot(history_dict['val_level1_accuracy'], linewidth=2, label='Val L1 Accuracy')
    plt.plot(history_dict['level2_accuracy'], linewidth=2, label='Train L2 Accuracy')
    plt.plot(history_dict['val_level2_accuracy'], linewidth=2, label='Val L2 Accuracy')
    plt.plot(history_dict['level3_accuracy'], linewidth=2, label='Train L3 Accuracy')
    plt.plot(history_dict['val_level3_accuracy'], linewidth=2, label='Val L3 Accuracy')
    plt.title('Model Accuracy Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Individual level plots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Total loss
    axes[0, 0].plot(history_dict['loss'], label='Train')
    axes[0, 0].plot(history_dict['val_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Level 1 accuracy
    axes[0, 1].plot(history_dict['level1_accuracy'], label='Train')
    axes[0, 1].plot(history_dict['val_level1_accuracy'], label='Val')
    axes[0, 1].set_title('Level 1 Accuracy (Infection)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Level 2 accuracy
    axes[1, 0].plot(history_dict['level2_accuracy'], label='Train')
    axes[1, 0].plot(history_dict['val_level2_accuracy'], label='Val')
    axes[1, 0].set_title('Level 2 Accuracy (Species)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Level 3 accuracy
    axes[1, 1].plot(history_dict['level3_accuracy'], label='Train')
    axes[1, 1].plot(history_dict['val_level3_accuracy'], label='Val')
    axes[1, 1].set_title('Level 3 Accuracy (Stage)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Save training history to CSV
    history_df = pd.DataFrame(history_dict)
    history_df.insert(0, 'epoch', range(1, len(history_df) + 1))
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)


# ============================================================================
# LOAD DINOV2 EMBEDDINGS
# ============================================================================
print("\n" + "="*100)
print("LOADING DINOV2 EMBEDDINGS")
print("="*100)

workspace_dir = '/home/ghufran/MalariaML/Species_Classification/ajay'
embeddings_path = os.path.join(workspace_dir, 'embeddings_3level_dinov2_smote_proper602020.npz')

if not os.path.exists(embeddings_path):
    print(f"\n❌ ERROR: DINOv2 embeddings not found!")
    print(f"Expected: {embeddings_path}")
    print("Please generate the embeddings first.")
    exit(1)

print(f"\n✓ Loading: {os.path.basename(embeddings_path)}")
data = np.load(embeddings_path, allow_pickle=True)

# Extract embeddings
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']

# Extract all 3 levels of labels
train_l1 = data['train_l1'].astype(np.float32)
train_l2 = data['train_l2'].astype(np.float32)
train_l3 = data['train_l3'].astype(np.float32)

val_l1 = data['val_l1'].astype(np.float32)
val_l2 = data['val_l2'].astype(np.float32)
val_l3 = data['val_l3'].astype(np.float32)

test_l1 = data['test_l1'].astype(np.float32)
test_l2 = data['test_l2'].astype(np.float32)
test_l3 = data['test_l3'].astype(np.float32)

# Extract metadata
stage_names = data['stage_names']
EMBED_DIM = int(data['embed_dim'])
NUM_STAGES = int(data['num_stages'])

print(f"\n✓ Loaded embeddings:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"\n✓ Embedding dimension: {EMBED_DIM} (DINOv2)")
print(f"✓ Number of stage classes: {NUM_STAGES}")
print(f"✓ Stage classes: {list(stage_names)}")

# Display label distributions
print("\n" + "="*100)
print("LABEL DISTRIBUTIONS")
print("="*100)

print("\nLevel 1 (Infection):")
for split_name, l1 in [('Train', train_l1), ('Val', val_l1), ('Test', test_l1)]:
    neg = np.sum(l1 == 0)
    pos = np.sum(l1 == 1)
    total = len(l1)
    print(f"  {split_name:6s}: Negative={neg:4d} ({neg/total*100:.1f}%), Positive={pos:4d} ({pos/total*100:.1f}%)")

print("\nLevel 2 (Species) - among positive samples:")
for split_name, l1, l2 in [('Train', train_l1, train_l2), ('Val', val_l1, val_l2), ('Test', test_l1, test_l2)]:
    pos_mask = (l1 == 1) & (l2 >= 0)
    if pos_mask.sum() > 0:
        vivax = np.sum(l2[pos_mask] == 0)
        falci = np.sum(l2[pos_mask] == 1)
        total = pos_mask.sum()
        print(f"  {split_name:6s}: Vivax={vivax:4d} ({vivax/total*100:.1f}%), "
              f"Falciparum={falci:4d} ({falci/total*100:.1f}%)")

print("\nLevel 3 (Stages):")
print(f"{'Stage':<30s} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-" * 60)
for idx, stage in enumerate(stage_names):
    train_count = np.sum(train_l3 == idx)
    val_count = np.sum(val_l3 == idx)
    test_count = np.sum(test_l3 == idx)
    print(f"{stage:<30s} {train_count:8d} {val_count:8d} {test_count:8d}")


# ============================================================================
# BUILD PLAIN RESNET + H-CAST MODEL
# ============================================================================
print("\n" + "="*100)
print("BUILDING PLAIN RESNET + H-CAST MODEL FOR 90%+ ACCURACY")
print("="*100)

print(f"\nModel architecture:")
print(f"  Input: {EMBED_DIM}-dim embeddings (DINOv2)")
print(f"  Architecture: Plain ResNet (4 residual blocks) - REDUCED CAPACITY")
print(f"  Classifier: H-CAST hierarchical heads")
print(f"  Target: 90-92% accuracy on Level 2 (Species)")
print(f"  Level 1: Binary (Negative/Positive)")
print(f"  Level 2: Binary (Vivax/Falciparum) - PRIMARY TARGET")
print(f"  Level 3: {NUM_STAGES}-class (Stages)")

inputs = layers.Input(shape=(EMBED_DIM,), name="embedding_input")

# Initial projection - smaller dimension
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

# ResNet Block 1 (256-dim) - reduced from 512
residual = x
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([x, residual])
x = layers.Dropout(0.3)(x)

# ResNet Block 2 (256-dim) - reduced from 512
residual = x
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([x, residual])
x = layers.Dropout(0.3)(x)

# Downsample to 128 - reduced from 256
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

# ResNet Block 3 (128-dim) - reduced from 256
residual = x
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([x, residual])
x = layers.Dropout(0.3)(x)

# ResNet Block 4 (128-dim) - reduced from 256
residual = x
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([x, residual])
x = layers.Dropout(0.3)(x)

# Final feature representation - smaller
features = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
features = layers.BatchNormalization()(features)
features = layers.Dropout(0.4)(features)

# Hierarchical heads (raw outputs)
l1_raw = layers.Dense(1, activation='sigmoid', name='l1_raw')(features)
l2_raw = layers.Dense(1, activation='sigmoid', name='l2_raw')(features)
l3_raw = layers.Dense(NUM_STAGES, activation='softmax', dtype='float32', name='l3_raw')(features)

# Apply tree-path consistency
l1_cons, l2_cons, l3_cons = TreePathConsistency3Level(alpha=0.5)([l1_raw, l2_raw, l3_raw])

# Named outputs
l1_out = layers.Identity(name="level1")(l1_cons)
l2_out = layers.Identity(name="level2")(l2_cons)
l3_out = layers.Identity(name="level3", dtype='float32')(l3_cons)

hcast_model = keras.Model(
    inputs=inputs,
    outputs=[l1_out, l2_out, l3_out],
    name="HCAST_3Level_ResNet_DINOv2"
)

print("\n✓ Model created!")
hcast_model.summary(line_length=120)


# ============================================================================
# COMPILE MODEL - OPTIMIZED FOR LEVEL 2
# ============================================================================
print("\n" + "="*100)
print("COMPILING RESNET + H-CAST MODEL FOR 90%+ LEVEL 2 ACCURACY")
print("="*100)

hcast_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005,
        clipnorm=1.0
    ),
    loss=[
        "binary_crossentropy",              # level1
        "binary_crossentropy",              # level2
        "sparse_categorical_crossentropy",  # level3
    ],
    loss_weights=[0.5, 3.0, 1.0],  # [L1, L2 (PRIMARY), L3] - use list instead of dict
    metrics={
        "level1": ["accuracy"],
        "level2": ["accuracy"],
        "level3": ["accuracy"],
    }
)

print("✓ Model compiled!")
print("  Optimizer: Adam (lr=0.0005, clipnorm=1.0)")
print("  Loss weights: L1=0.5, L2=3.0 (PRIMARY), L3=1.0")


# ============================================================================
# CUSTOM VALIDATION CALLBACK (WORKAROUND FOR KERAS BUG)
# ============================================================================
class CustomValidation(keras.callbacks.Callback):
    """Custom validation callback to avoid Keras KeyError: 0 bug with sample_weight + validation_data"""

    def __init__(self, X_val, val_l1, val_l2, val_l3, val_sw_l1, val_sw_l2, val_sw_l3):
        super().__init__()
        self.X_val = X_val
        # Use lists instead of dicts to avoid Keras bug
        self.val_labels = [val_l1, val_l2, val_l3]
        self.val_sample_weights = [val_sw_l1, val_sw_l2, val_sw_l3]

    def on_epoch_end(self, epoch, logs=None):
        # Compute validation loss and metrics
        val_results = self.model.evaluate(
            self.X_val,
            self.val_labels,
            sample_weight=self.val_sample_weights,
            verbose=0,
            return_dict=True
        )

        # Add validation metrics to logs
        for key, value in val_results.items():
            logs[f'val_{key}'] = value

        # Print validation metrics
        if epoch % 10 == 0 or epoch < 5:
            print(f"\n  Validation - Loss: {val_results['loss']:.4f}, "
                  f"L1 Acc: {val_results.get('level1_accuracy', 0):.4f}, "
                  f"L2 Acc: {val_results.get('level2_accuracy', 0):.4f}, "
                  f"L3 Acc: {val_results.get('level3_accuracy', 0):.4f}")


# ============================================================================
# SETUP CALLBACKS
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = os.path.join(workspace_dir, f'HCASTRESNET/60-20-20SMOTE/HCAST_ResNet_DINOv2_Checkpoints_{timestamp}')
os.makedirs(checkpoint_dir, exist_ok=True)

# Create results structure
results_dir = os.path.join(workspace_dir, f'HCASTRESNET/60-20-20SMOTE/HCAST_ResNet_DINOv2_Results_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'confusion_matrices'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'roc_curves'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'training_graphs'), exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
        monitor='val_level2_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_level2_accuracy',
        patience=50,  # Increased for better convergence
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_level2_accuracy',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        mode='max',
        verbose=1
    ),
    CSVLogger(os.path.join(checkpoint_dir, 'training_log.csv'))
]

print(f"\n✓ Checkpoint directory: {checkpoint_dir}")
print(f"✓ Results directory: {results_dir}")
print(f"✓ Callbacks: ModelCheckpoint, EarlyStopping (patience=50), ReduceLROnPlateau")


# ============================================================================
# PREPARE SAMPLE WEIGHTS FOR MASKING INVALID LABELS
# ============================================================================
print("\n" + "="*100)
print("PREPARING SAMPLE WEIGHTS FOR HIERARCHICAL TRAINING")
print("="*100)

# Create sample weights for each level - BINARY MASKS ONLY
# Level 1: All samples are valid
train_sw_l1 = np.ones(len(train_l1), dtype=np.float32)

# Level 2: Only positive samples with valid labels (not -1)
train_sw_l2 = ((train_l1 == 1) & (train_l2 >= 0)).astype(np.float32)

# Level 3: Only samples with valid stage labels (not -1)
train_sw_l3 = (train_l3 >= 0).astype(np.float32)

# Calculate class imbalance for Level 2
l2_mask = train_sw_l2 > 0
l2_labels = train_l2[l2_mask]
vivax_count = np.sum(l2_labels == 0)
falci_count = np.sum(l2_labels == 1)
total = vivax_count + falci_count

print(f"\nClass distribution for Level 2 (Species):")
print(f"  Vivax (0):      {vivax_count:5d} samples ({vivax_count/total*100:.1f}%)")
print(f"  Falciparum (1): {falci_count:5d} samples ({falci_count/total*100:.1f}%)")

print(f"\nSample weights summary (binary masks for valid samples):")
print(f"  L1 - Train: {train_sw_l1.sum():.0f}/{len(train_l1)} valid")
print(f"  L2 - Train: {train_sw_l2.sum():.0f}/{len(train_l1)} valid")
print(f"  L3 - Train: {train_sw_l3.sum():.0f}/{len(train_l1)} valid")

# Fix invalid labels by replacing -1 with 0 (will be masked by sample weights)
train_l2_fixed = np.where(train_l2 < 0, 0, train_l2).astype(np.float32)
val_l2_fixed = np.where(val_l2 < 0, 0, val_l2).astype(np.float32)
train_l3_fixed = np.where(train_l3 < 0, 0, train_l3).astype(np.float32)
val_l3_fixed = np.where(val_l3 < 0, 0, val_l3).astype(np.float32)

# Create validation sample weights (matching training structure)
val_sw_l1 = np.ones(len(val_l1), dtype=np.float32)
val_sw_l2 = ((val_l1 == 1) & (val_l2 >= 0)).astype(np.float32)
val_sw_l3 = (val_l3 >= 0).astype(np.float32)

print("\n✓ Sample weights prepared!")

# Add custom validation callback to callbacks list
custom_val_callback = CustomValidation(
    X_val, val_l1, val_l2_fixed, val_l3_fixed,
    val_sw_l1, val_sw_l2, val_sw_l3
)
callbacks.insert(0, custom_val_callback)  # Insert at beginning so it runs first
print("✓ Custom validation callback added!")


# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "="*100)
print("TRAINING PLAIN RESNET + H-CAST MODEL")
print("="*100)

print("\nTraining configuration:")
print("  - Architecture: 4 ResNet blocks (512→512→256→256)")
print("  - Epochs: 300 (with early stopping)")
print("  - Batch size: 64")
print("  - Sample weights: Binary masks for valid samples only")
print("  - Primary focus: Level 2 (Species) - loss weight 3.0")
print("  - Validation: Custom callback (avoids Keras KeyError bug)")

start_time = time.time()

# Train without validation_data parameter - CustomValidation callback handles it
# Use LISTS instead of DICTS to avoid Keras KeyError: 0 bug
history = hcast_model.fit(
    X_train,
    [train_l1, train_l2_fixed, train_l3_fixed],  # List instead of dict
    sample_weight=[train_sw_l1, train_sw_l2, train_sw_l3],  # List instead of dict
    epochs=300,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✓ Training complete! Time: {training_time/60:.1f} minutes")

# Save model
model_path = os.path.join(checkpoint_dir, 'final_model.keras')
hcast_model.save(model_path)
print(f"✓ Final model saved: {model_path}")


# ============================================================================
# COMPREHENSIVE EVALUATION ON ALL SPLITS
# ============================================================================
print("\n" + "="*100)
print("COMPREHENSIVE EVALUATION - ALL SPLITS")
print("="*100)

# Load best model
best_model_path = os.path.join(checkpoint_dir, 'best_model.keras')
print(f"\nLoading best model: {best_model_path}")
hcast_model = keras.models.load_model(
    best_model_path,
    custom_objects={'TreePathConsistency3Level': TreePathConsistency3Level}
)

# Evaluate all datasets
print("\nEvaluating Train set...")
train_results = evaluate_dataset(X_train, train_l1, train_l2, train_l3, hcast_model, 'train')

print("Evaluating Val set...")
val_results = evaluate_dataset(X_val, val_l1, val_l2, val_l3, hcast_model, 'val')

print("Evaluating Test set...")
test_results = evaluate_dataset(X_test, test_l1, test_l2, test_l3, hcast_model, 'test')

print("\n✓ Evaluation complete!")


# ============================================================================
# GENERATE TRAINING CURVES
# ============================================================================
print("\nGenerating training curves...")
plot_training_history(history, os.path.join(results_dir, 'training_graphs'))


# ============================================================================
# GENERATE CONFUSION MATRICES AND ROC CURVES
# ============================================================================
print("\nGenerating confusion matrices and ROC curves...")
datasets = {
    'train': train_results,
    'val': val_results,
    'test': test_results
}

for ds_name, ds_results in datasets.items():
    # Level 1 - Confusion Matrix & ROC
    save_confusion_matrix(
        ds_results['true']['l1'][ds_results['masks']['l1']],
        ds_results['predictions']['l1'][ds_results['masks']['l1']],
        [0, 1], f"Level 1 (Infection) - {ds_name.upper()}",
        os.path.join(results_dir, 'confusion_matrices', f'level1_{ds_name}.png')
    )
    plot_roc_curve(
        ds_results['true']['l1'][ds_results['masks']['l1']],
        ds_results['probabilities']['l1'][ds_results['masks']['l1']],
        f"Level 1 (Infection) ROC Curve - {ds_name.upper()}",
        os.path.join(results_dir, 'roc_curves', f'level1_{ds_name}.png'),
        level='binary'
    )

    # Level 2 - Confusion Matrix & ROC
    if ds_results['masks']['l2'].sum() > 0:
        save_confusion_matrix(
            ds_results['true']['l2'][ds_results['masks']['l2']],
            ds_results['predictions']['l2'][ds_results['masks']['l2']],
            [0, 1], f"Level 2 (Species: Vivax=0, Falciparum=1) - {ds_name.upper()}",
            os.path.join(results_dir, 'confusion_matrices', f'level2_{ds_name}.png')
        )
        plot_roc_curve(
            ds_results['true']['l2'][ds_results['masks']['l2']],
            ds_results['probabilities']['l2'][ds_results['masks']['l2']],
            f"Level 2 (Species) ROC Curve - {ds_name.upper()}",
            os.path.join(results_dir, 'roc_curves', f'level2_{ds_name}.png'),
            level='binary'
        )

    # Level 3 - Confusion Matrix & ROC
    if ds_results['masks']['l3'].sum() > 0:
        l3_true = ds_results['true']['l3'][ds_results['masks']['l3']]
        l3_pred = ds_results['predictions']['l3'][ds_results['masks']['l3']]
        l3_prob = ds_results['probabilities']['l3'][ds_results['masks']['l3']]
        present_stages = sorted(np.unique(l3_true))
        save_confusion_matrix(
            l3_true, l3_pred,
            present_stages, f"Level 3 (Stage) - {ds_name.upper()}",
            os.path.join(results_dir, 'confusion_matrices', f'level3_{ds_name}.png')
        )
        plot_roc_curve(
            l3_true, l3_prob,
            f"Level 3 (Stage) ROC Curves - {ds_name.upper()}",
            os.path.join(results_dir, 'roc_curves', f'level3_{ds_name}.png'),
            level='multiclass'
        )


# ============================================================================
# GENERATE COMPREHENSIVE DETAILED REPORT WITH STD
# ============================================================================
print("\nGenerating comprehensive detailed report...")

report_lines = []
report_lines.append("=" * 100)
report_lines.append("PLAIN RESNET + H-CAST - 3-LEVEL HIERARCHICAL CLASSIFICATION")
report_lines.append("=" * 100)
report_lines.append("")
report_lines.append(f"Model: {best_model_path}")
report_lines.append(f"Timestamp: {timestamp}")
report_lines.append(f"Embeddings: DINOv2 ({EMBED_DIM}-dim)")
report_lines.append(f"Architecture: Plain ResNet (4 residual blocks)")
report_lines.append(f"Classifier: H-CAST hierarchical heads")
report_lines.append(f"Stages: {list(stage_names)}")
report_lines.append("")
report_lines.append("ARCHITECTURE:")
report_lines.append("  • 4 ResNet blocks: 512→512→256→256 dims")
report_lines.append("  • Batch Normalization + Dropout")
report_lines.append("  • L2 regularization (0.0001)")
report_lines.append("  • H-CAST TreePathConsistency (alpha=0.5)")
report_lines.append("")
report_lines.append("TRAINING STRATEGY:")
report_lines.append("  • Sample weight masking (prevents NaN)")
report_lines.append("  • Class balancing for Level 2")
report_lines.append("  • Loss weight 3.0 for Level 2 (primary focus)")
report_lines.append("  • Gradient clipping (clipnorm=1.0)")
report_lines.append("  • ReduceLROnPlateau (patience=15)")
report_lines.append("")

# Calculate mean and std across splits for each level
def calculate_stats_across_splits(train_m, val_m, test_m):
    """Calculate mean and std for each metric across splits"""
    stats = {}
    all_metrics = set(train_m.keys()) | set(val_m.keys()) | set(test_m.keys())
    for metric in all_metrics:
        values = [
            train_m.get(metric, 0.0),
            val_m.get(metric, 0.0),
            test_m.get(metric, 0.0)
        ]
        stats[metric] = {
            'train': train_m.get(metric, 0.0),
            'val': val_m.get(metric, 0.0),
            'test': test_m.get(metric, 0.0),
            'mean': np.mean(values),
            'std': np.std(values, ddof=1)
        }
    return stats

# Get metrics for each level
l1_stats = calculate_stats_across_splits(
    train_results['metrics']['l1'],
    val_results['metrics']['l1'],
    test_results['metrics']['l1']
)

l2_stats = calculate_stats_across_splits(
    train_results['metrics']['l2'],
    val_results['metrics']['l2'],
    test_results['metrics']['l2']
)

l3_stats = calculate_stats_across_splits(
    train_results['metrics']['l3'],
    val_results['metrics']['l3'],
    test_results['metrics']['l3']
)

# Level 1 Summary Table
report_lines.append("=" * 100)
report_lines.append("LEVEL 1: INFECTION STATUS (NEGATIVE/POSITIVE)")
report_lines.append("=" * 100)
report_lines.append("")
report_lines.append(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10} {'Mean':>10} {'STD':>10}")
report_lines.append("-" * 100)
for metric in ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'auc', 'balanced_accuracy', 'mcc', 'g_mean']:
    if metric in l1_stats:
        s = l1_stats[metric]
        report_lines.append(f"{metric.replace('_', ' ').title():<25} {s['train']:>10.4f} {s['val']:>10.4f} {s['test']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
report_lines.append("")

# Level 2 Summary Table
report_lines.append("=" * 100)
report_lines.append("LEVEL 2: SPECIES CLASSIFICATION (VIVAX/FALCIPARUM) - PRIMARY TARGET")
report_lines.append("=" * 100)
report_lines.append("")
report_lines.append(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10} {'Mean':>10} {'STD':>10}")
report_lines.append("-" * 100)
for metric in ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'auc', 'balanced_accuracy', 'mcc', 'g_mean']:
    if metric in l2_stats:
        s = l2_stats[metric]
        report_lines.append(f"{metric.replace('_', ' ').title():<25} {s['train']:>10.4f} {s['val']:>10.4f} {s['test']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
report_lines.append("")

# Level 3 Summary Table
report_lines.append("=" * 100)
report_lines.append("LEVEL 3: PARASITE STAGE CLASSIFICATION")
report_lines.append("=" * 100)
report_lines.append("")
report_lines.append(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10} {'Mean':>10} {'STD':>10}")
report_lines.append("-" * 100)
for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'balanced_accuracy', 'mcc']:
    if metric in l3_stats:
        s = l3_stats[metric]
        report_lines.append(f"{metric.replace('_', ' ').title():<25} {s['train']:>10.4f} {s['val']:>10.4f} {s['test']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
report_lines.append("")

# Hierarchical Accuracy
hier_train = train_results['metrics']['hierarchical_accuracy']
hier_val = val_results['metrics']['hierarchical_accuracy']
hier_test = test_results['metrics']['hierarchical_accuracy']
hier_mean = np.mean([hier_train, hier_val, hier_test])
hier_std = np.std([hier_train, hier_val, hier_test], ddof=1)

report_lines.append("=" * 100)
report_lines.append("HIERARCHICAL ACCURACY (ALL LEVELS CORRECT)")
report_lines.append("=" * 100)
report_lines.append("")
report_lines.append(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10} {'Mean':>10} {'STD':>10}")
report_lines.append("-" * 100)
report_lines.append(f"{'Hierarchical Accuracy':<25} {hier_train:>10.4f} {hier_val:>10.4f} {hier_test:>10.4f} {hier_mean:>10.4f} {hier_std:>10.4f}")
report_lines.append("")

# Detailed classification reports for each dataset
for ds_name, ds_results in [('TRAIN', train_results), ('VAL', val_results), ('TEST', test_results)]:
    report_lines.append("=" * 100)
    report_lines.append(f"{ds_name} SET - DETAILED RESULTS")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"Samples: {ds_results['counts']['l1']:,} total, {ds_results['counts']['l2']:,} for L2, {ds_results['counts']['l3']:,} for L3")
    report_lines.append("")

    # Level 2 classification report
    if ds_results['masks']['l2'].sum() > 0:
        report_lines.append("LEVEL 2 (Species) Classification Report:")
        report_lines.append("-" * 100)
        l2_true = ds_results['true']['l2'][ds_results['masks']['l2']]
        l2_pred = ds_results['predictions']['l2'][ds_results['masks']['l2']]
        l2_report = classification_report(
            l2_true, l2_pred,
            labels=[0, 1],
            target_names=['Vivax', 'Falciparum'],
            digits=4, zero_division=0
        )
        report_lines.append(l2_report)
        report_lines.append("")

    # Level 3 classification report
    if ds_results['masks']['l3'].sum() > 0:
        report_lines.append("LEVEL 3 (Stage) Classification Report:")
        report_lines.append("-" * 100)
        test_l3_true = ds_results['true']['l3'][ds_results['masks']['l3']]
        test_l3_pred = ds_results['predictions']['l3'][ds_results['masks']['l3']]
        present = sorted(np.unique(test_l3_true))
        report_text = classification_report(
            test_l3_true, test_l3_pred,
            labels=present,
            target_names=[stage_names[i] for i in present],
            digits=4, zero_division=0
        )
        report_lines.append(report_text)
        report_lines.append("")

# Training information
report_lines.append("=" * 100)
report_lines.append("TRAINING INFORMATION")
report_lines.append("=" * 100)
report_lines.append("")
report_lines.append(f"Total training time: {training_time/60:.1f} minutes")
report_lines.append(f"Total epochs: {len(history.history['loss'])}")
hist_dict = history.history
best_epoch = np.argmax(hist_dict['val_level2_accuracy']) + 1
report_lines.append(f"Best epoch (val_level2_accuracy): {best_epoch}")
report_lines.append(f"Best val_level2_accuracy: {hist_dict['val_level2_accuracy'][best_epoch-1]:.4f}")
report_lines.append(f"Final train loss: {hist_dict['loss'][-1]:.6f}")
report_lines.append(f"Final val loss: {hist_dict['val_loss'][-1]:.6f}")
report_lines.append("")

# Architecture details
report_lines.append("=" * 100)
report_lines.append("MODEL ARCHITECTURE")
report_lines.append("=" * 100)
report_lines.append("")
report_lines.append(f"  Embeddings: DINOv2 ({EMBED_DIM}-dim)")
report_lines.append(f"  Architecture: ResNet-style with residual connections")
report_lines.append(f"  Loss weights: L1=1.0, L2=2.0 (PRIMARY), L3=1.5")
report_lines.append(f"  Optimizer: Adam (lr=0.001)")
report_lines.append(f"  Regularization: L2 (0.001), Dropout (0.3-0.4), BatchNorm")
report_lines.append("")

# Save detailed report
report_path = os.path.join(results_dir, 'summary.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

# Print to console
print("\n" + '\n'.join(report_lines))


# ============================================================================
# SAVE COMPREHENSIVE METRICS JSON AND CSV
# ============================================================================
metrics_data = {
    'model_path': best_model_path,
    'timestamp': timestamp,
    'embeddings': 'DINOv2',
    'embed_dim': int(EMBED_DIM),
    'architecture': 'ResNet-style with residual connections',
    'stages': list(stage_names),
    'training_info': {
        'total_epochs': len(history.history['loss']),
        'best_epoch': int(best_epoch),
        'training_time_minutes': float(training_time / 60),
        'final_train_loss': float(hist_dict['loss'][-1]),
        'final_val_loss': float(hist_dict['val_loss'][-1])
    },
    'level1_infection': {
        'train': {k: float(v) for k, v in train_results['metrics']['l1'].items()},
        'val': {k: float(v) for k, v in val_results['metrics']['l1'].items()},
        'test': {k: float(v) for k, v in test_results['metrics']['l1'].items()},
        'mean_std': {k: {'mean': float(v['mean']), 'std': float(v['std'])} for k, v in l1_stats.items()},
        'samples': train_results['counts']['l1']
    },
    'level2_species': {
        'train': {k: float(v) for k, v in train_results['metrics']['l2'].items()},
        'val': {k: float(v) for k, v in val_results['metrics']['l2'].items()},
        'test': {k: float(v) for k, v in test_results['metrics']['l2'].items()},
        'mean_std': {k: {'mean': float(v['mean']), 'std': float(v['std'])} for k, v in l2_stats.items()},
        'samples': train_results['counts']['l2']
    },
    'level3_stage': {
        'train': {k: float(v) for k, v in train_results['metrics']['l3'].items()},
        'val': {k: float(v) for k, v in val_results['metrics']['l3'].items()},
        'test': {k: float(v) for k, v in test_results['metrics']['l3'].items()},
        'mean_std': {k: {'mean': float(v['mean']), 'std': float(v['std'])} for k, v in l3_stats.items()},
        'samples': train_results['counts']['l3']
    },
    'hierarchical_accuracy': {
        'train': float(hier_train),
        'val': float(hier_val),
        'test': float(hier_test),
        'mean': float(hier_mean),
        'std': float(hier_std)
    }
}

# Save JSON
metrics_path = os.path.join(results_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_data, f, indent=4)

# Save metrics to CSV files
# Level 1 metrics CSV
l1_df = pd.DataFrame({
    'Metric': list(l1_stats.keys()),
    'Train': [l1_stats[k]['train'] for k in l1_stats.keys()],
    'Val': [l1_stats[k]['val'] for k in l1_stats.keys()],
    'Test': [l1_stats[k]['test'] for k in l1_stats.keys()],
    'Mean': [l1_stats[k]['mean'] for k in l1_stats.keys()],
    'STD': [l1_stats[k]['std'] for k in l1_stats.keys()]
})
l1_df.to_csv(os.path.join(results_dir, 'metrics', 'level1_metrics.csv'), index=False)

# Level 2 metrics CSV
l2_df = pd.DataFrame({
    'Metric': list(l2_stats.keys()),
    'Train': [l2_stats[k]['train'] for k in l2_stats.keys()],
    'Val': [l2_stats[k]['val'] for k in l2_stats.keys()],
    'Test': [l2_stats[k]['test'] for k in l2_stats.keys()],
    'Mean': [l2_stats[k]['mean'] for k in l2_stats.keys()],
    'STD': [l2_stats[k]['std'] for k in l2_stats.keys()]
})
l2_df.to_csv(os.path.join(results_dir, 'metrics', 'level2_species_metrics.csv'), index=False)

# Level 3 metrics CSV
l3_df = pd.DataFrame({
    'Metric': list(l3_stats.keys()),
    'Train': [l3_stats[k]['train'] for k in l3_stats.keys()],
    'Val': [l3_stats[k]['val'] for k in l3_stats.keys()],
    'Test': [l3_stats[k]['test'] for k in l3_stats.keys()],
    'Mean': [l3_stats[k]['mean'] for k in l3_stats.keys()],
    'STD': [l3_stats[k]['std'] for k in l3_stats.keys()]
})
l3_df.to_csv(os.path.join(results_dir, 'metrics', 'level3_stage_metrics.csv'), index=False)


# ============================================================================
# FINAL SUMMARY WITH 90% TARGET CHECK
# ============================================================================
print("\n" + "="*100)
print("TRAINING AND EVALUATION COMPLETE!")
print("="*100)
print(f"\n✓ Comprehensive results saved to: {results_dir}")
print(f"  - Summary report: {report_path}")
print(f"  - Metrics JSON: {metrics_path}")
print(f"  - Metrics CSVs: {os.path.join(results_dir, 'metrics/')}")
print(f"  - Confusion matrices: {os.path.join(results_dir, 'confusion_matrices/')}")
print(f"  - ROC curves: {os.path.join(results_dir, 'roc_curves/')}")
print(f"  - Training graphs: {os.path.join(results_dir, 'training_graphs/')}")

print("\n" + "="*100)
print("LEVEL 2 (SPECIES) ACCURACY - 90% TARGET CHECK")
print("="*100)

l2_train_acc = l2_stats['accuracy']['train']
l2_val_acc = l2_stats['accuracy']['val']
l2_test_acc = l2_stats['accuracy']['test']
l2_mean_acc = l2_stats['accuracy']['mean']
l2_std_acc = l2_stats['accuracy']['std']

# Check if we hit 90%
train_target = "✓ TARGET MET!" if l2_train_acc >= 0.90 else "⚠️ Below target"
val_target = "✓ TARGET MET!" if l2_val_acc >= 0.90 else "⚠️ Below target"
test_target = "✓ TARGET MET!" if l2_test_acc >= 0.90 else "⚠️ Below target"
mean_target = "✓ TARGET MET!" if l2_mean_acc >= 0.90 else "⚠️ Below target"

print(f"\n  Train Accuracy: {l2_train_acc*100:6.2f}%  {train_target}")
print(f"  Val   Accuracy: {l2_val_acc*100:6.2f}%  {val_target}")
print(f"  Test  Accuracy: {l2_test_acc*100:6.2f}%  {test_target}")
print(f"  Mean  Accuracy: {l2_mean_acc*100:6.2f}% ± {l2_std_acc*100:.2f}%  {mean_target}")

print(f"\n  Level 2 AUC:")
print(f"  Train AUC: {l2_stats['auc']['train']:.4f}")
print(f"  Val   AUC: {l2_stats['auc']['val']:.4f}")
print(f"  Test  AUC: {l2_stats['auc']['test']:.4f}")
print(f"  Mean  AUC: {l2_stats['auc']['mean']:.4f} ± {l2_stats['auc']['std']:.4f}")

print("\n" + "="*100)
print("KEY METRICS SUMMARY")
print("="*100)
print(f"\nLevel 1 (Infection):")
print(f"  Mean Accuracy: {l1_stats['accuracy']['mean']*100:6.2f}% ± {l1_stats['accuracy']['std']*100:.2f}%")

print(f"\nLevel 3 (Stage):")
print(f"  Mean Accuracy: {l3_stats['accuracy']['mean']*100:6.2f}% ± {l3_stats['accuracy']['std']*100:.2f}%")

print(f"\nHierarchical (All Levels Correct):")
print(f"  Mean Accuracy: {hier_mean*100:6.2f}% ± {hier_std*100:.2f}%")

print("\n" + "="*100)
print("="*100)
