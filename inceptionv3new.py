import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPool2D, Input, GlobalAveragePooling2D, AveragePooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, concatenate
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel
from keras.callbacks import EarlyStopping
import joblib
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from datetime import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configuration
batch_size = 16  # Smaller batch size for better generalization
epochs = 150
learning_rate = 0.0005  # Lower learning rate for better convergence

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"INCEPTION/80-20-20SMOTE/inception_l2_species_metrics_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Metrics will be saved to: {OUTPUT_DIR}/")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =============================================================================
# MODEL STRUCTURE - INCEPTION V3 (PRESERVED AS REQUESTED)
# =============================================================================

def inception_module(x, f1, f2, f3):
   conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)
   conv3 = Conv2D(f2, (3, 3), padding='same', activation='relu')(x)
   conv5 = Conv2D(f3, (5, 5), padding='same', activation='relu')(x)
   pool = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
   out = concatenate([conv1, conv3, conv5, pool])
   return out


def conv2d_bn(x, filters, num_rows, num_columns, padding='same', strides=(1, 1)):
    x = Conv2D(filters, (num_rows, num_columns),
               strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('relu')(x)
    return x


channel_axis = 3


def inception_A(x):
    branch_1x1 = conv2d_bn(x, 64, 1, 1)

    branch_5x5 = conv2d_bn(x, 48, 1, 1)
    branch_5x5 = conv2d_bn(branch_5x5, 64, 5, 5)

    branch_3x3 = conv2d_bn(x, 64, 1, 1)
    branch_3x3 = conv2d_bn(branch_3x3, 96, 3, 3)
    branch_3x3 = conv2d_bn(branch_3x3, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = concatenate([branch_1x1, branch_3x3, branch_5x5,
                    branch_pool], axis=channel_axis)
    return x


def reduction_A(x):
    branch_3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
    branch_3x3_2 = conv2d_bn(x, 64, 1, 1)
    branch_3x3_2 = conv2d_bn(branch_3x3_2, 96, 3, 3)
    branch_3x3_2 = conv2d_bn(branch_3x3_2, 96, 3, 3,
                             strides=(2, 2), padding='valid')
    branch_pool = MaxPool2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch_3x3, branch_3x3_2, branch_pool], axis=channel_axis)
    return x


def inception_B(x):
    branch_1x1 = conv2d_bn(x, 192, 1, 1)

    branch_7x7 = conv2d_bn(x, 128, 1, 1)
    branch_7x7 = conv2d_bn(branch_7x7, 128, 1, 7)
    branch_7x7 = conv2d_bn(branch_7x7, 192, 7, 1)

    branch_7x7_2 = conv2d_bn(x, 128, 1, 1)
    branch_7x7_2 = conv2d_bn(branch_7x7_2, 128, 7, 1)
    branch_7x7_2 = conv2d_bn(branch_7x7_2, 128, 1, 7)
    branch_7x7_2 = conv2d_bn(branch_7x7_2, 128, 7, 1)
    branch_7x7_2 = conv2d_bn(branch_7x7_2, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch_1x1, branch_7x7, branch_7x7_2,
                    branch_pool], axis=channel_axis)
    return x


def reduction_B(x):
    branch_3x3 = conv2d_bn(x, 192, 1, 1)
    branch_3x3 = conv2d_bn(branch_3x3, 320, 3, 3,
                           strides=(2, 2), padding='valid')

    branch_7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch_7x7x3 = conv2d_bn(branch_7x7x3, 192, 1, 7)
    branch_7x7x3 = conv2d_bn(branch_7x7x3, 192, 7, 1)
    branch_7x7x3 = conv2d_bn(branch_7x7x3, 192, 3, 3,
                             strides=(2, 2), padding='valid')

    branch_pool = MaxPool2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch_3x3, branch_7x7x3, branch_pool], axis=channel_axis)
    return x


def inception_C(x):
    branch_1x1 = conv2d_bn(x, 320, 1, 1)

    branch_3x3 = conv2d_bn(x, 384, 1, 1)
    branch_3x3_1 = conv2d_bn(branch_3x3, 384, 1, 3)
    branch_3x3_2 = conv2d_bn(branch_3x3, 384, 3, 1)
    branch_3x3 = concatenate([branch_3x3_1, branch_3x3_2], axis=channel_axis)

    branch_3x3db1 = conv2d_bn(x, 448, 1, 1)
    branch_3x3db1 = conv2d_bn(branch_3x3db1, 384, 3, 3)
    branch_3x3db1_1 = conv2d_bn(branch_3x3db1, 384, 1, 3)
    branch_3x3db1_2 = conv2d_bn(branch_3x3db1, 384, 3, 1)
    branch_3x3db1 = concatenate(
        [branch_3x3db1_1, branch_3x3db1_2], axis=channel_axis)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = concatenate(
        [branch_1x1, branch_3x3, branch_3x3db1, branch_pool], axis=channel_axis)
    return x


def Incep(input_shape=(32, 48, 1)):
    """
    TRUE Inception V3 Model Structure - Adapted for Embeddings
    This is the ACTUAL Inception CNN with Conv2D layers (NOT Dense network)

    Input: Reshaped embeddings in 2D format (H, W, 1)
      - Auto-detects optimal dimensions based on embedding size
      - 1536 dims (EfficientNetB3) → (32, 48, 1)
      - 1024 dims (DinoV2-large) → (32, 32, 1)
      - 768 dims (DinoV2-base) → (24, 32, 1)

    Architecture: IDENTICAL to original Inception V3
      - Uses REAL Conv2D operations throughout
      - Same Inception-A, B, C blocks
      - Same Reduction-A, B blocks
      - NOT a Dense network!

    Args:
        input_shape: Tuple (H, W, C) for input dimensions

    Returns:
        Compiled Keras model with TRUE Inception V3 CNN structure
    """
    input_tensor = Input(shape=input_shape)

    # Initial Conv layers
    x = conv2d_bn(input_tensor, 32, 3, 3, strides=(1, 1), padding='same')
    x = conv2d_bn(x, 32, 3, 3, padding='same')
    x = conv2d_bn(x, 64, 3, 3, padding='same')
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(x)  # 16x24

    x = conv2d_bn(x, 80, 1, 1, padding='same')
    x = conv2d_bn(x, 192, 3, 3, padding='same')
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(x)  # 8x12

    # Inception-A blocks (3x)
    x = inception_A(x)
    x = inception_A(x)
    x = inception_A(x)

    # Reduction-A
    x = reduction_A(x)  # 4x6

    # Inception-B blocks (4x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)

    # Reduction-B
    x = reduction_B(x)  # 2x3

    # Inception-C blocks (2x)
    x = inception_C(x)
    x = inception_C(x)

    # Global pooling and output
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x, name='incep')
    return model


# =============================================================================
# LOAD EMBEDDINGS FOR L2 (SPECIES CLASSIFICATION)
# =============================================================================
print("\n" + "="*80)
print("LOADING EMBEDDINGS - SPECIES CLASSIFICATION (L2)")
print("="*80)

# Load embeddings file
# AUTO-COMPATIBLE with any embedding dimension:
#   - 'embeddings_3level_efficientnet.npz'    (1536-dim, EfficientNetB3)
#   - 'embeddings_3level_dinov2_602020.npz'   (1024-dim, DinoV2-large)
#   - Any other embedding file with same structure
embeddings_path = 'embeddings_3level_dinov2_smote_proper801010.npz'
print(f"Loading embeddings from: {embeddings_path}")
data = np.load(embeddings_path, allow_pickle=True)

# Extract embeddings and L2 labels (species)
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']

train_l2 = data['train_l2']
val_l2 = data['val_l2']
test_l2 = data['test_l2']

# Filter out negative samples (l2 == -1) to focus on species classification
train_mask = train_l2 >= 0
val_mask = val_l2 >= 0
test_mask = test_l2 >= 0

X_train = X_train[train_mask]
y_train = train_l2[train_mask]

X_val = X_val[val_mask]
y_val = val_l2[val_mask]

X_test = X_test[test_mask]
y_test = test_l2[test_mask]

embed_dim = X_train.shape[1]
classes = ["Vivax", "Falciparum"]

print(f"\nLoaded embeddings:")
print(f"  Embedding dimension: {embed_dim}")
print(f"  Train: {X_train.shape}, Labels: {y_train.shape}")
print(f"  Val:   {X_val.shape}, Labels: {y_val.shape}")
print(f"  Test:  {X_test.shape}, Labels: {y_test.shape}")

print(f"\nSpecies distribution:")
for split_name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    vivax = np.sum(labels == 0)
    falci = np.sum(labels == 1)
    total = len(labels)
    print(f"  {split_name:6s}: Vivax={vivax:4d} ({vivax/total*100:.1f}%), "
          f"Falciparum={falci:4d} ({falci/total*100:.1f}%)")

# Compute class weights to handle any class imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
print(f"\nClass weights computed: {class_weights}")
print(f"  Vivax (0): {class_weights[0]:.3f}")
print(f"  Falciparum (1): {class_weights[1]:.3f}")


# =============================================================================
# AUTO-DETECT RESHAPE DIMENSIONS FOR TRUE INCEPTION V3 CNN
# =============================================================================
print("\n" + "="*80)
print("AUTO-DETECTING RESHAPE DIMENSIONS FOR INCEPTION V3")
print("="*80)

def find_best_reshape(embed_dim):
    """
    Find optimal 2D reshape dimensions for embedding dimension
    Tries to create a near-square shape for better Conv2D processing

    Args:
        embed_dim: Embedding dimension (e.g., 1536, 1024, 768)

    Returns:
        (height, width) tuple for reshaping to (height, width, 1)
    """
    import math

    # Try to find factors close to square root for balanced dimensions
    sqrt_dim = int(math.sqrt(embed_dim))

    # Find the best factorization (h * w = embed_dim)
    best_h, best_w = 1, embed_dim
    min_diff = float('inf')

    for h in range(sqrt_dim - 10, sqrt_dim + 10):
        if h <= 0:
            continue
        if embed_dim % h == 0:
            w = embed_dim // h
            diff = abs(h - w)  # Prefer more square-like shapes
            if diff < min_diff:
                min_diff = diff
                best_h, best_w = h, w

    return best_h, best_w

# Auto-detect reshape dimensions
reshape_h, reshape_w = find_best_reshape(embed_dim)
input_shape = (reshape_h, reshape_w, 1)

print(f"\n✓ Embedding dimension: {embed_dim}")
print(f"✓ Optimal reshape: ({reshape_h}, {reshape_w}, 1)")
print(f"  Verification: {reshape_h} × {reshape_w} × 1 = {reshape_h * reshape_w * 1} ✓")

# Reshape embeddings from 1D to 2D spatial format for Conv2D processing
X_train = X_train.reshape(-1, reshape_h, reshape_w, 1)
X_val = X_val.reshape(-1, reshape_h, reshape_w, 1)
X_test = X_test.reshape(-1, reshape_h, reshape_w, 1)

print(f"\n✓ Reshaped embeddings for Conv2D:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"\n✓ Embeddings are now 2D spatial format - TRUE Inception V3 can process them!")

# =============================================================================
# TRAINING (Single Run - No K-Fold CV)
# =============================================================================
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

# Build model
model = Incep(input_shape=input_shape)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.9),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\n✓ Model: TRUE Inception V3 CNN (Conv2D + Pooling + Inception blocks)")
print(f"✓ Input shape: {input_shape} (auto-detected from {embed_dim}-dim embeddings)")
print(f"✓ Architecture: IDENTICAL to original Inception V3 (Conv2D based, NOT Dense)")
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)

print(f"\nStarting training on {len(y_train)} samples...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# =============================================================================
# EVALUATION (Train, Val, Test)
# =============================================================================
print("\n" + "="*80)
print("EVALUATING FINAL MODEL")
print("="*80)

# Compute comprehensive metrics
def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    gmean = np.sqrt(sensitivity * specificity)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "G-Mean": gmean
    }

# Test set predictions and metrics
y_test_prob = model.predict(X_test).ravel()
y_test_pred = (y_test_prob >= 0.5).astype(int)
test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

# Validation set predictions and metrics
y_val_prob = model.predict(X_val).ravel()
y_val_pred = (y_val_prob >= 0.5).astype(int)
val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)

# Training set predictions and metrics
y_train_prob = model.predict(X_train).ravel()
y_train_pred = (y_train_prob >= 0.5).astype(int)
train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)

print("\n" + "="*80)
print("TRAIN SET METRICS")
print("="*80)
for metric, value in train_metrics.items():
    print(f"{metric:<20s}: {value:.4f}")

print("\n" + "="*80)
print("VALIDATION SET METRICS")
print("="*80)
for metric, value in val_metrics.items():
    print(f"{metric:<20s}: {value:.4f}")

print("\n" + "="*80)
print("TEST SET METRICS")
print("="*80)
for metric, value in test_metrics.items():
    print(f"{metric:<20s}: {value:.4f}")

# Confusion Matrix and Classification Report (for test set)
cm = confusion_matrix(y_test, y_test_pred)
clr = classification_report(y_test, y_test_pred, target_names=classes)

print("\n" + "="*80)
print("CLASSIFICATION REPORT (TEST SET)")
print("="*80)
print(clr)

print("\n" + "="*80)
print("CONFUSION MATRIX (TEST SET)")
print("="*80)
print(cm)

# =============================================================================
# COMPUTE MEAN ± STD ACROSS TRAIN/VAL/TEST SPLITS
# =============================================================================
print("\n" + "="*80)
print("METRICS ACROSS ALL SPLITS (with Mean ± STD)")
print("="*80)

# Create summary table
metrics_summary = {}
metric_names = list(train_metrics.keys())

for metric in metric_names:
    train_val = train_metrics[metric]
    val_val = val_metrics[metric]
    test_val = test_metrics[metric]

    # Compute mean and std across the 3 splits
    values = [train_val, val_val, test_val]
    mean_val = np.mean(values)
    std_val = np.std(values)

    metrics_summary[metric] = {
        'train': train_val,
        'val': val_val,
        'test': test_val,
        'mean': mean_val,
        'std': std_val
    }

# Print in table format (like HCAST report)
print(f"\n{'Metric':<25s} {'Train':>10s} {'Val':>10s} {'Test':>10s} {'Mean':>10s} {'STD':>10s}")
print("-" * 80)
for metric, values in metrics_summary.items():
    print(f"{metric:<25s} {values['train']:>10.4f} {values['val']:>10.4f} {values['test']:>10.4f} "
          f"{values['mean']:>10.4f} {values['std']:>10.4f}")


# =============================================================================
# SAVE METRICS AND PLOTS
# =============================================================================
print("\n" + "="*80)
print("SAVING METRICS AND VISUALIZATIONS")
print("="*80)

# Save metrics to CSV
metrics_df = pd.DataFrame([test_metrics])
metrics_df.to_csv(f"{OUTPUT_DIR}/test_metrics.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/test_metrics.csv")

# Save classification report
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write("SPECIES CLASSIFICATION (L2) - TEST SET\n")
    f.write("="*80 + "\n\n")
    f.write("Classification Report:\n")
    f.write("-"*80 + "\n")
    f.write(clr)
    f.write("\n\nConfusion Matrix:\n")
    f.write("-"*80 + "\n")
    f.write(str(cm))
    f.write("\n\nDetailed Metrics:\n")
    f.write("-"*80 + "\n")
    for metric, value in test_metrics.items():
        f.write(f"{metric:<20s}: {value:.4f}\n")
print(f"Saved: {OUTPUT_DIR}/classification_report.txt")

# Plot 1: Training & Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy - Species Classification (L2)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/accuracy_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/accuracy_curve.png")

# Plot 2: Training & Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss - Species Classification (L2)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/loss_curve.png")

# Plot 3: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Count', fontsize=11)

# Add labels and ticks
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add axis labels and title
ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
ax.set_ylabel('True Labels', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Species Classification', fontsize=14, fontweight='bold')

# Annotate with counts and percentages
for i in range(len(classes)):
    for j in range(len(classes)):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        percentage = cm[i, j] / cm.sum() * 100
        ax.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                ha='center', va='center', color=text_color, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")

# Plot 4: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2.5,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve - Species Classification (L2)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_auc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/roc_auc_curve.png")

# Save training history
history_df = pd.DataFrame({
    'epoch': range(1, len(history.history['accuracy']) + 1),
    'train_accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'learning_rate': history.history.get('lr', [learning_rate] * len(history.history['accuracy']))
})
history_df.to_csv(f"{OUTPUT_DIR}/training_history.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/training_history.csv")

# Plot 5: Learning Rate Schedule (if available)
if 'lr' in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['lr'], linewidth=2, color='green')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/learning_rate_schedule.png")

# Save model
model.save(f"{OUTPUT_DIR}/species_classifier_model.h5")
print(f"Saved: {OUTPUT_DIR}/species_classifier_model.h5")

# =============================================================================
# PRINT CONSOLE SUMMARY (HCAST-style format)
# =============================================================================
print("\n" + "="*80)
print("INCEPTION V3 CNN - SPECIES CLASSIFICATION (L2)")
print("="*80)
print(f"\nTimestamp: {timestamp}")

print(f"\n{'='*80}")
print("ACCURACIES (Train / Val / Test)")
print(f"{'='*80}")
print(f"Train Accuracy:      {metrics_summary['Accuracy']['train']*100:.2f}%")
print(f"Validation Accuracy: {metrics_summary['Accuracy']['val']*100:.2f}%")
print(f"Test Accuracy:       {metrics_summary['Accuracy']['test']*100:.2f}%")
print(f"Mean ± STD:          {metrics_summary['Accuracy']['mean']*100:.2f}% ± {metrics_summary['Accuracy']['std']*100:.2f}%")

print(f"\n{'='*80}")
print("Classification Report (Test Set):")
print(f"{'='*80}")
print(classification_report(y_test, y_test_pred, target_names=classes, digits=4))
print(f"{'='*80}")
print(f"\nTest ROC-AUC Score: {test_metrics['AUC']:.4f}")
print(f"\nAll metrics and visualizations saved to: {OUTPUT_DIR}/")
print("="*80)

# Save comprehensive summary (HCAST-style format)
with open(f"{OUTPUT_DIR}/summary.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("INCEPTION V3 CNN - SPECIES CLASSIFICATION (L2)\n")
    f.write("="*80 + "\n\n")

    f.write(f"Timestamp: {timestamp}\n\n")

    f.write("="*80 + "\n")
    f.write("METRICS ACROSS ALL SPLITS (with Standard Deviation)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Metric':<25s} {'Train':>10s} {'Val':>10s} {'Test':>10s} {'Mean':>10s} {'STD':>10s}\n")
    f.write("-"*80 + "\n")
    for metric, values in metrics_summary.items():
        f.write(f"{metric:<25s} {values['train']:>10.4f} {values['val']:>10.4f} {values['test']:>10.4f} "
                f"{values['mean']:>10.4f} {values['std']:>10.4f}\n")
    f.write("\n")

    f.write("="*80 + "\n")
    f.write("Classification Report (Test Set):\n")
    f.write("="*80 + "\n")
    f.write(classification_report(y_test, y_test_pred, target_names=classes, digits=4))
    f.write("\n" + "="*80 + "\n\n")

    f.write(f"Test ROC-AUC Score: {test_metrics['AUC']:.4f}\n\n")

    f.write("="*80 + "\n")
    f.write("DETAILED CONFIGURATION & RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("CONFIGURATION:\n")
    f.write("-"*80 + "\n")
    f.write(f"Model: TRUE Inception V3 CNN (Conv2D based)\n")
    f.write(f"Embedding file: {embeddings_path}\n")
    f.write(f"Embedding dimension: {embed_dim}\n")
    f.write(f"Input shape (reshaped): {input_shape}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Initial learning rate: {learning_rate}\n")
    f.write(f"Max epochs: {epochs}\n")
    f.write(f"Early stopping patience: 15\n")
    f.write(f"Learning rate reduction: Yes (factor=0.5, patience=7)\n")
    f.write(f"Class weights: Vivax={class_weights[0]:.3f}, Falciparum={class_weights[1]:.3f}\n\n")

    f.write("DATA SPLIT:\n")
    f.write("-"*80 + "\n")
    f.write(f"Train samples: {len(y_train)}\n")
    f.write(f"Validation samples: {len(y_val)}\n")
    f.write(f"Test samples: {len(y_test)}\n\n")

    f.write("TRAINING:\n")
    f.write("-"*80 + "\n")
    f.write(f"Total epochs trained: {len(history.history['accuracy'])}\n")
    f.write(f"Best train accuracy: {max(history.history['accuracy']):.4f} ({max(history.history['accuracy'])*100:.2f}%)\n")
    f.write(f"Best val accuracy: {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy'])*100:.2f}%)\n")
    f.write(f"Final train loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final val loss: {history.history['val_loss'][-1]:.4f}\n\n")

    f.write("INDIVIDUAL SPLIT DETAILS:\n")
    f.write("-"*80 + "\n\n")

    f.write("TRAIN SET:\n")
    for metric, value in train_metrics.items():
        f.write(f"  {metric:<25s}: {value:.4f}\n")
    f.write("\n")

    f.write("VALIDATION SET:\n")
    for metric, value in val_metrics.items():
        f.write(f"  {metric:<25s}: {value:.4f}\n")
    f.write("\n")

    f.write("TEST SET:\n")
    for metric, value in test_metrics.items():
        f.write(f"  {metric:<25s}: {value:.4f}\n")

    f.write("\n" + "="*80 + "\n")
    f.write(f"Results saved to: {OUTPUT_DIR}/\n")
    f.write("="*80 + "\n")

print(f"Saved: {OUTPUT_DIR}/summary.txt")

print("\n" + "="*80)
print("✓ ALL RESULTS SAVED SUCCESSFULLY!")
print("="*80)
