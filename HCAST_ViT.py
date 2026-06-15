import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import ops
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
from datetime import datetime

print("\nConfiguring GPU...")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)

print("\nLoading embeddings...")

# Load 3-level embeddings
embedding_data = np.load(
    "/home/ghufran/MalariaML/Species_Classification/ajay/embeddings_3level_dinov2_smote_proper602020.npz"
)

# Extract embeddings
X_train = embedding_data['X_train']
X_val = embedding_data['X_val']
X_test = embedding_data['X_test']

# Extract L1 and L2 labels
train_l1 = embedding_data['train_l1']
train_l2 = embedding_data['train_l2']
val_l1 = embedding_data['val_l1']
val_l2 = embedding_data['val_l2']
test_l1 = embedding_data['test_l1']
test_l2 = embedding_data['test_l2']

print(f"\nEmbedding shapes:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# Create L2 flat labels (3-class species classification)
# Class 0: Negative (l1=0)
# Class 1: Vivax (l1=1, l2=0)
# Class 2: Falciparum (l1=1, l2=1)
def create_l2_labels(l1, l2):
    """Convert L1 (infection) + L2 (species) to flat 3-class labels"""
    labels = np.zeros(len(l1), dtype=np.int32)
    labels[l1 == 0] = 0  # Negative
    labels[(l1 == 1) & (l2 == 0)] = 1  # Vivax
    labels[(l1 == 1) & (l2 == 1)] = 2  # Falciparum
    return labels

y_flat_train = create_l2_labels(train_l1, train_l2)
y_flat_val = create_l2_labels(val_l1, val_l2)
y_flat_test = create_l2_labels(test_l1, test_l2)

print(f"\nClass distribution before SMOTE:")
for i, class_name in enumerate(['Negative', 'Vivax', 'Falciparum']):
    count = np.sum(y_flat_train == i)
    print(f"  {class_name:12s}: {count:5d} ({count/len(y_flat_train)*100:.1f}%)")

# Apply SMOTE to balance training data
print(f"\nApplying SMOTE to balance training data...")
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train, y_flat_train = smote.fit_resample(X_train, y_flat_train)

print(f"\nClass distribution after SMOTE:")
for i, class_name in enumerate(['Negative', 'Vivax', 'Falciparum']):
    count = np.sum(y_flat_train == i)
    print(f"  {class_name:12s}: {count:5d} ({count/len(y_flat_train)*100:.1f}%)")

print(f"\nFinal dataset sizes:")
print(f"  Training samples:   {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")
print(f"  Test samples:       {len(X_test):,}")

EMBED_DIM = X_train.shape[1]
def make_hierarchical_labels(y_flat):
    """
    Level 1: Infection (0=Negative, 1=Positive)
    Level 2: Species (0=Vivax, 1=Falciparum, -1 for Negative)
    """
    y_l1 = (y_flat != 0).astype(np.float32)

    y_l2 = np.full_like(y_flat, -1, dtype=np.float32)
    y_l2[y_flat == 1] = 0  # Vivax
    y_l2[y_flat == 2] = 1  # Falciparum

    return y_l1, y_l2


y_l1_train, y_l2_train = make_hierarchical_labels(y_flat_train)
y_l1_val, y_l2_val = make_hierarchical_labels(y_flat_val)
y_l1_test, y_l2_test = make_hierarchical_labels(y_flat_test)

# Create sample weights: Level 2 should only train on positive samples
train_l2_weights = y_l1_train.copy()  # 1.0 for positive, 0.0 for negative
val_l2_weights = y_l1_val.copy()

# Replace -1 with 0 in Level 2 labels (will be masked by weights)
y_l2_train[y_l2_train == -1] = 0
y_l2_val[y_l2_val == -1] = 0
y_l2_test_original = y_l2_test.copy()  # Keep original for evaluation
y_l2_test[y_l2_test == -1] = 0

NUM_PATCHES = 8
PATCH_DIM = 256
NUM_HEADS = 4
TRANSFORMER_BLOCKS = 2
MLP_DIM = 512
DROPOUT = 0.3

class EmbeddingPatcher(layers.Layer):
    def __init__(self, num_patches, patch_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.proj = layers.Dense(patch_dim)

    def call(self, x):
        batch = ops.shape(x)[0]
        patch_size = x.shape[-1] // self.num_patches
        x = ops.reshape(x, (batch, self.num_patches, patch_size))
        return self.proj(x)


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(num_patches, dim)

    def call(self, x):
        positions = ops.arange(ops.shape(x)[1])
        return x + self.pos_emb(positions)


def transformer_block(x):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=PATCH_DIM,
        dropout=DROPOUT
    )(x1, x1)
    x2 = layers.Add()([x, attn])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    mlp = layers.Dense(MLP_DIM, activation="gelu")(x3)
    mlp = layers.Dropout(DROPOUT)(mlp)
    mlp = layers.Dense(PATCH_DIM)(mlp)

    return layers.Add()([x2, mlp])

class TreePathConsistencyLayer(layers.Layer):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        p_l1, p_l2 = inputs

        # H-CAST tree-path constraint
        loss = self.alpha * ops.mean(
            ops.maximum(0.0, p_l2 - p_l1)
        )

        self.add_loss(loss)
        return inputs


inputs = layers.Input(shape=(EMBED_DIM,), name="embedding_input")

x = EmbeddingPatcher(NUM_PATCHES, PATCH_DIM)(inputs)
x = PatchEncoder(NUM_PATCHES, PATCH_DIM)(x)

for _ in range(TRANSFORMER_BLOCKS):
    x = transformer_block(x)

x = layers.LayerNormalization(epsilon=1e-6)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.4)(x)

features = layers.Dense(256, activation="gelu")(x)
features = layers.Dropout(0.3)(features)
features = layers.Dense(128, activation="gelu")(features)

# Hierarchical heads
# Raw heads
l1_raw = layers.Dense(1, activation="sigmoid")(features)
l2_raw = layers.Dense(1, activation="sigmoid")(features)

# Apply H-CAST tree-path constraint
l1_cons, l2_cons = TreePathConsistencyLayer(alpha=0.5)(
    [l1_raw, l2_raw]
)

# 🔑 Restore explicit output names
l1_out = layers.Identity(name="level1")(l1_cons)
l2_out = layers.Identity(name="level2")(l2_cons)

model = keras.Model(
    inputs=inputs,
    outputs=[l1_out, l2_out],
    name="ViT_HCAST"
)


model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={
        "level1": "binary_crossentropy",
        "level2": "binary_crossentropy",
    },
    metrics={
        "level1": "accuracy",
        "level2": "accuracy",
    }
)


print("\nTraining H-CAST–inspired ViT...")
start = time.time()

# Use list format instead of dict to avoid KeyError
history = model.fit(
    x=X_train,
    y=[y_l1_train, y_l2_train],
    sample_weight=[np.ones(len(y_l1_train)), train_l2_weights],
    validation_data=(X_val, [y_l1_val, y_l2_val]),
    epochs=50,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

print(f"\nTraining time: {(time.time() - start)/60:.1f} minutes")

training_time = (time.time() - start) / 60

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE OUTPUT DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
workspace_dir = '/home/ghufran/MalariaML/Species_Classification/ajay'
output_dir = os.path.join(workspace_dir, f'ViTHCAST/60-20-20SMOTE/ViT_HCAST_ResultsSMOTE_dinov2_602020_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

confusion_dir = os.path.join(output_dir, 'confusion_matrices')
os.makedirs(confusion_dir, exist_ok=True)

roc_dir = os.path.join(output_dir, 'roc_curves')
os.makedirs(roc_dir, exist_ok=True)

training_dir = os.path.join(output_dir, 'training_graphs')
os.makedirs(training_dir, exist_ok=True)

print(f"\nOutput directory created: {output_dir}")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_comprehensive_metrics(y_true, y_pred, y_prob, class_names):
    """
    Compute comprehensive metrics for multi-class classification

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        class_names: List of class names

    Returns:
        Dictionary with all metrics
    """
    # Handle edge cases
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # AUC calculations for multi-class
    auc_macro = 0.0
    auc_micro = 0.0

    try:
        if len(unique_true) > 1:
            # Multi-class AUC
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

            # Ensure y_prob has the right shape
            if y_prob.ndim == 1:
                y_prob_reshaped = np.zeros((len(y_true), len(class_names)))
                y_prob_reshaped[np.arange(len(y_true)), y_pred] = y_prob
                y_prob = y_prob_reshaped

            # Compute AUC for each class
            auc_scores = []
            for i in range(len(class_names)):
                if i in unique_true:  # Only compute AUC for classes present in y_true
                    try:
                        auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                        auc_scores.append(auc_score)
                    except:
                        pass

            if auc_scores:
                auc_macro = np.mean(auc_scores)

            # Micro-average AUC
            try:
                auc_micro = roc_auc_score(y_true_bin.ravel(), y_prob.ravel())
            except:
                auc_micro = 0.0
    except:
        pass

    # Confusion matrix derived metrics
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class sensitivity and specificity
    sensitivities = []
    specificities = []

    for i in range(len(class_names)):
        if i in unique_true:
            tp = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
            fn = np.sum(cm[i, :]) - tp if i < cm.shape[0] else 0
            fp = np.sum(cm[:, i]) - tp if i < cm.shape[1] else 0
            tn = np.sum(cm) - tp - fn - fp

            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)

            sensitivities.append(sensitivity)
            specificities.append(specificity)

    sensitivity_avg = np.mean(sensitivities) if sensitivities else 0.0
    specificity_avg = np.mean(specificities) if specificities else 0.0
    gmean = np.sqrt(sensitivity_avg * specificity_avg)

    return {
        "Accuracy": accuracy,
        "Precision_Macro": precision_macro,
        "Precision_Micro": precision_micro,
        "Recall_Macro": recall_macro,
        "Recall_Micro": recall_micro,
        "F1_Macro": f1_macro,
        "F1_Micro": f1_micro,
        "Sensitivity_Avg": sensitivity_avg,
        "Specificity_Avg": specificity_avg,
        "Balanced_Accuracy": balanced_acc,
        "Matthews_Corrcoef": mcc,
        "Cohens_Kappa": kappa,
        "AUC_Macro": auc_macro,
        "AUC_Micro": auc_micro,
        "G_Mean": gmean
    }

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION - PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

# Get predictions for all splits
print("\nPredicting on train set...")
p_l1_train, p_l2_train = model.predict(X_train, verbose=0)
pred_l1_train = (p_l1_train > 0.5).astype(int).flatten()
pred_l2_train = (p_l2_train > 0.5).astype(int).flatten()

y_flat_pred_train = np.zeros(len(pred_l1_train), dtype=int)
y_flat_pred_train[pred_l1_train == 0] = 0
y_flat_pred_train[(pred_l1_train == 1) & (pred_l2_train == 0)] = 1
y_flat_pred_train[(pred_l1_train == 1) & (pred_l2_train == 1)] = 2

print("Predicting on validation set...")
p_l1_val, p_l2_val = model.predict(X_val, verbose=0)
pred_l1_val = (p_l1_val > 0.5).astype(int).flatten()
pred_l2_val = (p_l2_val > 0.5).astype(int).flatten()

y_flat_pred_val = np.zeros(len(pred_l1_val), dtype=int)
y_flat_pred_val[pred_l1_val == 0] = 0
y_flat_pred_val[(pred_l1_val == 1) & (pred_l2_val == 0)] = 1
y_flat_pred_val[(pred_l1_val == 1) & (pred_l2_val == 1)] = 2

print("Predicting on test set...")
p_l1_test, p_l2_test = model.predict(X_test, verbose=0)
pred_l1_test = (p_l1_test > 0.5).astype(int).flatten()
pred_l2_test = (p_l2_test > 0.5).astype(int).flatten()

y_flat_pred_test = np.zeros(len(pred_l1_test), dtype=int)
y_flat_pred_test[pred_l1_test == 0] = 0
y_flat_pred_test[(pred_l1_test == 1) & (pred_l2_test == 0)] = 1
y_flat_pred_test[(pred_l1_test == 1) & (pred_l2_test == 1)] = 2

# Calculate accuracies
train_acc = accuracy_score(y_flat_train, y_flat_pred_train)
val_acc = accuracy_score(y_flat_val, y_flat_pred_val)
test_acc = accuracy_score(y_flat_test, y_flat_pred_test)

print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Val Accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

# Calculate prediction probabilities for ROC curves
# For 3-class ROC, we need probabilities for each class
def get_class_probs(p_l1, p_l2):
    """Convert binary predictions to 3-class probabilities"""
    probs = np.zeros((len(p_l1), 3))
    probs[:, 0] = 1 - p_l1.flatten()  # Negative
    probs[:, 1] = p_l1.flatten() * (1 - p_l2.flatten())  # Vivax
    probs[:, 2] = p_l1.flatten() * p_l2.flatten()  # Falciparum
    return probs

train_pred_probs = get_class_probs(p_l1_train, p_l2_train)
val_pred_probs = get_class_probs(p_l1_val, p_l2_val)
test_pred_probs = get_class_probs(p_l1_test, p_l2_test)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating confusion matrices...")

CLASS_NAMES = ['Negative', 'Vivax', 'Falciparum']

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add percentage annotations
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j + 0.5, i + 0.7, f'({cm_percentage[i, j]:.1f}%)',
                          ha="center", va="center", color="gray", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_confusion_matrices(train_pred, val_pred, test_pred,
                                   train_true, val_true, test_true,
                                   class_names, save_path):
    """Create comprehensive confusion matrix plot with all splits"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('H-CAST ViT Confusion Matrices - All Splits', fontsize=16, fontweight='bold', y=1.02)

    splits = [
        ('Train', train_true, train_pred, axes[0]),
        ('Validation', val_true, val_pred, axes[1]),
        ('Test', test_true, test_pred, axes[2])
    ]

    for split_name, y_true, y_pred, ax in splits:
        cm = confusion_matrix(y_true, y_pred)

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, ax=ax)

        # Add percentage annotations
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j + 0.5, i + 0.7, f'({cm_percentage[i, j]:.1f}%)',
                       ha="center", va="center", color="gray", fontsize=9)

        # Labels and title
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('True', fontsize=11, fontweight='bold')
        ax.set_title(f'{split_name} Set', fontsize=12, fontweight='bold')

        # Add accuracy info
        acc = accuracy_score(y_true, y_pred)
        ax.text(0.5, -0.15, f'Accuracy: {acc*100:.2f}% (n={len(y_true):,})',
               ha='center', transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

plot_confusion_matrix(y_flat_train, y_flat_pred_train, CLASS_NAMES,
                     'H-CAST ViT Confusion Matrix - Train Set',
                     os.path.join(confusion_dir, 'train_confusion_matrix.png'))

plot_confusion_matrix(y_flat_val, y_flat_pred_val, CLASS_NAMES,
                     'H-CAST ViT Confusion Matrix - Validation Set',
                     os.path.join(confusion_dir, 'val_confusion_matrix.png'))

plot_confusion_matrix(y_flat_test, y_flat_pred_test, CLASS_NAMES,
                     'H-CAST ViT Confusion Matrix - Test Set',
                     os.path.join(confusion_dir, 'test_confusion_matrix.png'))

# Generate comprehensive combined confusion matrix
plot_combined_confusion_matrices(y_flat_pred_train, y_flat_pred_val, y_flat_pred_test,
                               y_flat_train, y_flat_val, y_flat_test,
                               CLASS_NAMES,
                               os.path.join(output_dir, 'confusion_matrices_all.png'))

print(f"  - Saved: {confusion_dir}/train_confusion_matrix.png")
print(f"  - Saved: {confusion_dir}/val_confusion_matrix.png")
print(f"  - Saved: {confusion_dir}/test_confusion_matrix.png")
print(f"  - Saved: {output_dir}/confusion_matrices_all.png")

# ═══════════════════════════════════════════════════════════════════════════════
# ROC CURVES AND AUC
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating ROC curves and AUC scores...")

def plot_roc_curves(y_true, y_pred_probs, class_names, title, save_path):
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(class_names)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # Plot ROC curve for each class
    for i, color in zip(range(len(class_names)), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    # Plot micro and macro averages
    ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    ax.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=3,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return roc_auc

train_auc = plot_roc_curves(y_flat_train, train_pred_probs, CLASS_NAMES,
                            'H-CAST ViT ROC Curves - Train Set',
                            os.path.join(roc_dir, 'train_roc_curves.png'))

val_auc = plot_roc_curves(y_flat_val, val_pred_probs, CLASS_NAMES,
                          'H-CAST ViT ROC Curves - Validation Set',
                          os.path.join(roc_dir, 'val_roc_curves.png'))

test_auc = plot_roc_curves(y_flat_test, test_pred_probs, CLASS_NAMES,
                           'H-CAST ViT ROC Curves - Test Set',
                           os.path.join(roc_dir, 'test_roc_curves.png'))

print(f"  - Saved: {roc_dir}/train_roc_curves.png")
print(f"  - Saved: {roc_dir}/val_roc_curves.png")
print(f"  - Saved: {roc_dir}/test_roc_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HISTORY PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating training history plots...")

# Plot Level 1 training history
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Level 1 Accuracy
axes[0, 0].plot(history.history['level1_accuracy'], label='Train', linewidth=2, color='#3498db')
axes[0, 0].plot(history.history['val_level1_accuracy'], label='Validation', linewidth=2, color='#e74c3c')
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Level 1 (Infection) Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Level 1 Loss
axes[0, 1].plot(history.history['level1_loss'], label='Train', linewidth=2, color='#3498db')
axes[0, 1].plot(history.history['val_level1_loss'], label='Validation', linewidth=2, color='#e74c3c')
axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Level 1 (Infection) Loss', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Level 2 Accuracy
axes[1, 0].plot(history.history['level2_accuracy'], label='Train', linewidth=2, color='#3498db')
axes[1, 0].plot(history.history['val_level2_accuracy'], label='Validation', linewidth=2, color='#e74c3c')
axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Level 2 (Species) Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Level 2 Loss
axes[1, 1].plot(history.history['level2_loss'], label='Train', linewidth=2, color='#3498db')
axes[1, 1].plot(history.history['val_level2_loss'], label='Validation', linewidth=2, color='#e74c3c')
axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Level 2 (Species) Loss', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(training_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  - Saved: {training_dir}/training_history.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION REPORTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("CLASSIFICATION REPORTS")
print("="*80)

print("\nTRAIN SET:")
print(classification_report(y_flat_train, y_flat_pred_train, target_names=CLASS_NAMES, digits=4))

print("\nVALIDATION SET:")
print(classification_report(y_flat_val, y_flat_pred_val, target_names=CLASS_NAMES, digits=4))

print("\nTEST SET:")
print(classification_report(y_flat_test, y_flat_pred_test, target_names=CLASS_NAMES, digits=4))

# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputing comprehensive metrics for all splits...")

# Compute comprehensive metrics for each split
train_metrics = compute_comprehensive_metrics(y_flat_train, y_flat_pred_train, train_pred_probs, CLASS_NAMES)
val_metrics = compute_comprehensive_metrics(y_flat_val, y_flat_pred_val, val_pred_probs, CLASS_NAMES)
test_metrics = compute_comprehensive_metrics(y_flat_test, y_flat_pred_test, test_pred_probs, CLASS_NAMES)

# Calculate standard deviations across splits for each metric
print("Computing standard deviations across splits...")
metrics_std = {}
metrics_mean = {}

for metric_name in train_metrics.keys():
    values = [train_metrics[metric_name], val_metrics[metric_name], test_metrics[metric_name]]
    metrics_std[metric_name] = np.std(values)
    metrics_mean[metric_name] = np.mean(values)

print("\n" + "="*80)
print("COMPREHENSIVE METRICS SUMMARY")
print("="*80)

print(f"\n{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12} {'Mean':<12} {'STD':<12}")
print("-" * 80)

for metric_name in train_metrics.keys():
    train_val = train_metrics[metric_name]
    val_val = val_metrics[metric_name]
    test_val = test_metrics[metric_name]
    mean_val = metrics_mean[metric_name]
    std_val = metrics_std[metric_name]

    print(f"{metric_name:<20} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f} "
          f"{mean_val:>12.4f} {std_val:>12.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL METRICS (BACKWARD COMPATIBILITY)
# ═══════════════════════════════════════════════════════════════════════════════

# Extract key metrics for backward compatibility
train_kappa = train_metrics["Cohens_Kappa"]
train_mcc = train_metrics["Matthews_Corrcoef"]
val_kappa = val_metrics["Cohens_Kappa"]
val_mcc = val_metrics["Matthews_Corrcoef"]
test_kappa = test_metrics["Cohens_Kappa"]
test_mcc = test_metrics["Matthews_Corrcoef"]

print("\n" + "="*80)
print("ADDITIONAL METRICS")
print("="*80)

print(f"\nTrain Set:")
print(f"  Cohen's Kappa:      {train_kappa:.4f}")
print(f"  Matthews Corr Coef: {train_mcc:.4f}")

print(f"\nValidation Set:")
print(f"  Cohen's Kappa:      {val_kappa:.4f}")
print(f"  Matthews Corr Coef: {val_mcc:.4f}")

print(f"\nTest Set:")
print(f"  Cohen's Kappa:      {test_kappa:.4f}")
print(f"  Matthews Corr Coef: {test_mcc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE METRICS JSON
# ═══════════════════════════════════════════════════════════════════════════════

metrics_data = {
    "model_config": {
        "architecture": "Vision Transformer with H-CAST (Hierarchical Consistency)",
        "embedding_dim": EMBED_DIM,
        "num_patches": NUM_PATCHES,
        "patch_dim": PATCH_DIM,
        "num_heads": NUM_HEADS,
        "transformer_blocks": TRANSFORMER_BLOCKS,
        "mlp_dim": MLP_DIM,
        "dropout": DROPOUT,
        "tree_path_alpha": 0.5
    },
    "training": {
        "time_minutes": float(training_time),
        "epochs_run": len(history.history['loss']),
        "batch_size": 64
    },
    "comprehensive_metrics": {
        "train": {k: float(v) for k, v in train_metrics.items()},
        "val": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
        "std": {k: float(v) for k, v in metrics_std.items()},
        "mean": {k: float(v) for k, v in metrics_mean.items()}
    },
    "train": {
        "accuracy": float(train_acc),
        "cohens_kappa": float(train_kappa),
        "matthews_corrcoef": float(train_mcc),
        "auc_micro": float(train_auc["micro"]),
        "auc_macro": float(train_auc["macro"]),
        "auc_per_class": {CLASS_NAMES[i]: float(train_auc[i]) for i in range(3)},
        "samples": len(y_flat_train)
    },
    "val": {
        "accuracy": float(val_acc),
        "cohens_kappa": float(val_kappa),
        "matthews_corrcoef": float(val_mcc),
        "auc_micro": float(val_auc["micro"]),
        "auc_macro": float(val_auc["macro"]),
        "auc_per_class": {CLASS_NAMES[i]: float(val_auc[i]) for i in range(3)},
        "samples": len(y_flat_val)
    },
    "test": {
        "accuracy": float(test_acc),
        "cohens_kappa": float(test_kappa),
        "matthews_corrcoef": float(test_mcc),
        "auc_micro": float(test_auc["micro"]),
        "auc_macro": float(test_auc["macro"]),
        "auc_per_class": {CLASS_NAMES[i]: float(test_auc[i]) for i in range(3)},
        "samples": len(y_flat_test)
    },
    "timestamp": timestamp,
    "class_names": CLASS_NAMES
}

metrics_path = os.path.join(output_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_data, f, indent=4)

print(f"\nMetrics saved: {metrics_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE COMPREHENSIVE METRICS CSV
# ═══════════════════════════════════════════════════════════════════════════════

print("Saving comprehensive metrics to CSV...")
import csv

metrics_csv_path = os.path.join(output_dir, 'comprehensive_metrics.csv')
with open(metrics_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Header
    writer.writerow(['Metric', 'Train', 'Val', 'Test', 'Mean', 'STD'])
    writer.writerow([])  # Empty row for spacing

    # Write all comprehensive metrics
    for metric_name in train_metrics.keys():
        train_val = train_metrics[metric_name]
        val_val = val_metrics[metric_name]
        test_val = test_metrics[metric_name]
        mean_val = metrics_mean[metric_name]
        std_val = metrics_std[metric_name]

        writer.writerow([metric_name, f'{train_val:.4f}', f'{val_val:.4f}',
                        f'{test_val:.4f}', f'{mean_val:.4f}', f'{std_val:.4f}'])

print(f"Comprehensive metrics CSV saved: {metrics_csv_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE DETAILED REPORT
# ═══════════════════════════════════════════════════════════════════════════════

report_path = os.path.join(output_dir, 'detailed_report.txt')

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("VISION TRANSFORMER WITH H-CAST (HIERARCHICAL CONSISTENCY)\n")
    f.write("="*80 + "\n\n")

    f.write("Model Configuration:\n")
    f.write(f"  - Architecture: Vision Transformer with H-CAST\n")
    f.write(f"  - Embedding Dimension: {EMBED_DIM}\n")
    f.write(f"  - Number of Patches: {NUM_PATCHES}\n")
    f.write(f"  - Patch Dimension: {PATCH_DIM}\n")
    f.write(f"  - Number of Heads: {NUM_HEADS}\n")
    f.write(f"  - Transformer Blocks: {TRANSFORMER_BLOCKS}\n")
    f.write(f"  - MLP Dimension: {MLP_DIM}\n")
    f.write(f"  - Dropout Rate: {DROPOUT}\n")
    f.write(f"  - Tree-Path Alpha: 0.5\n")
    f.write(f"  - Training Time: {training_time:.2f} minutes\n\n")

    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Classes: {CLASS_NAMES}\n\n")

    f.write("="*80 + "\n")
    f.write("TRAIN SET RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Accuracy:           {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"Cohen's Kappa:      {train_kappa:.4f}\n")
    f.write(f"Matthews Corr Coef: {train_mcc:.4f}\n")
    f.write(f"AUC (Micro):        {train_auc['micro']:.4f}\n")
    f.write(f"AUC (Macro):        {train_auc['macro']:.4f}\n")
    f.write(f"Samples:            {len(y_flat_train):,}\n\n")

    f.write("="*80 + "\n")
    f.write("VALIDATION SET RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Accuracy:           {val_acc:.4f} ({val_acc*100:.2f}%)\n")
    f.write(f"Cohen's Kappa:      {val_kappa:.4f}\n")
    f.write(f"Matthews Corr Coef: {val_mcc:.4f}\n")
    f.write(f"AUC (Micro):        {val_auc['micro']:.4f}\n")
    f.write(f"AUC (Macro):        {val_auc['macro']:.4f}\n")
    f.write(f"Samples:            {len(y_flat_val):,}\n\n")

    f.write("="*80 + "\n")
    f.write("TEST SET RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Accuracy:           {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"Cohen's Kappa:      {test_kappa:.4f}\n")
    f.write(f"Matthews Corr Coef: {test_mcc:.4f}\n")
    f.write(f"AUC (Micro):        {test_auc['micro']:.4f}\n")
    f.write(f"AUC (Macro):        {test_auc['macro']:.4f}\n")
    f.write(f"Samples:            {len(y_flat_test):,}\n\n")

    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE METRICS SUMMARY (ALL SPLITS)\n")
    f.write("="*80 + "\n\n")

    f.write(f"{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12} {'Mean':<12} {'STD':<12}\n")
    f.write("-" * 80 + "\n")

    for metric_name in train_metrics.keys():
        train_val = train_metrics[metric_name]
        val_val = val_metrics[metric_name]
        test_val = test_metrics[metric_name]
        mean_val = metrics_mean[metric_name]
        std_val = metrics_std[metric_name]

        f.write(f"{metric_name:<20} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f} "
               f"{mean_val:>12.4f} {std_val:>12.4f}\n")

    f.write("\n")

    f.write("="*80 + "\n")
    f.write("PER-CLASS AUC SCORES (TEST SET)\n")
    f.write("="*80 + "\n\n")
    for i, class_name in enumerate(CLASS_NAMES):
        f.write(f"{class_name:15s}: {test_auc[i]:.4f}\n")
    f.write(f"\n")

    f.write("="*80 + "\n")
    f.write("DETAILED CLASSIFICATION REPORT (TEST SET)\n")
    f.write("="*80 + "\n\n")
    f.write(classification_report(y_flat_test, y_flat_pred_test, target_names=CLASS_NAMES, digits=4))

    f.write("\n")
    f.write("="*80 + "\n")
    f.write("HIERARCHICAL STRUCTURE\n")
    f.write("="*80 + "\n\n")
    f.write("The model uses H-CAST (Hierarchical Classification with Soft Trees)\n")
    f.write("to enforce tree-path consistency between hierarchical levels:\n\n")
    f.write("  Level 1: Infection Detection (Negative vs Positive)\n")
    f.write("  Level 2: Species Classification (Vivax vs Falciparum)\n\n")
    f.write("Tree-path constraint ensures that Level 2 predictions (species)\n")
    f.write("are consistent with Level 1 predictions (infection status).\n")

print(f"\nDetailed report saved: {report_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

model_path = os.path.join(output_dir, 'vit_hcast_model.keras')
model.save(model_path)
print(f"\nModel saved: {model_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print(f"\nOutput Directory: {output_dir}")
print(f"\nGenerated Files:")
print(f"  - metrics.json")
print(f"  - comprehensive_metrics.csv")
print(f"  - detailed_report.txt")
print(f"  - vit_hcast_model.keras")
print(f"  - confusion_matrices_all.png")
print(f"  - confusion_matrices/ (3 images)")
print(f"  - roc_curves/ (3 images)")
print(f"  - training_graphs/ (1 image)")

print(f"\nFinal Test Set Performance:")
print(f"  Accuracy:           {test_acc*100:.2f}%")
print(f"  Cohen's Kappa:      {test_kappa:.4f}")
print(f"  Matthews Corr Coef: {test_mcc:.4f}")
print(f"  AUC (Micro):        {test_auc['micro']:.4f}")
print(f"  AUC (Macro):        {test_auc['macro']:.4f}")

print("\n" + "="*80)
