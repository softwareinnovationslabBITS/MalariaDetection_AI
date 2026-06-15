import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import ops
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
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

# Load from single SMOTE file (train is SMOTE-balanced, val/test are original)
data = np.load("/home/ghufran/MalariaML/Species_Classification/ajay/embeddings_3level_dinov2_smote_proper602020.npz")

# Extract embeddings and labels
X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

# For 3-class flat classification, create combined labels:
# 0 = Negative (l1 == 0)
# 1 = Vivax (l1 == 1 and l2 == 0)
# 2 = Falciparum (l1 == 1 and l2 == 1)
def create_flat_labels(l1, l2):
    """Convert hierarchical labels to flat 3-class labels"""
    labels = np.zeros(len(l1), dtype=np.int32)
    labels[l1 == 0] = 0  # Negative
    labels[(l1 == 1) & (l2 == 0)] = 1  # Vivax
    labels[(l1 == 1) & (l2 == 1)] = 2  # Falciparum
    return labels

y_train = create_flat_labels(data["train_l1"], data["train_l2"])
y_val = create_flat_labels(data["val_l1"], data["val_l2"])
y_test = create_flat_labels(data["test_l1"], data["test_l2"])

EMBED_DIM = X_train.shape[1]

print(f"✓ Loaded embeddings from single file")
print(f"  Train: {X_train.shape} (SMOTE-balanced)")
print(f"  Val:   {X_val.shape} (original)")
print(f"  Test:  {X_test.shape} (original)")

y_train_oh = keras.utils.to_categorical(y_train, 3)
y_val_oh   = keras.utils.to_categorical(y_val, 3)
y_test_oh  = keras.utils.to_categorical(y_test, 3)


PATCH_SIZE  = 128
NUM_PATCHES = 16
WINDOW_SIZE = 4
NUM_HEADS   = 4
DEPTHS      = [2, 2, 6]     # 3 stages (correct for 1D)
MLP_RATIO   = 4
DROPOUT     = 0.1


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim)

    def call(self, x):
        B = ops.shape(x)[0]
        patch_dim = x.shape[-1] // self.num_patches
        x = ops.reshape(x, (B, self.num_patches, patch_dim))
        return self.proj(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PatchMerging(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.reduction = layers.Dense(2 * dim, use_bias=False)

    def call(self, x):
        if ops.shape(x)[1] % 2 == 1:
            x = ops.pad(x, [[0, 0], [0, 1], [0, 0]])
        x0, x1 = x[:, 0::2, :], x[:, 1::2, :]
        x = ops.concatenate([x0, x1], axis=-1)
        return self.reduction(self.norm(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def window_partition(x, window_size):
    B, L, C = ops.shape(x)[0], ops.shape(x)[1], x.shape[-1]
    pad = (window_size - L % window_size) % window_size
    if pad > 0:
        x = ops.pad(x, [[0, 0], [0, pad], [0, 0]])
    Lp = L + pad
    x = ops.reshape(x, (B * (Lp // window_size), window_size, C))
    return x, Lp


def window_reverse(windows, window_size, Lp):
    B = ops.shape(windows)[0] // (Lp // window_size)
    C = windows.shape[-1]
    x = ops.reshape(windows, (B, Lp // window_size, window_size, C))
    return ops.reshape(x, (B, Lp, C))


class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=False)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

        self.rel_bias = self.add_weight(
            shape=(2 * window_size - 1, num_heads),
            initializer="truncated_normal",
            trainable=True
        )

    def call(self, x):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], x.shape[-1]

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ops.matmul(q * self.scale, ops.transpose(k, (0, 1, 3, 2)))

        coords = ops.arange(N)
        rel = coords[:, None] - coords[None, :]
        rel += self.window_size - 1
        rel = ops.clip(rel, 0, 2 * self.window_size - 2)

        bias = tf.gather(self.rel_bias, rel)
        bias = ops.transpose(bias, (2, 0, 1))
        attn = attn + ops.expand_dims(bias, 0)

        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = self.proj(ops.reshape(x, (B, N, C)))
        return self.proj_drop(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "attn_drop": self.attn_drop_rate,
            "proj_drop": self.proj_drop_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SwinBlock(layers.Layer):
    def __init__(self, dim, num_heads, shift, drop=0.0, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.shift = shift
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = WindowAttention(dim, WINDOW_SIZE, num_heads, attn_drop, drop)
        self.drop_path = layers.Dropout(drop)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(dim * MLP_RATIO, activation="gelu"),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])

    def call(self, x):
        shortcut = x
        x = self.norm1(x)

        if self.shift > 0:
            x = ops.roll(x, -self.shift, axis=1)

        xw, Lp = window_partition(x, WINDOW_SIZE)
        xw = self.attn(xw)
        x = window_reverse(xw, WINDOW_SIZE, Lp)[:, :ops.shape(shortcut)[1], :]

        if self.shift > 0:
            x = ops.roll(x, self.shift, axis=1)

        x = shortcut + self.drop_path(x)
        return x + self.mlp(self.norm2(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "shift": self.shift,
            "drop": self.drop_rate,
            "attn_drop": self.attn_drop_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BasicLayer(layers.Layer):
    def __init__(self, dim, depth, num_heads, downsample, drop=0.0, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.has_downsample = downsample is not None
        self.blocks = [
            SwinBlock(dim, num_heads, 0 if i % 2 == 0 else WINDOW_SIZE // 2, drop, attn_drop)
            for i in range(depth)
        ]
        self.downsample = downsample(dim) if downsample else None

    def call(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "downsample": "PatchMerging" if self.has_downsample else None,
            "drop": self.drop_rate,
            "attn_drop": self.attn_drop_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        downsample_name = config.pop("downsample")
        if downsample_name == "PatchMerging":
            config["downsample"] = PatchMerging
        else:
            config["downsample"] = None
        return cls(**config)


inputs = layers.Input(shape=(EMBED_DIM,))
x = PatchEmbedding(NUM_PATCHES, PATCH_SIZE)(inputs)

for i, d in enumerate(DEPTHS):
    x = BasicLayer(
        dim=PATCH_SIZE * (2 ** i),
        depth=d,
        num_heads=NUM_HEADS * (2 ** i),
        downsample=PatchMerging if i < len(DEPTHS) - 1 else None,
        drop=DROPOUT,
        attn_drop=DROPOUT
    )(x)

x = layers.LayerNormalization(epsilon=1e-6)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation="gelu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation="gelu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, x)
model.compile(
    optimizer=keras.optimizers.AdamW(1e-4, weight_decay=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"/home/ghufran/MalariaML/Species_Classification/ajay/SWIN/SWIN_RESULT_602020SMOTE{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"\n✓ Results will be saved to: {results_dir}")


history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=50,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)


y_pred = np.argmax(model.predict(X_test), axis=1)
y_pred_proba = model.predict(X_test)

class_names = ["Negative", "Vivax", "Falciparum"]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================================================
# SAVE TRAINING CURVES (Loss & Accuracy)
# ============================================================================
print("\n" + "="*60)
print("SAVING METRICS AND PLOTS")
print("="*60)

# Loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: training_curves.png")

# ============================================================================
# SAVE CONFUSION MATRIX
# ============================================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 14})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: confusion_matrix.png")

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 12})
plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: confusion_matrix_normalized.png")

# ============================================================================
# SAVE ROC-AUC CURVES
# ============================================================================
plt.figure(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
roc_auc_scores = {}

for i, (class_name, color) in enumerate(zip(class_names, colors)):
    # One-vs-Rest for each class
    y_true_binary = (y_test == i).astype(int)
    y_score = y_pred_proba[:, i]

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    roc_auc_scores[class_name] = float(roc_auc)

    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{class_name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_auc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: roc_auc_curves.png")

# ============================================================================
# SAVE METRICS JSON
# ============================================================================
test_accuracy = accuracy_score(y_test, y_pred)

# Per-class metrics
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

metrics = {
    'timestamp': timestamp,
    'model': 'SwinTransformer_1D',
    'hyperparameters': {
        'patch_size': PATCH_SIZE,
        'num_patches': NUM_PATCHES,
        'window_size': WINDOW_SIZE,
        'num_heads': NUM_HEADS,
        'depths': DEPTHS,
        'mlp_ratio': MLP_RATIO,
        'dropout': DROPOUT
    },
    'training': {
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    },
    'test_results': {
        'accuracy': float(test_accuracy),
        'confusion_matrix': cm.tolist(),
        'roc_auc_scores': roc_auc_scores,
        'classification_report': report_dict
    },
    'data_info': {
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test),
        'embedding_dim': EMBED_DIM
    }
}

metrics_path = os.path.join(results_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"✓ Saved: metrics.json")

# ============================================================================
# SAVE CLASSIFICATION REPORT
# ============================================================================
report_path = os.path.join(results_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("SWIN TRANSFORMER 1D - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write("-"*60 + "\n")
    f.write(classification_report(y_test, y_pred, target_names=class_names))
    f.write("\n" + "-"*60 + "\n")
    f.write("\nROC-AUC Scores:\n")
    for class_name, auc_score in roc_auc_scores.items():
        f.write(f"  {class_name}: {auc_score:.4f}\n")
    f.write("\n" + "="*60 + "\n")
print(f"✓ Saved: classification_report.txt")

# ============================================================================
# SAVE TRAINING HISTORY
# ============================================================================
history_path = os.path.join(results_dir, 'training_history.json')
history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=4)
print(f"✓ Saved: training_history.json")

# ============================================================================
# SAVE MODEL AND PREDICTIONS
# ============================================================================
model.save(os.path.join(results_dir, "swin1d_model.keras"))
print(f"✓ Saved: swin1d_model.keras")

np.savez(
    os.path.join(results_dir, "predictions.npz"),
    y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba
)
print(f"✓ Saved: predictions.npz")

# Also save to original locations for backward compatibility
model.save("/home/ghufran/MalariaML/Species_Classification/ajay/SWIN/swin1d_true.keras")
np.savez(
    "/home/ghufran/MalariaML/Species_Classification/ajay/SWIN/swin1d_preds.npz",
    y_test=y_test, y_pred=y_pred
)

print("\n" + "="*60)
print(f"✅ ALL RESULTS SAVED TO: {results_dir}")
print("="*60)
print(f"\nFiles saved:")
print(f"  - training_curves.png          (Loss & Accuracy curves)")
print(f"  - confusion_matrix.png         (Raw counts)")
print(f"  - confusion_matrix_normalized.png (Percentages)")
print(f"  - roc_auc_curves.png           (ROC curves for each class)")
print(f"  - metrics.json                 (All metrics in JSON)")
print(f"  - classification_report.txt    (Text report)")
print(f"  - training_history.json        (Full training history)")
print(f"  - swin1d_model.keras           (Saved model)")
print(f"  - predictions.npz              (Test predictions)")
print("="*60 + "\n")