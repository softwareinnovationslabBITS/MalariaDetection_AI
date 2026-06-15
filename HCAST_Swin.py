"""
H-CAST + Swin Transformer for 3-Level Malaria Classification
=============================================================
Combines Swin Transformer backbone (from SwinTransformer.py) with
H-CAST hierarchical heads (from HCAST_ViT_3Lvl.py).

Level 1: Infection Detection (Negative/Positive)
Level 2: Species Classification (Vivax/Falciparum)
Level 3: Stage Classification (7 stage classes)

Architecture:
- Backbone: Swin Transformer (window attention + shifted windows + patch merging)
- Classifier: H-CAST with additive TreePathConsistency penalty
"""

print("\n" + "="*80)
print("H-CAST + SWIN TRANSFORMER - 3-LEVEL CLASSIFICATION")
print("="*80)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import ops
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc, roc_auc_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
from datetime import datetime

# GPU Configuration
print("\nConfiguring GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"✓ GPU enabled: {gpus[0].name}")
else:
    print("⚠️  No GPU detected - running on CPU")
print(f"TensorFlow: {tf.__version__}")


# ============================================================================
# SWIN HYPERPARAMETERS
# ============================================================================
'''
PATCH_SIZE  = 128       # initial embedding dim per patch
NUM_PATCHES = 16        # number of patches to split embedding into
WINDOW_SIZE = 4         # local attention window size
NUM_HEADS   = 4         # base number of heads (doubles each stage)
DEPTHS      = [2, 2, 6] # SwinBlock depth per stage
MLP_RATIO   = 4
DROPOUT     = 0.1
'''
PATCH_SIZE  = 128       # initial embedding dim per patch
NUM_PATCHES = 16        # number of patches to split embedding into
WINDOW_SIZE = 8         # local attention window size
NUM_HEADS   = 8         # base number of heads (doubles each stage)
DEPTHS      = [2, 6, 8] # SwinBlock depth per stage
MLP_RATIO   = 4
DROPOUT     = 0.2

# ============================================================================
# SWIN COMPONENTS  (identical to SwinTransformer.py)
# ============================================================================

class PatchEmbedding(layers.Layer):
    """Split 1D embedding into N patches and project to embed_dim"""
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
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PatchMerging(layers.Layer):
    """Downsample sequence by 2x, double dim (same as Swin patch merging)"""
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
        cfg = super().get_config()
        cfg.update({"dim": self.dim})
        return cfg

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
    """Local window self-attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv      = layers.Dense(dim * 3, use_bias=False)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj      = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
        self.rel_bias  = self.add_weight(
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
        rel = ops.clip(rel + self.window_size - 1, 0, 2 * self.window_size - 2)
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
        cfg = super().get_config()
        cfg.update({
            "dim": self.dim, "window_size": self.window_size,
            "num_heads": self.num_heads, "attn_drop": self.attn_drop_rate,
            "proj_drop": self.proj_drop_rate
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SwinBlock(layers.Layer):
    """Swin Transformer block with optional window shift"""
    def __init__(self, dim, num_heads, shift, drop=0.0, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.shift = shift
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.norm1     = layers.LayerNormalization(epsilon=1e-6)
        self.attn      = WindowAttention(dim, WINDOW_SIZE, num_heads, attn_drop, drop)
        self.drop_path = layers.Dropout(drop)
        self.norm2     = layers.LayerNormalization(epsilon=1e-6)
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
        cfg = super().get_config()
        cfg.update({
            "dim": self.dim, "num_heads": self.num_heads, "shift": self.shift,
            "drop": self.drop_rate, "attn_drop": self.attn_drop_rate
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BasicLayer(layers.Layer):
    """One stage of Swin blocks + optional PatchMerging downsample"""
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
        cfg = super().get_config()
        cfg.update({
            "dim": self.dim, "depth": self.depth, "num_heads": self.num_heads,
            "downsample": "PatchMerging" if self.has_downsample else None,
            "drop": self.drop_rate, "attn_drop": self.attn_drop_rate
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        ds = config.pop("downsample")
        config["downsample"] = PatchMerging if ds == "PatchMerging" else None
        return cls(**config)


# ============================================================================
# H-CAST: TREE-PATH CONSISTENCY  (additive penalty, same as HCAST_ViT_3Lvl.py)
# ============================================================================

@tf.keras.utils.register_keras_serializable()
class TreePathConsistency3Level(layers.Layer):
    """
    Penalizes L2/L3 firing when L1 says Negative.
    Penalty added via add_loss() — predictions passed through unchanged.
    """
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        l1, l2, l3 = inputs
        l2_penalty = tf.nn.relu(l2 - l1)
        l3_penalty = tf.nn.relu(tf.reduce_max(l3, axis=-1, keepdims=True) - l1)
        self.add_loss(self.alpha * tf.reduce_mean(l2_penalty))
        self.add_loss(self.alpha * tf.reduce_mean(l3_penalty))
        return l1, l2, l3

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha})
        return cfg


# ============================================================================
# LOAD 3-LEVEL EMBEDDINGS
# ============================================================================
print("\n" + "="*80)
print("LOADING 3-LEVEL EMBEDDINGS")
print("="*80)

workspace_dir   = '/home/ghufran/MalariaML/Species_Classification/ajay'
embeddings_path = os.path.join(workspace_dir, 'embeddings_3level_dinov2_smote_proper801010.npz')
#embeddings_path = os.path.join(workspace_dir, 'embeddings_3level_efficientnet.npz')

if not os.path.exists(embeddings_path):
    print("❌ ERROR: EFFNET.npz not found!")
    exit(1)

data = np.load(embeddings_path, allow_pickle=True)

X_train = data['X_train'];  X_val = data['X_val'];  X_test = data['X_test']

train_l1 = data['train_l1'].astype(np.float32)
train_l2 = data['train_l2'].astype(np.float32)
train_l3 = data['train_l3'].astype(np.float32)
val_l1   = data['val_l1'].astype(np.float32)
val_l2   = data['val_l2'].astype(np.float32)
val_l3   = data['val_l3'].astype(np.float32)
test_l1  = data['test_l1'].astype(np.float32)
test_l2  = data['test_l2'].astype(np.float32)
test_l3  = data['test_l3'].astype(np.float32)

stage_names = data['stage_names']
EMBED_DIM   = int(data['embed_dim'])
NUM_STAGES  = int(data['num_stages'])

print(f"✓ Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"✓ Embedding dim: {EMBED_DIM}  Stages: {NUM_STAGES}")
print(f"✓ Stages: {list(stage_names)}")


# ============================================================================
# PREPARE LABELS & SAMPLE WEIGHTS  (list format — avoids Keras KeyError bug)
# ============================================================================
# L2 weight = 0 for negatives, L3 weight = 0 for no-stage samples
train_l2_w = (train_l1 == 1).astype(np.float32)
train_l3_w = (train_l3 >= 0).astype(np.float32)
train_l1_w = np.ones(len(train_l1), dtype=np.float32)

val_l2_w   = (val_l1 == 1).astype(np.float32)
val_l3_w   = (val_l3 >= 0).astype(np.float32)
val_l1_w   = np.ones(len(val_l1), dtype=np.float32)

# Replace -1 with 0 (dummy, masked by weight=0)
train_l2 = np.where(train_l2 < 0, 0, train_l2).astype(np.float32)
val_l2   = np.where(val_l2   < 0, 0, val_l2  ).astype(np.float32)
train_l3 = np.where(train_l3 < 0, 0, train_l3).astype(np.float32)
val_l3   = np.where(val_l3   < 0, 0, val_l3  ).astype(np.float32)

print(f"\nDataset sizes (all samples, masked weights):")
print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
print(f"\nLevel 1 (Infection):")
for name, l1 in [('Train', train_l1), ('Val', val_l1), ('Test', test_l1)]:
    print(f"  {name:5s}: Neg={int(np.sum(l1==0)):5d}, Pos={int(np.sum(l1==1)):5d}")


# ============================================================================
# BUILD  SWIN + H-CAST MODEL
# ============================================================================
print("\n" + "="*80)
print("BUILDING SWIN + H-CAST MODEL")
print("="*80)

# --- Swin backbone ---
inputs = layers.Input(shape=(EMBED_DIM,), name="embedding_input")
x = PatchEmbedding(NUM_PATCHES, PATCH_SIZE)(inputs)

for i, depth in enumerate(DEPTHS):
    x = BasicLayer(
        dim        = PATCH_SIZE * (2 ** i),
        depth      = depth,
        num_heads  = NUM_HEADS  * (2 ** i),
        downsample = PatchMerging if i < len(DEPTHS) - 1 else None,
        drop       = DROPOUT,
        attn_drop  = DROPOUT
    )(x)
# After 3 stages with 2 PatchMergings:
# Stage 0: (B, 16, 128) → PatchMerging → (B, 8, 256)
# Stage 1: (B,  8, 256) → PatchMerging → (B, 4, 512)
# Stage 2: (B,  4, 512) → no downsample
x = layers.LayerNormalization(epsilon=1e-6)(x)
x = layers.GlobalAveragePooling1D()(x)   # → (B, 512)

# --- Feature head (same capacity as SwinTransformer.py) ---
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation="gelu")(x)
x = layers.Dropout(0.2)(x)
features = layers.Dense(256, activation="gelu")(x)

# --- Hierarchical classification heads ---
l1_raw = layers.Dense(1,          activation='sigmoid',  name='l1_raw')(features)
l2_raw = layers.Dense(1,          activation='sigmoid',  name='l2_raw')(features)
l3_raw = layers.Dense(NUM_STAGES, activation='softmax',  dtype='float32', name='l3_raw')(features)

# Tree-path consistency (additive penalty)
l1_out, l2_out, l3_out = TreePathConsistency3Level(alpha=0.5)([l1_raw, l2_raw, l3_raw])
l1_out = layers.Identity(name="level1")(l1_out)
l2_out = layers.Identity(name="level2")(l2_out)
l3_out = layers.Identity(name="level3", dtype='float32')(l3_out)

model = keras.Model(inputs=inputs, outputs=[l1_out, l2_out, l3_out], name="HCAST_Swin")

print("✓ Model created!")
model.summary(line_length=100)


# ============================================================================
# COMPILE
# ============================================================================
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01),
    loss={
        "level1": "binary_crossentropy",
        "level2": "binary_crossentropy",
        "level3": "sparse_categorical_crossentropy",
    },
    loss_weights={"level1": 0.5, "level2": 10.0, "level3": 0.1},
    metrics={
        "level1": ["accuracy"],
        "level2": ["accuracy"],
        "level3": ["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')],
    }
)
print("✓ Compiled!")


# ============================================================================
# CALLBACKS
# ============================================================================
timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = os.path.join(workspace_dir, f'HCAST_SWIN/80-10-10_SMOTE/HCAST_Swin_Checkpoints_{timestamp}')
os.makedirs(checkpoint_dir, exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
        monitor='val_level2_accuracy', save_best_only=True, mode='max', verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_level2_accuracy', patience=30, restore_best_weights=True, mode='max', verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_level2_accuracy', factor=0.5, patience=10, mode='max', verbose=1
    ),
    keras.callbacks.CSVLogger(os.path.join(checkpoint_dir, 'training_log.csv'))
]


# ============================================================================
# TRAIN
# ============================================================================
print("\n" + "="*80)
print("TRAINING")
print("="*80)

start = time.time()
history = model.fit(
    X_train,
    [train_l1, train_l2, train_l3],
    sample_weight=[train_l1_w, train_l2_w, train_l3_w],
    validation_data=(
        X_val,
        [val_l1, val_l2, val_l3],
        [val_l1_w, val_l2_w, val_l3_w]  
    ),
    epochs=200,
    #batch_size=64,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - start
print(f"\n✓ Training done in {training_time/60:.1f} minutes")

# Best model already saved by ModelCheckpoint callback
# Note: EarlyStopping with restore_best_weights=True has already restored the best model
model_path = os.path.join(checkpoint_dir, 'best_model.keras')
print(f"✓ Best model saved at: {model_path}")

# Optionally save a copy to workspace root
final_model_path = os.path.join(workspace_dir, f'HCAST_SWIN/80-10-10_SMOTE/hcast_swin_3level_{timestamp}.keras')
model.save(final_model_path)
print(f"✓ Copy saved: {final_model_path}")


# ============================================================================
# EVALUATE  (each level on appropriate subset)
# ============================================================================
print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

p_l1, p_l2, p_l3 = model.predict(X_test, verbose=1)
pred_l1 = (p_l1 > 0.5).astype(int).flatten()
pred_l2 = (p_l2 > 0.5).astype(int).flatten()
pred_l3 = np.argmax(p_l3, axis=1)

test_l1_int = test_l1.astype(int)
test_l2_int = test_l2.astype(int)
test_l3_int = test_l3.astype(int)

l1_mask = np.ones(len(test_l1_int), dtype=bool)            # ALL samples
l2_mask = (test_l1_int == 1) & (test_l2_int >= 0)          # Positive, valid species
l3_mask = test_l3_int >= 0                                  # Valid stage label

l1_acc = accuracy_score(test_l1_int[l1_mask], pred_l1[l1_mask])
l2_acc = accuracy_score(test_l2_int[l2_mask], pred_l2[l2_mask])
l3_acc = accuracy_score(test_l3_int[l3_mask], pred_l3[l3_mask])
hier_acc = np.mean(
    (pred_l1[l3_mask] == test_l1_int[l3_mask]) &
    (pred_l2[l3_mask] == test_l2_int[l3_mask]) &
    (pred_l3[l3_mask] == test_l3_int[l3_mask])
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"  Level 1 (Infection):        {l1_acc*100:6.2f}%  [{l1_mask.sum():,} samples - ALL]")
print(f"  Level 2 (Species):          {l2_acc*100:6.2f}%  [{l2_mask.sum():,} samples - Positive only]")
print(f"  Level 3 (Stage):            {l3_acc*100:6.2f}%  [{l3_mask.sum():,} samples - Valid stages only]")
print(f"  Hierarchical (All correct): {hier_acc*100:6.2f}%  [{l3_mask.sum():,} samples]")

print("\n" + "="*80)
print("LEVEL 1: INFECTION DETECTION")
print("="*80)
print(classification_report(test_l1_int[l1_mask], pred_l1[l1_mask],
      target_names=["Negative", "Positive"], digits=4))

print("\n" + "="*80)
print("LEVEL 2: SPECIES CLASSIFICATION")
print("="*80)
print(classification_report(test_l2_int[l2_mask], pred_l2[l2_mask],
      target_names=["Vivax", "Falciparum"], digits=4))

print("\n" + "="*80)
print("LEVEL 3: STAGE CLASSIFICATION")
print("="*80)
present = np.unique(test_l3_int[l3_mask])
print(classification_report(
    test_l3_int[l3_mask], pred_l3[l3_mask],
    labels=list(present),
    target_names=[stage_names[i] for i in present],
    digits=4, zero_division=0
))


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute comprehensive metrics for binary classification

    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities (for AUC)

    Returns:
        Dictionary with all metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class is present
        tn = fp = fn = tp = 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)

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
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "G-Mean": gmean
    }


# ============================================================================
# VISUALIZATION AND REPORTING FUNCTIONS
# ============================================================================

def plot_confusion_matrices(results_dir, model, X_train, X_val, X_test,
                            train_l1, train_l2, train_l3,
                            val_l1, val_l2, val_l3,
                            test_l1, test_l2, test_l3,
                            stage_names):
    """Generate and save confusion matrices for all levels and splits"""

    # Get predictions for all splits
    print("\n  Generating confusion matrices...")
    train_p_l1, train_p_l2, train_p_l3 = model.predict(X_train, verbose=0)
    val_p_l1, val_p_l2, val_p_l3 = model.predict(X_val, verbose=0)
    test_p_l1, test_p_l2, test_p_l3 = model.predict(X_test, verbose=0)

    # Convert to class predictions
    train_pred_l1 = (train_p_l1 > 0.5).astype(int).flatten()
    val_pred_l1 = (val_p_l1 > 0.5).astype(int).flatten()
    test_pred_l1 = (test_p_l1 > 0.5).astype(int).flatten()

    train_pred_l2 = (train_p_l2 > 0.5).astype(int).flatten()
    val_pred_l2 = (val_p_l2 > 0.5).astype(int).flatten()
    test_pred_l2 = (test_p_l2 > 0.5).astype(int).flatten()

    train_pred_l3 = np.argmax(train_p_l3, axis=1)
    val_pred_l3 = np.argmax(val_p_l3, axis=1)
    test_pred_l3 = np.argmax(test_p_l3, axis=1)

    # Convert labels to int
    train_l1_int = train_l1.astype(int)
    val_l1_int = val_l1.astype(int)
    test_l1_int = test_l1.astype(int)

    train_l2_int = train_l2.astype(int)
    val_l2_int = val_l2.astype(int)
    test_l2_int = test_l2.astype(int)

    train_l3_int = train_l3.astype(int)
    val_l3_int = val_l3.astype(int)
    test_l3_int = test_l3.astype(int)

    # Define masks for each level
    train_l2_mask = (train_l1_int == 1) & (train_l2_int >= 0)
    val_l2_mask = (val_l1_int == 1) & (val_l2_int >= 0)
    test_l2_mask = (test_l1_int == 1) & (test_l2_int >= 0)

    train_l3_mask = train_l3_int >= 0
    val_l3_mask = val_l3_int >= 0
    test_l3_mask = test_l3_int >= 0

    # Create confusion matrices for each level
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Confusion Matrices - All Levels and Splits', fontsize=16, fontweight='bold')

    # Level 1 - Infection Detection
    for idx, (split_name, y_true, y_pred) in enumerate([
        ('TRAIN', train_l1_int, train_pred_l1),
        ('VAL', val_l1_int, val_pred_l1),
        ('TEST', test_l1_int, test_pred_l1)
    ]):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[0, idx].set_title(f'Level 1 (Infection) - {split_name}', fontweight='bold')
        axes[0, idx].set_ylabel('True Label')
        axes[0, idx].set_xlabel('Predicted Label')
        acc = accuracy_score(y_true, y_pred)
        axes[0, idx].text(0.5, -0.15, f'Accuracy: {acc*100:.2f}%',
                         ha='center', transform=axes[0, idx].transAxes)

    # Level 2 - Species Classification
    for idx, (split_name, y_true, y_pred, mask) in enumerate([
        ('TRAIN', train_l2_int, train_pred_l2, train_l2_mask),
        ('VAL', val_l2_int, val_pred_l2, val_l2_mask),
        ('TEST', test_l2_int, test_pred_l2, test_l2_mask)
    ]):
        cm = confusion_matrix(y_true[mask], y_pred[mask])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1, idx],
                   xticklabels=['Vivax', 'Falciparum'],
                   yticklabels=['Vivax', 'Falciparum'])
        axes[1, idx].set_title(f'Level 2 (Species) - {split_name}', fontweight='bold')
        axes[1, idx].set_ylabel('True Label')
        axes[1, idx].set_xlabel('Predicted Label')
        acc = accuracy_score(y_true[mask], y_pred[mask])
        axes[1, idx].text(0.5, -0.15, f'Accuracy: {acc*100:.2f}% (n={mask.sum()})',
                         ha='center', transform=axes[1, idx].transAxes)

    # Level 3 - Stage Classification
    for idx, (split_name, y_true, y_pred, mask) in enumerate([
        ('TRAIN', train_l3_int, train_pred_l3, train_l3_mask),
        ('VAL', val_l3_int, val_pred_l3, val_l3_mask),
        ('TEST', test_l3_int, test_pred_l3, test_l3_mask)
    ]):
        present_classes = np.unique(np.concatenate([y_true[mask], y_pred[mask]]))
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=present_classes)
        stage_labels = [stage_names[i] for i in present_classes]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[2, idx],
                   xticklabels=stage_labels, yticklabels=stage_labels)
        axes[2, idx].set_title(f'Level 3 (Stage) - {split_name}', fontweight='bold')
        axes[2, idx].set_ylabel('True Label')
        axes[2, idx].set_xlabel('Predicted Label')
        axes[2, idx].tick_params(axis='x', rotation=45)
        axes[2, idx].tick_params(axis='y', rotation=0)
        acc = accuracy_score(y_true[mask], y_pred[mask])
        axes[2, idx].text(0.5, -0.25, f'Accuracy: {acc*100:.2f}% (n={mask.sum()})',
                         ha='center', transform=axes[2, idx].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Confusion matrices saved")


def plot_training_history(results_dir, history):
    """Plot and save training history curves"""
    print("\n  Generating training history plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History - Loss and Accuracy', fontsize=16, fontweight='bold')

    # Loss plots
    for idx, level in enumerate(['level1', 'level2', 'level3']):
        ax = axes[0, idx]
        if f'{level}_loss' in history.history:
            ax.plot(history.history[f'{level}_loss'], label='Train', linewidth=2)
            ax.plot(history.history[f'val_{level}_loss'], label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{level.replace("level", "Level ")} Loss', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Accuracy plots
    level_names = ['Level 1 (Infection)', 'Level 2 (Species)', 'Level 3 (Stage)']
    for idx, (level, name) in enumerate(zip(['level1', 'level2', 'level3'], level_names)):
        ax = axes[1, idx]
        if f'{level}_accuracy' in history.history:
            ax.plot(history.history[f'{level}_accuracy'], label='Train', linewidth=2)
            ax.plot(history.history[f'val_{level}_accuracy'], label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{name} Accuracy', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Training history plots saved")


def plot_roc_curves(results_dir, model, X_test, test_l1, test_l2):
    """Plot and save ROC curves for Level 1 and Level 2"""
    print("\n  Generating ROC curves...")

    # Get predictions
    p_l1, p_l2, _ = model.predict(X_test, verbose=0)

    test_l1_int = test_l1.astype(int)
    test_l2_int = test_l2.astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ROC Curves', fontsize=16, fontweight='bold')

    # Level 1 ROC
    fpr_l1, tpr_l1, _ = roc_curve(test_l1_int, p_l1)
    roc_auc_l1 = auc(fpr_l1, tpr_l1)

    axes[0].plot(fpr_l1, tpr_l1, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc_l1:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Level 1 (Infection Detection)', fontweight='bold')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Level 2 ROC (only for positive samples)
    l2_mask = (test_l1_int == 1) & (test_l2_int >= 0)
    if l2_mask.sum() > 0:
        fpr_l2, tpr_l2, _ = roc_curve(test_l2_int[l2_mask], p_l2[l2_mask])
        roc_auc_l2 = auc(fpr_l2, tpr_l2)

        axes[1].plot(fpr_l2, tpr_l2, color='green', lw=2,
                    label=f'ROC curve (AUC = {roc_auc_l2:.4f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Level 2 (Species: Vivax vs Falciparum)', fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ ROC curves saved")

    return roc_auc_l1, roc_auc_l2 if l2_mask.sum() > 0 else None


def create_detailed_report(results_dir, model,
                          X_train, X_val, X_test,
                          train_l1, train_l2, train_l3,
                          val_l1, val_l2, val_l3,
                          test_l1, test_l2, test_l3,
                          stage_names, training_time, history, roc_auc_l1, roc_auc_l2):
    """Create comprehensive detailed report"""
    print("\n  Generating detailed report...")

    report_path = os.path.join(results_dir, 'detailed_report.txt')

    # Get predictions for all splits
    train_p_l1, train_p_l2, train_p_l3 = model.predict(X_train, verbose=0)
    val_p_l1, val_p_l2, val_p_l3 = model.predict(X_val, verbose=0)
    test_p_l1, test_p_l2, test_p_l3 = model.predict(X_test, verbose=0)

    # Convert to class predictions
    train_pred_l1 = (train_p_l1 > 0.5).astype(int).flatten()
    val_pred_l1 = (val_p_l1 > 0.5).astype(int).flatten()
    test_pred_l1 = (test_p_l1 > 0.5).astype(int).flatten()

    train_pred_l2 = (train_p_l2 > 0.5).astype(int).flatten()
    val_pred_l2 = (val_p_l2 > 0.5).astype(int).flatten()
    test_pred_l2 = (test_p_l2 > 0.5).astype(int).flatten()

    train_pred_l3 = np.argmax(train_p_l3, axis=1)
    val_pred_l3 = np.argmax(val_p_l3, axis=1)
    test_pred_l3 = np.argmax(test_p_l3, axis=1)

    # Convert labels to int
    train_l1_int = train_l1.astype(int)
    val_l1_int = val_l1.astype(int)
    test_l1_int = test_l1.astype(int)

    train_l2_int = train_l2.astype(int)
    val_l2_int = val_l2.astype(int)
    test_l2_int = test_l2.astype(int)

    train_l3_int = train_l3.astype(int)
    val_l3_int = val_l3.astype(int)
    test_l3_int = test_l3.astype(int)

    # Define masks for Level 2
    train_l2_mask = (train_l1_int == 1) & (train_l2_int >= 0)
    val_l2_mask = (val_l1_int == 1) & (val_l2_int >= 0)
    test_l2_mask = (test_l1_int == 1) & (test_l2_int >= 0)

    # ========================================================================
    # COMPUTE COMPREHENSIVE METRICS FOR ALL LEVELS
    # ========================================================================

    # Level 1 Metrics (all samples)
    print("  Computing Level 1 metrics...")
    l1_metrics_train = compute_metrics(train_l1_int, train_pred_l1, train_p_l1.flatten())
    l1_metrics_val = compute_metrics(val_l1_int, val_pred_l1, val_p_l1.flatten())
    l1_metrics_test = compute_metrics(test_l1_int, test_pred_l1, test_p_l1.flatten())

    # Calculate STD for Level 1
    l1_metrics_std = {}
    for key in l1_metrics_train.keys():
        values = [l1_metrics_train[key], l1_metrics_val[key], l1_metrics_test[key]]
        l1_metrics_std[key] = np.std(values)

    # Level 2 Metrics (positive samples only)
    print("  Computing Level 2 metrics...")
    l2_metrics_train = compute_metrics(
        train_l2_int[train_l2_mask],
        train_pred_l2[train_l2_mask],
        train_p_l2[train_l2_mask].flatten()
    ) if train_l2_mask.sum() > 0 else None

    l2_metrics_val = compute_metrics(
        val_l2_int[val_l2_mask],
        val_pred_l2[val_l2_mask],
        val_p_l2[val_l2_mask].flatten()
    ) if val_l2_mask.sum() > 0 else None

    l2_metrics_test = compute_metrics(
        test_l2_int[test_l2_mask],
        test_pred_l2[test_l2_mask],
        test_p_l2[test_l2_mask].flatten()
    ) if test_l2_mask.sum() > 0 else None

    # Calculate STD for Level 2
    l2_metrics_std = {}
    if all([l2_metrics_train, l2_metrics_val, l2_metrics_test]):
        for key in l2_metrics_train.keys():
            values = [l2_metrics_train[key], l2_metrics_val[key], l2_metrics_test[key]]
            l2_metrics_std[key] = np.std(values)

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("H-CAST + SWIN TRANSFORMER - DETAILED EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*80 + "\n")
        f.write(f"Backbone: Swin Transformer (window attention + patch merging)\n")
        f.write(f"Patch Size: {PATCH_SIZE}, Num Patches: {NUM_PATCHES}\n")
        f.write(f"Window Size: {WINDOW_SIZE}, Num Heads: {NUM_HEADS}\n")
        f.write(f"Depths: {DEPTHS}\n")
        f.write(f"Dropout: {DROPOUT}, MLP Ratio: {MLP_RATIO}\n")
        f.write(f"Classification: H-CAST with TreePathConsistency (alpha=0.5)\n\n")

        f.write("DATASET SPLIT\n")
        f.write("-"*80 + "\n")
        total = len(train_l1) + len(val_l1) + len(test_l1)
        f.write(f"Total samples: {total:,}\n")
        f.write(f"  Train: {len(train_l1):,} ({len(train_l1)/total*100:.1f}%)\n")
        f.write(f"  Val:   {len(val_l1):,} ({len(val_l1)/total*100:.1f}%)\n")
        f.write(f"  Test:  {len(test_l1):,} ({len(test_l1)/total*100:.1f}%)\n\n")

        f.write("TRAINING INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Training time: {training_time/60:.1f} minutes\n")
        f.write(f"Total epochs: {len(history.history['loss'])}\n")
        f.write(f"Optimizer: AdamW (lr=1e-4, weight_decay=0.01)\n")
        f.write(f"Loss weights: L1=1.0, L2=1.5, L3=2.0\n\n")

        # ====================================================================
        # COMPREHENSIVE METRICS - LEVEL 1
        # ====================================================================
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE METRICS - LEVEL 1: INFECTION DETECTION\n")
        f.write("="*80 + "\n\n")

        # Write metrics table with STD
        f.write("METRICS ACROSS ALL SPLITS (with Standard Deviation):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<20} {'Train':>12} {'Val':>12} {'Test':>12} {'Mean':>12} {'STD':>12}\n")
        f.write("-"*80 + "\n")

        for metric_name in l1_metrics_train.keys():
            train_val = l1_metrics_train[metric_name]
            val_val = l1_metrics_val[metric_name]
            test_val = l1_metrics_test[metric_name]
            mean_val = np.mean([train_val, val_val, test_val])
            std_val = l1_metrics_std[metric_name]

            f.write(f"{metric_name:<20} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f} "
                   f"{mean_val:>12.4f} {std_val:>12.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("LEVEL 1: INFECTION DETECTION (Negative vs Positive)\n")
        f.write("="*80 + "\n\n")

        for split_name, y_true, y_pred in [
            ('TRAIN', train_l1_int, train_pred_l1),
            ('VAL', val_l1_int, val_pred_l1),
            ('TEST', test_l1_int, test_pred_l1)
        ]:
            f.write(f"\n{split_name} SET:\n")
            f.write("-"*80 + "\n")
            f.write(classification_report(y_true, y_pred,
                                         target_names=['Negative', 'Positive'],
                                         digits=4))
            cm = confusion_matrix(y_true, y_pred)
            f.write(f"\nConfusion Matrix:\n{cm}\n")

        f.write(f"\nROC-AUC Score (Test): {roc_auc_l1:.4f}\n")

        # ====================================================================
        # COMPREHENSIVE METRICS - LEVEL 2
        # ====================================================================
        f.write("\n" + "="*80 + "\n")
        f.write("COMPREHENSIVE METRICS - LEVEL 2: SPECIES CLASSIFICATION\n")
        f.write("="*80 + "\n\n")

        if all([l2_metrics_train, l2_metrics_val, l2_metrics_test]):
            f.write("METRICS ACROSS ALL SPLITS (with Standard Deviation):\n")
            f.write("Note: Computed on Positive samples only\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Metric':<20} {'Train':>12} {'Val':>12} {'Test':>12} {'Mean':>12} {'STD':>12}\n")
            f.write("-"*80 + "\n")

            for metric_name in l2_metrics_train.keys():
                train_val = l2_metrics_train[metric_name]
                val_val = l2_metrics_val[metric_name]
                test_val = l2_metrics_test[metric_name]
                mean_val = np.mean([train_val, val_val, test_val])
                std_val = l2_metrics_std[metric_name]

                f.write(f"{metric_name:<20} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f} "
                       f"{mean_val:>12.4f} {std_val:>12.4f}\n")
        else:
            f.write("Insufficient samples for comprehensive metrics\n")

        f.write("\n" + "="*80 + "\n")
        f.write("LEVEL 2: SPECIES CLASSIFICATION (Vivax vs Falciparum)\n")
        f.write("="*80 + "\n\n")

        for split_name, y_true, y_pred, l1_true in [
            ('TRAIN', train_l2_int, train_pred_l2, train_l1_int),
            ('VAL', val_l2_int, val_pred_l2, val_l1_int),
            ('TEST', test_l2_int, test_pred_l2, test_l1_int)
        ]:
            mask = (l1_true == 1) & (y_true >= 0)
            f.write(f"\n{split_name} SET (Positive samples only, n={mask.sum()}):\n")
            f.write("-"*80 + "\n")
            if mask.sum() > 0:
                f.write(classification_report(y_true[mask], y_pred[mask],
                                             target_names=['Vivax', 'Falciparum'],
                                             digits=4))
                cm = confusion_matrix(y_true[mask], y_pred[mask])
                f.write(f"\nConfusion Matrix:\n{cm}\n")

        if roc_auc_l2 is not None:
            f.write(f"\nROC-AUC Score (Test): {roc_auc_l2:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("LEVEL 3: STAGE CLASSIFICATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Stage Classes: {list(stage_names)}\n\n")

        for split_name, y_true, y_pred in [
            ('TRAIN', train_l3_int, train_pred_l3),
            ('VAL', val_l3_int, val_pred_l3),
            ('TEST', test_l3_int, test_pred_l3)
        ]:
            mask = y_true >= 0
            present = np.unique(y_true[mask])
            f.write(f"\n{split_name} SET (Valid stages, n={mask.sum()}):\n")
            f.write("-"*80 + "\n")
            if mask.sum() > 0:
                f.write(classification_report(y_true[mask], y_pred[mask],
                                             labels=list(present),
                                             target_names=[stage_names[i] for i in present],
                                             digits=4, zero_division=0))
                cm = confusion_matrix(y_true[mask], y_pred[mask], labels=list(present))
                f.write(f"\nConfusion Matrix:\n{cm}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("HIERARCHICAL ACCURACY (All 3 levels correct)\n")
        f.write("="*80 + "\n\n")

        for split_name, pred_l1, pred_l2, pred_l3, true_l1, true_l2, true_l3 in [
            ('TRAIN', train_pred_l1, train_pred_l2, train_pred_l3,
             train_l1_int, train_l2_int, train_l3_int),
            ('VAL', val_pred_l1, val_pred_l2, val_pred_l3,
             val_l1_int, val_l2_int, val_l3_int),
            ('TEST', test_pred_l1, test_pred_l2, test_pred_l3,
             test_l1_int, test_l2_int, test_l3_int)
        ]:
            mask = true_l3 >= 0
            hier_acc = np.mean(
                (pred_l1[mask] == true_l1[mask]) &
                (pred_l2[mask] == true_l2[mask]) &
                (pred_l3[mask] == true_l3[mask])
            )
            f.write(f"{split_name}: {hier_acc*100:.2f}% (n={mask.sum()})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Test set summary
        l1_mask = np.ones(len(test_l1_int), dtype=bool)
        l2_mask = (test_l1_int == 1) & (test_l2_int >= 0)
        l3_mask = test_l3_int >= 0

        l1_acc = accuracy_score(test_l1_int[l1_mask], test_pred_l1[l1_mask])
        l2_acc = accuracy_score(test_l2_int[l2_mask], test_pred_l2[l2_mask])
        l3_acc = accuracy_score(test_l3_int[l3_mask], test_pred_l3[l3_mask])
        hier_acc = np.mean(
            (test_pred_l1[l3_mask] == test_l1_int[l3_mask]) &
            (test_pred_l2[l3_mask] == test_l2_int[l3_mask]) &
            (test_pred_l3[l3_mask] == test_l3_int[l3_mask])
        )

        f.write(f"Test Set Performance:\n")
        f.write(f"  Level 1 (Infection):        {l1_acc*100:6.2f}%\n")
        f.write(f"  Level 2 (Species):          {l2_acc*100:6.2f}%\n")
        f.write(f"  Level 3 (Stage):            {l3_acc*100:6.2f}%\n")
        f.write(f"  Hierarchical (All correct): {hier_acc*100:6.2f}%\n")
        f.write(f"\nTraining time: {training_time/60:.1f} minutes\n")
        f.write(f"ROC-AUC L1: {roc_auc_l1:.4f}\n")
        if roc_auc_l2 is not None:
            f.write(f"ROC-AUC L2: {roc_auc_l2:.4f}\n")

    print("  ✓ Detailed report saved")


# ============================================================================
# SAVE RESULTS
# ============================================================================
results_dir = os.path.join(workspace_dir, f'HCAST_SWIN/80-10-10_SMOTE/dinoHCAST_Swin_Results_801010_SMOTE_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

print("\n" + "="*80)
print("GENERATING COMPREHENSIVE RESULTS")
print("="*80)

# Generate all visualizations and reports
plot_confusion_matrices(results_dir, model, X_train, X_val, X_test,
                        train_l1, train_l2, train_l3,
                        val_l1, val_l2, val_l3,
                        test_l1, test_l2, test_l3,
                        stage_names)

plot_training_history(results_dir, history)

roc_auc_l1, roc_auc_l2 = plot_roc_curves(results_dir, model, X_test, test_l1, test_l2)

create_detailed_report(results_dir, model,
                      X_train, X_val, X_test,
                      train_l1, train_l2, train_l3,
                      val_l1, val_l2, val_l3,
                      test_l1, test_l2, test_l3,
                      stage_names, training_time, history, roc_auc_l1, roc_auc_l2)

# Save metrics JSON with comprehensive metrics
metrics = {
    'approach': 'H-CAST + Swin Transformer (3-level)',
    'backbone': 'Swin (window attention + patch merging)',
    'depths': DEPTHS, 'patch_size': PATCH_SIZE, 'num_patches': NUM_PATCHES,
    'window_size': WINDOW_SIZE, 'num_heads': NUM_HEADS,

    # High-level accuracies
    'level1_accuracy': float(l1_acc),
    'level2_accuracy': float(l2_acc),
    'level3_accuracy': float(l3_acc),
    'hierarchical_accuracy': float(hier_acc),

    # ROC-AUC scores
    'roc_auc_l1': float(roc_auc_l1),
    'roc_auc_l2': float(roc_auc_l2) if roc_auc_l2 is not None else None,

    # Comprehensive Level 1 metrics
    'level1_comprehensive': {
        'train': {k: float(v) for k, v in l1_metrics_train.items()},
        'val': {k: float(v) for k, v in l1_metrics_val.items()},
        'test': {k: float(v) for k, v in l1_metrics_test.items()},
        'std': {k: float(v) for k, v in l1_metrics_std.items()},
        'mean': {k: float(np.mean([l1_metrics_train[k], l1_metrics_val[k], l1_metrics_test[k]]))
                for k in l1_metrics_train.keys()}
    },

    # Comprehensive Level 2 metrics
    'level2_comprehensive': {
        'train': {k: float(v) for k, v in l2_metrics_train.items()} if l2_metrics_train else None,
        'val': {k: float(v) for k, v in l2_metrics_val.items()} if l2_metrics_val else None,
        'test': {k: float(v) for k, v in l2_metrics_test.items()} if l2_metrics_test else None,
        'std': {k: float(v) for k, v in l2_metrics_std.items()} if l2_metrics_std else None,
        'mean': {k: float(np.mean([l2_metrics_train[k], l2_metrics_val[k], l2_metrics_test[k]]))
                for k in l2_metrics_train.keys()} if l2_metrics_train else None
    },

    # Training info
    'training_time_minutes': float(training_time / 60),
    'total_epochs': len(history.history['loss']),
    'train_samples': int(len(train_l1)),
    'val_samples': int(len(val_l1)),
    'test_samples': int(len(test_l1)),
    'stage_names': list(stage_names),
    'timestamp': timestamp
}
with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n  ✓ Metrics JSON saved")

# Save comprehensive metrics to CSV for easy analysis
import csv
metrics_csv_path = os.path.join(results_dir, 'comprehensive_metrics.csv')
with open(metrics_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Level 1 metrics
    writer.writerow(['LEVEL 1: INFECTION DETECTION'])
    writer.writerow(['Metric', 'Train', 'Val', 'Test', 'Mean', 'STD'])
    for metric_name in l1_metrics_train.keys():
        train_val = l1_metrics_train[metric_name]
        val_val = l1_metrics_val[metric_name]
        test_val = l1_metrics_test[metric_name]
        mean_val = np.mean([train_val, val_val, test_val])
        std_val = l1_metrics_std[metric_name]
        writer.writerow([metric_name, f'{train_val:.4f}', f'{val_val:.4f}',
                        f'{test_val:.4f}', f'{mean_val:.4f}', f'{std_val:.4f}'])

    writer.writerow([])  # Empty row

    # Level 2 metrics
    writer.writerow(['LEVEL 2: SPECIES CLASSIFICATION (Positive samples only)'])
    if all([l2_metrics_train, l2_metrics_val, l2_metrics_test]):
        writer.writerow(['Metric', 'Train', 'Val', 'Test', 'Mean', 'STD'])
        for metric_name in l2_metrics_train.keys():
            train_val = l2_metrics_train[metric_name]
            val_val = l2_metrics_val[metric_name]
            test_val = l2_metrics_test[metric_name]
            mean_val = np.mean([train_val, val_val, test_val])
            std_val = l2_metrics_std[metric_name]
            writer.writerow([metric_name, f'{train_val:.4f}', f'{val_val:.4f}',
                            f'{test_val:.4f}', f'{mean_val:.4f}', f'{std_val:.4f}'])
    else:
        writer.writerow(['Insufficient samples'])

print("  ✓ Comprehensive metrics CSV saved")

# Save quick summary report with enhanced metrics
with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
    f.write("H-CAST + SWIN TRANSFORMER - QUICK SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write("TEST SET ACCURACIES:\n")
    f.write(f"  Level 1 (Infection):        {l1_acc*100:.2f}%  [{l1_mask.sum()} samples]\n")
    f.write(f"  Level 2 (Species):          {l2_acc*100:.2f}%  [{l2_mask.sum()} samples]\n")
    f.write(f"  Level 3 (Stage):            {l3_acc*100:.2f}%  [{l3_mask.sum()} samples]\n")
    f.write(f"  Hierarchical (All correct): {hier_acc*100:.2f}%\n")
    f.write(f"\n  ROC-AUC L1: {roc_auc_l1:.4f}\n")
    if roc_auc_l2 is not None:
        f.write(f"  ROC-AUC L2: {roc_auc_l2:.4f}\n")
    f.write(f"\n  Training time: {training_time/60:.1f} minutes\n")

    # Add comprehensive metrics summary
    f.write("\n" + "="*80 + "\n")
    f.write("COMPREHENSIVE METRICS (TEST SET)\n")
    f.write("="*80 + "\n\n")
    f.write("Level 1 (Infection Detection):\n")
    for metric_name, value in l1_metrics_test.items():
        f.write(f"  {metric_name:<20}: {value:.4f}\n")

    if l2_metrics_test:
        f.write("\nLevel 2 (Species Classification - Positive only):\n")
        for metric_name, value in l2_metrics_test.items():
            f.write(f"  {metric_name:<20}: {value:.4f}\n")

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print(f"Directory: {results_dir}")
print("\nGenerated files:")
print("  - confusion_matrices_all.png")
print("  - training_history.png")
print("  - roc_curves.png")
print("  - detailed_report.txt")
print("  - summary.txt")
print("  - metrics.json")
print("  - comprehensive_metrics.csv")
print(f"\nCheckpoint directory: {checkpoint_dir}")
print(f"  - best_model.keras")
print(f"  - training_log.csv")
