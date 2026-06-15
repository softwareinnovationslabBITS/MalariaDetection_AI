print("\n" + "="*80)
print("BUILDING VISION TRANSFORMER (ViT) FROM SCRATCH")
print("Using SMOTE-preprocessed ResNet50 embeddings")
print("Classes: Negative (0), Vivax (1), Falciparum (2)")
print("="*80)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import time
import os
from datetime import datetime

print("\nConfiguring GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"GPU enabled: {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
        print(f"Device: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected - running on CPU")

print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

print("\nLoading SMOTE-balanced embeddings...")
smote_data = np.load('/home/ghufran/MalariaML/Species_Classification/ajay/embeddings_smote_balanced.npz')

train_emb_smote = smote_data['train_emb']
train_flat_smote = smote_data['train_flat']

print(f"Loaded {len(train_emb_smote):,} SMOTE-balanced training samples")
print(f"Embedding dimension: {train_emb_smote.shape[1]}")

# Load original test/val embeddings (NO SMOTE on test/val!)
original_data = np.load('/home/ghufran/MalariaML/Species_Classification/ajay/embeddings_original.npz')
val_embeddings = original_data['val_emb']
val_flat = original_data['val_flat']
test_embeddings = original_data['test_emb']
test_flat = original_data['test_flat']

print(f"Validation samples: {len(val_embeddings):,}")
print(f"Test samples: {len(test_embeddings):,}")

print(f"\nClass Distribution (SMOTE-balanced training):")
unique_train, counts_train = np.unique(train_flat_smote, return_counts=True)
class_names = {0: 'Negative', 1: 'Vivax', 2: 'Falciparum'}
for label, count in zip(unique_train, counts_train):
 percentage = (count / len(train_flat_smote)) * 100
 print(f"{class_names[label]:12s}: {count:,} samples ({percentage:.1f}%)")

# VISION TRANSFORMER ARCHITECTURE (LIGHTWEIGHT FOR CPU)

print("\n  Building Vision Transformer Architecture...")

# Hyperparameters (optimized for CPU training)
EMBEDDING_DIM = 2048  # ResNet50 embeddings
PATCH_DIM = 256 # Patch dimension after projection
NUM_PATCHES = 8 # Number of patches (split embedding into 8 parts)
NUM_HEADS = 4# Attention heads (reduced from 8 for speed)
TRANSFORMER_BLOCKS = 2  # Number of transformer blocks (reduced from 4 for speed)
MLP_DIM = 512# MLP hidden dimension
NUM_CLASSES = 3
DROPOUT_RATE = 0.3

print(f"\nModel Configuration:")
print(f"Input dimension: {EMBEDDING_DIM} (ResNet50 embeddings)")
print(f"Patch dimension: {PATCH_DIM}")
print(f"Number of patches: {NUM_PATCHES}")
print(f"Attention heads: {NUM_HEADS}")
print(f"Transformer blocks: {TRANSFORMER_BLOCKS}")
print(f"MLP dimension: {MLP_DIM}")
print(f"Output classes: {NUM_CLASSES}")
print(f"Dropout rate: {DROPOUT_RATE}")

@tf.keras.utils.register_keras_serializable()
class EmbeddingPatcher(layers.Layer):
 def __init__(self, num_patches, patch_dim, **kwargs):
  super().__init__(**kwargs)
  self.num_patches = num_patches
  self.patch_dim = patch_dim
  self.projection = layers.Dense(patch_dim)
  
 def call(self, embeddings):
  batch_size = tf.shape(embeddings)[0]
  embedding_dim = embeddings.shape[-1]
  
  patch_size = embedding_dim // self.num_patches
  patches = tf.reshape(embeddings, [batch_size, self.num_patches, patch_size])
  
  patches = self.projection(patches)
  return patches
 
 def get_config(self):
  config = super().get_config()
  config.update({
"num_patches": self.num_patches,
"patch_dim": self.patch_dim
  })
  return config


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
 def __init__(self, num_patches, patch_dim, **kwargs):
  super().__init__(**kwargs)
  self.num_patches = num_patches
  self.patch_dim = patch_dim
  self.position_embedding = layers.Embedding(
input_dim=num_patches,
output_dim=patch_dim
  )
  
 def call(self, patches):
  positions = tf.range(start=0, limit=self.num_patches, delta=1)
  encoded = patches + self.position_embedding(positions)
  return encoded
 
 def get_config(self):
  config = super().get_config()
  config.update({
"num_patches": self.num_patches,
"patch_dim": self.patch_dim
  })
  return config


def create_transformer_block(patch_dim, num_heads, mlp_dim, dropout_rate):
 def apply(encoded_patches):
  x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
  
  attention_output = layers.MultiHeadAttention(
num_heads=num_heads,
key_dim=patch_dim,
dropout=dropout_rate
  )(x1, x1)
  
  x2 = layers.Add()([attention_output, encoded_patches])
  
  x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
  
  x3 = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x3)
  x3 = layers.Dropout(dropout_rate)(x3)
  x3 = layers.Dense(patch_dim)(x3)
  x3 = layers.Dropout(dropout_rate)(x3)
  
  output = layers.Add()([x3, x2])
  
  return output
 
 return apply



print(f"\n Building ViT model...")

# Input layer
inputs = layers.Input(shape=(EMBEDDING_DIM,), name='embedding_input')

patches = EmbeddingPatcher(NUM_PATCHES, PATCH_DIM)(inputs)
encoded_patches = PatchEncoder(NUM_PATCHES, PATCH_DIM)(patches)

for i in range(TRANSFORMER_BLOCKS):
 encoded_patches = create_transformer_block(
  patch_dim=PATCH_DIM,
  num_heads=NUM_HEADS,
  mlp_dim=MLP_DIM,
  dropout_rate=DROPOUT_RATE
 )(encoded_patches)

representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

representation = layers.GlobalAveragePooling1D()(representation)

representation = layers.Dropout(0.4)(representation)

features = layers.Dense(256, activation=tf.nn.gelu)(representation)
features = layers.Dropout(0.3)(features)
features = layers.Dense(128, activation=tf.nn.gelu)(features)
features = layers.Dropout(0.2)(features)

outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='classification_output')(features)

vit_model = keras.Model(inputs=inputs, outputs=outputs, name='ViT_Malaria_Classifier')

print(f"\nVision Transformer model created!")
print(f"\nModel Summary:")
vit_model.summary(line_length=100)

total_params = vit_model.count_params()
print(f"\nTotal parameters: {total_params:,}")

print(f"\nCompiling model...")

vit_model.compile(
 optimizer=keras.optimizers.Adam(learning_rate=0.001),
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
)

print(f"Model compiled with Adam optimizer (lr=0.001)")


print(f"\n  Training Vision Transformer...")
print(f"Epochs: 50")
print(f"Batch size: 64")
print(f"Training samples: {len(train_emb_smote):,}")
print(f"Validation samples: {len(val_embeddings):,}")

# Callbacks
callbacks = [
 keras.callbacks.EarlyStopping(
  monitor='val_loss',
  patience=15,
  restore_best_weights=True,
  verbose=1
 ),
 keras.callbacks.ReduceLROnPlateau(
  monitor='val_loss',
  factor=0.5,
  patience=7,
  min_lr=1e-7,
  verbose=1
 )
]

start_time = time.time()

history_vit = vit_model.fit(
 train_emb_smote,
 train_flat_smote,
 validation_data=(val_embeddings, val_flat),
 epochs=50,
 batch_size=64,
 callbacks=callbacks,
 verbose=1
)

training_time = time.time() - start_time

print(f"\nTRAINING COMPLETE!")
print(f"Training time: {training_time/60:.1f} minutes")

# Get best metrics
best_epoch = np.argmax(history_vit.history['val_accuracy']) + 1
best_val_acc = max(history_vit.history['val_accuracy'])
best_val_loss = min(history_vit.history['val_loss'])

print(f"\n Best Model Performance:")
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Best validation loss: {best_val_loss:.4f}")

model_path = '/home/ghufran/MalariaML/Species_Classification/ajay/vit_malaria_3class_model.keras'
vit_model.save(model_path)
print(f"\nModel saved: {model_path}")


print(f"\n" + "="*80)
print(" EVALUATING ON TEST SET")
print("="*80)

# Predict on test set
print(f"\nGenerating predictions...")
test_pred_probs = vit_model.predict(test_embeddings, verbose=0)
test_pred = np.argmax(test_pred_probs, axis=1)

# Calculate metrics
test_accuracy = accuracy_score(test_flat, test_pred)
test_kappa = cohen_kappa_score(test_flat, test_pred)
test_mcc = matthews_corrcoef(test_flat, test_pred)

print(f"\nTest Set Performance:")
print(f"Overall Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Cohen's Kappa: {test_kappa:.4f}")
print(f"Matthews Correlation Coefficient: {test_mcc:.4f}")

print(f"\nClassification Report:")
print(classification_report(
 test_flat,
 test_pred,
 target_names=['Negative', 'Vivax', 'Falciparum'],
 digits=4
))

# Confusion matrix
cm = confusion_matrix(test_flat, test_pred)
print(f"\n Confusion Matrix:")
print(f"{'':>12} {'Pred Neg':>12} {'Pred Vivax':>12} {'Pred Falci':>12}")
print(f"{'True Neg':>12} {cm[0,0]:>12} {cm[0,1]:>12} {cm[0,2]:>12}")
print(f"{'True Vivax':>12} {cm[1,0]:>12} {cm[1,1]:>12} {cm[1,2]:>12}")
print(f"{'True Falci':>12} {cm[2,0]:>12} {cm[2,1]:>12} {cm[2,2]:>12}")


print(f"\n Creating visualizations...")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Training History
axes[0].plot(history_vit.history['loss'], label='Training Loss', linewidth=2, marker='o')
axes[0].plot(history_vit.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
axes[0].set_title('ViT Training History - Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
xticklabels=['Negative', 'Vivax', 'Falciparum'],
yticklabels=['Negative', 'Vivax', 'Falciparum'],
ax=axes[1], annot_kws={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('ViT Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ghufran/MalariaML/Species_Classification/ajay/vit_results_from_scratch.png', dpi=300, bbox_inches='tight')
print(f" Saved: vit_results_from_scratch.png")
plt.show()


print(f"\n Saving results...")

# Save predictions
np.savez('/home/ghufran/MalariaML/Species_Classification/ajay/vit_test_predictions.npz',
test_true=test_flat,
test_pred=test_pred,
test_pred_probs=test_pred_probs,
test_embeddings=test_embeddings)
print(f" Predictions saved: vit_test_predictions.npz")

# Save evaluation metrics
import pickle
vit_results = {
 'accuracy': test_accuracy,
 'kappa': test_kappa,
 'mcc': test_mcc,
 'confusion_matrix': cm,
 'classification_report': classification_report(
  test_flat, test_pred,
  target_names=['Negative', 'Vivax', 'Falciparum'],
  output_dict=True
 ),
 'training_time_minutes': training_time / 60,
 'best_epoch': best_epoch,
 'best_val_accuracy': best_val_acc,
 'model_path': model_path
}

with open('/home/ghufran/MalariaML/Species_Classification/ajay/vit_evaluation_results.pkl', 'wb') as f:
 pickle.dump(vit_results, f)
print(f" Metrics saved: vit_evaluation_results.pkl")

print("\n" + "="*80)
print(" VISION TRANSFORMER TRAINING & EVALUATION COMPLETE!")
print("="*80)
print(f"\n Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f" Model saved: {model_path}")
print("\n" + "="*80)