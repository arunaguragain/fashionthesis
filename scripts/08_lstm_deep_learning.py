# DEEP LEARNING - LSTM MODEL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    SpatialDropout1D, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import seaborn as sns

print("="*60)
print("DEEP LEARNING — LSTM MODEL")
print("Fashion Trend Prediction | Aruna Guragain")
print(f"TensorFlow version: {tf.__version__}")
print("="*60)

# STEP 1 - LOAD DATA


print("\n STEP 1: Loading Data")
print("-"*60)

# Load combined dataset
df = pd.read_csv(
    '../data/cleaned/combined_with_sentiment.csv'
)
df['text_clean'] = df['text_clean'].fillna('')

# Load Nepal primary
nepal = pd.read_csv(
    '../data/cleaned/nepal_with_sentiment.csv'
)
nepal['text_clean'] = nepal['text_clean'].fillna('')

print(f" Combined dataset : {len(df):,} rows")
print(f" Nepal primary    : {len(nepal):,} rows")
print(f"\nSentiment distribution (Combined):")
print(df['sentiment'].value_counts().to_string())


# STEP 2 - PREPARE LABELS

print("\n STEP 2: Preparing Labels")
print("-"*60)

# Encode sentiment labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(
    df['sentiment']
)

# Check encoding
print("Label encoding:")
for cls, idx in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    count = (df['label'] == idx).sum()
    print(f"  {idx} = {cls}: {count:,} samples")

num_classes = len(label_encoder.classes_)


# STEP 3 - TOKENIZATION

print("\n STEP 3: Tokenization")
print("-"*60)

# Hyperparameters
MAX_WORDS  = 10000   # vocabulary size
MAX_LEN    = 100     # max sequence length
EMBED_DIM  = 64      # embedding dimensions

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token='<OOV>'
)
tokenizer.fit_on_texts(df['text_clean'])

print(f"Vocabulary size : {len(tokenizer.word_index):,}")
print(f"Max sequence len: {MAX_LEN}")
print(f"Embedding dim   : {EMBED_DIM}")

# Convert text to sequences
X_sequences = tokenizer.texts_to_sequences(
    df['text_clean']
)
X_padded = pad_sequences(
    X_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)
y_labels = df['label'].values

print(f" Padded shape: {X_padded.shape}")


# STEP 4 - TRAIN/TEST SPLIT

print("\n STEP 4: Train/Validation/Test Split")
print("-"*60)

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_padded, y_labels,
    test_size=0.30,
    random_state=42,
    stratify=y_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"Training set  : {len(X_train):,} samples (70%)")
print(f"Validation set: {len(X_val):,} samples (15%)")
print(f"Test set      : {len(X_test):,} samples (15%)")

# One-hot encode for categorical crossentropy
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val,   num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)


# STEP 5 - BUILD LSTM MODEL

print("\n STEP 5: Building LSTM Model")
print("-"*60)

model = Sequential([
    # Embedding layer
    # Converts word indices to dense vectors
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBED_DIM,
        input_length=MAX_LEN,
        name='embedding'
    ),

    # Spatial Dropout on embeddings
    SpatialDropout1D(0.2, name='spatial_dropout'),

    # Bidirectional LSTM
    # Reads text forward AND backward
    # Better understanding of context
    Bidirectional(
        LSTM(
            64,                    # 64 LSTM units
            dropout=0.2,           # input dropout
            recurrent_dropout=0.2, # recurrent dropout
            return_sequences=False
        ),
        name='bilstm'
    ),

    # Dense hidden layer
    Dense(32, activation='relu', name='dense1'),
    Dropout(0.3, name='dropout1'),

    # Output layer
    # 3 neurons = Positive/Neutral/Negative
    Dense(
        num_classes,
        activation='softmax',
        name='output'
    )
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()


# STEP 6 - TRAIN MODEL


print("\n STEP 6: Training LSTM Model")
print("-"*60)
print("This may take 5-15 minutes...")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.0001,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n LSTM Training Complete!")


# STEP 7 - EVALUATE MODEL

print("\n STEP 7: Evaluating LSTM Model")
print("-"*60)

# Test set evaluation
test_loss, test_acc = model.evaluate(
    X_test, y_test_cat, verbose=0
)
print(f"LSTM Test Accuracy : {test_acc*100:.2f}%")
print(f"LSTM Test Loss     : {test_loss:.4f}")

# Predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)

# Classification report
print(f"\nLSTM Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_
))


# STEP 8 - COMPLETE MODEL COMPARISON

print("\n STEP 8: Complete Model Comparison")
print("-"*60)

phase1_nb  = 65.82
phase1_svm = 71.52
phase2_nb  = 88.58
phase2_svm = 92.51
lstm_acc   = test_acc * 100

print(f"\n{'Model':<30} {'Accuracy':>10} {'Dataset':>20}")
print("-"*62)
print(f"{'Naive Bayes (Phase 1)':<30} "
      f"{phase1_nb:>9.2f}% {'Nepal 790 posts':>20}")
print(f"{'SVM (Phase 1)':<30} "
      f"{phase1_svm:>9.2f}% {'Nepal 790 posts':>20}")
print(f"{'Naive Bayes (Phase 2)':<30} "
      f"{phase2_nb:>9.2f}% {'Combined 105k':>20}")
print(f"{'SVM (Phase 2)':<30} "
      f"{phase2_svm:>9.2f}% {'Combined 105k':>20}")
print(f"{'LSTM BiDirectional':<30} "
      f"{lstm_acc:>9.2f}% {'Combined 105k':>20}")

# Determine winner
all_accs = {
    'Naive Bayes P1': phase1_nb,
    'SVM P1':         phase1_svm,
    'Naive Bayes P2': phase2_nb,
    'SVM P2':         phase2_svm,
    'LSTM':           lstm_acc
}
winner = max(all_accs, key=all_accs.get)
print(f"\n OVERALL BEST MODEL: {winner}")
print(f"   Accuracy: {all_accs[winner]:.2f}%")


# STEP 9 - NEPAL SPECIFIC LSTM EVALUATION

print("\n STEP 9: Nepal-Specific LSTM Evaluation")
print("-"*60)

# Test LSTM on Nepal-only data
nepal['label'] = label_encoder.transform(
    nepal['sentiment']
)
nepal_seq = tokenizer.texts_to_sequences(
    nepal['text_clean']
)
nepal_padded = pad_sequences(
    nepal_seq, maxlen=MAX_LEN,
    padding='post', truncating='post'
)
nepal_cat = to_categorical(
    nepal['label'].values, num_classes
)

nepal_loss, nepal_acc = model.evaluate(
    nepal_padded, nepal_cat, verbose=0
)

nepal_pred_proba = model.predict(
    nepal_padded, verbose=0
)
nepal_pred = np.argmax(nepal_pred_proba, axis=1)

print(f"LSTM on Nepal data: {nepal_acc*100:.2f}%")
print(f"\nNepal Classification Report:")
print(classification_report(
    nepal['label'].values,
    nepal_pred,
    target_names=label_encoder.classes_
))

# STEP 10 - VISUALIZATIONS

print("\n STEP 10: Creating Visualizations")
print("-"*60)

os.makedirs('../outputs/phase2', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'Deep Learning (LSTM) Results\n'
    'Fashion Trend Prediction — Nepal (18-26)',
    fontsize=14, fontweight='bold'
)

# Chart 1: Training History (Accuracy)
axes[0,0].plot(
    history.history['accuracy'],
    color='#e91e8c', linewidth=2,
    marker='o', label='Train Accuracy'
)
axes[0,0].plot(
    history.history['val_accuracy'],
    color='#9c27b0', linewidth=2,
    marker='s', label='Val Accuracy'
)
axes[0,0].set_title(
    'LSTM Training History\n(Accuracy)',
    fontweight='bold'
)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

#  Chart 2: Training History (Loss) 
axes[0,1].plot(
    history.history['loss'],
    color='#e74c3c', linewidth=2,
    marker='o', label='Train Loss'
)
axes[0,1].plot(
    history.history['val_loss'],
    color='#e67e22', linewidth=2,
    marker='s', label='Val Loss'
)
axes[0,1].set_title(
    'LSTM Training History\n(Loss)',
    fontweight='bold'
)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# Chart 3: All Models Comparison 
model_names = [
    'NB\nPhase1', 'SVM\nPhase1',
    'NB\nPhase2', 'SVM\nPhase2',
    'LSTM\nPhase2'
]
accuracies = [
    phase1_nb, phase1_svm,
    phase2_nb, phase2_svm,
    lstm_acc
]
bar_colors = [
    '#95a5a6', '#95a5a6',
    '#3498db', '#3498db',
    '#e91e8c'
]
bars = axes[0,2].bar(
    model_names, accuracies,
    color=bar_colors, edgecolor='white',
    width=0.6
)
axes[0,2].set_title(
    'All Models Comparison\nPhase 1 → Phase 2 → LSTM',
    fontweight='bold'
)
axes[0,2].set_ylabel('Accuracy (%)')
axes[0,2].set_ylim(0, 105)
for bar, acc in zip(bars, accuracies):
    axes[0,2].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{acc:.1f}%',
        ha='center', fontsize=8,
        fontweight='bold'
    )

# Chart 4: LSTM Confusion Matrix 
cm = confusion_matrix(
    y_test, y_pred,
    labels=[0, 1, 2]
)
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='RdPu', ax=axes[1,0],
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    annot_kws={'size': 10}
)
axes[1,0].set_title(
    'LSTM Confusion Matrix\n(Combined Dataset)',
    fontweight='bold'
)
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('Actual')

#  Chart 5: Nepal LSTM Confusion Matrix 
cm_nepal = confusion_matrix(
    nepal['label'].values, nepal_pred,
    labels=[0, 1, 2]
)
sns.heatmap(
    cm_nepal, annot=True, fmt='d',
    cmap='Blues', ax=axes[1,1],
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    annot_kws={'size': 10}
)
axes[1,1].set_title(
    'LSTM Confusion Matrix\n(Nepal Primary Data)',
    fontweight='bold'
)
axes[1,1].set_xlabel('Predicted')
axes[1,1].set_ylabel('Actual')

#  Chart 6: Final Comparison Bar 
final_models = [
    'NB\n(P1)', 'SVM\n(P1)',
    'NB\n(P2)', 'SVM\n(P2)',
    'LSTM\n(P2)'
]
final_accs = [
    phase1_nb, phase1_svm,
    phase2_nb, phase2_svm,
    lstm_acc
]
colors_final = [
    '#bdc3c7','#95a5a6',
    '#5dade2','#2980b9',
    '#e91e8c'
]
bars6 = axes[1,2].barh(
    final_models, final_accs,
    color=colors_final,
    edgecolor='white', height=0.5
)
axes[1,2].set_title(
    'Complete Model Progression\n(All Phases)',
    fontweight='bold'
)
axes[1,2].set_xlabel('Accuracy (%)')
axes[1,2].set_xlim(0, 105)
for bar, acc in zip(bars6, final_accs):
    axes[1,2].text(
        acc + 0.5,
        bar.get_y() + bar.get_height()/2,
        f'{acc:.1f}%',
        va='center', fontsize=9,
        fontweight='bold'
    )

plt.tight_layout(pad=3.0)
plt.savefig(
    '../outputs/phase2/lstm_results.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print(" Chart saved: lstm_results.png")


# STEP 11 - SAVE MODEL

print("\n STEP 11: Saving LSTM Model")
print("-"*60)

os.makedirs('../models', exist_ok=True)
model.save('../models/lstm_fashion_model.h5')
print(" Model saved: models/lstm_fashion_model.h5")

# Save final comparison
final_df = pd.DataFrame({
    'Model': [
        'Naive Bayes', 'SVM',
        'Naive Bayes', 'SVM', 'LSTM'
    ],
    'Phase': [
        'Phase1', 'Phase1',
        'Phase2', 'Phase2', 'Phase2'
    ],
    'Dataset': [
        'Nepal_790', 'Nepal_790',
        'Combined_105k', 'Combined_105k',
        'Combined_105k'
    ],
    'Test_Accuracy': [
        phase1_nb, phase1_svm,
        phase2_nb, phase2_svm,
        lstm_acc
    ]
})
final_df.to_csv(
    '../data/cleaned/complete_model_comparison.csv',
    index=False
)
print(" complete_model_comparison.csv saved")

# FINAL SUMMARY
print(f"\n{'='*60}")
print(f"DEEP LEARNING COMPLETE — FINAL SUMMARY")
print(f"{'='*60}")

print(f"\n COMPLETE MODEL COMPARISON:")
print(f"{'Model':<25} {'Phase':<12} {'Accuracy':>10}")
print(f"  {'-'*50}")
for _, row in final_df.iterrows():
    marker = "" if row['Test_Accuracy'] == max(
        final_df['Test_Accuracy']
    ) else ""
    print(f"  {row['Model']:<23} "
          f"{row['Phase']:<12} "
          f"{row['Test_Accuracy']:>8.2f}%{marker}")

print(f"\n BEST MODEL: {winner} ({all_accs[winner]:.2f}%)")

print(f"\n PROGRESSION:")
print(f"  Phase 1 best (SVM)   : {phase1_svm:.2f}%")
print(f"  Phase 2 best (SVM)   : {phase2_svm:.2f}%")
print(f"  Deep Learning (LSTM) : {lstm_acc:.2f}%")

print(f"\n🇳🇵 LSTM ON NEPAL DATA: {nepal_acc*100:.2f}%")

print(f"\n ALL MODELS COMPLETE!")
print(f"{'='*60}")