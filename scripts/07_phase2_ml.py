# PHASE 2 - MACHINE LEARNING MODELS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score
)
from sklearn.utils import resample
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PHASE 2 - MACHINE LEARNING MODELS")
print("Naive Bayes vs SVM — Phase 1 vs Phase 2")
print("="*60)

# STEP 1 - LOAD DATASETS
print("\n STEP 1: Loading Datasets")
print("-"*60)

# Combined dataset (for training)
combined = pd.read_csv(
    '../data/cleaned/combined_with_sentiment.csv'
)
combined['text_clean'] = combined['text_clean'].fillna('')

# Nepal primary only (for Nepal-specific results)
nepal = pd.read_csv(
    '../data/cleaned/nepal_with_sentiment.csv'
)
nepal['text_clean'] = nepal['text_clean'].fillna('')

print(f" Combined dataset  : {len(combined):,} rows")
print(f" Nepal primary     : {len(nepal):,} rows")

print(f"\nCombined Sentiment Distribution:")
print(combined['sentiment'].value_counts().to_string())
print(f"\nNepal Sentiment Distribution:")
print(nepal['sentiment'].value_counts().to_string())

# STEP 2 - PREPARE FEATURES

print("\n STEP 2: Preparing Features")
print("-"*60)

#  COMBINED DATASET 
X_combined = combined['text_clean']
y_combined  = combined['sentiment']

#  NEPAL ONLY 
X_nepal = nepal['text_clean']
y_nepal  = nepal['sentiment']

print(f" Combined features ready: {len(X_combined):,}")
print(f" Nepal features ready   : {len(X_nepal):,}")

# STEP 3 - TRAIN/TEST SPLIT

print("\n STEP 3: Train/Test Split (80/20)")
print("-"*60)

# Combined split
X_c_train, X_c_test, y_c_train, y_c_test = (
    train_test_split(
        X_combined, y_combined,
        test_size=0.2, random_state=42,
        stratify=y_combined
    )
)

# Nepal split
X_n_train, X_n_test, y_n_train, y_n_test = (
    train_test_split(
        X_nepal, y_nepal,
        test_size=0.2, random_state=42,
        stratify=y_nepal
    )
)

print(f"Combined — Train: {len(X_c_train):,} | "
      f"Test: {len(X_c_test):,}")
print(f"Nepal    — Train: {len(X_n_train):,} | "
      f"Test: {len(X_n_test):,}")

# STEP 4 - TF-IDF VECTORIZATION

print("\n STEP 4: TF-IDF Vectorization")
print("-"*60)

# Combined TF-IDF
tfidf_combined = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english',
    sublinear_tf=True
)
Xc_train_v = tfidf_combined.fit_transform(X_c_train)
Xc_test_v  = tfidf_combined.transform(X_c_test)

# Nepal TF-IDF
tfidf_nepal = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english',
    sublinear_tf=True
)
Xn_train_v = tfidf_nepal.fit_transform(X_n_train)
Xn_test_v  = tfidf_nepal.transform(X_n_test)

print(f"Combined vocab : {len(tfidf_combined.vocabulary_):,}")
print(f"Nepal vocab    : {len(tfidf_nepal.vocabulary_):,}")


# STEP 5 - TRAIN MODELS ON COMBINED DATA
print("\n STEP 5: Training Models — Combined Dataset")
print("-"*60)

#  Naive Bayes (Combined) 
print("\n  Training Naive Bayes (Combined)...")
nb_combined = MultinomialNB(alpha=0.1)
nb_combined.fit(Xc_train_v, y_c_train)
nb_c_pred = nb_combined.predict(Xc_test_v)
nb_c_acc  = accuracy_score(y_c_test, nb_c_pred)
nb_c_cv   = cross_val_score(
    nb_combined, Xc_train_v, y_c_train,
    cv=5, scoring='accuracy'
)
print(f"   NB Combined Accuracy : {nb_c_acc*100:.2f}%")
print(f"   NB Combined CV       : "
      f"{nb_c_cv.mean()*100:.2f}% "
      f"(±{nb_c_cv.std()*100:.2f}%)")

#  SVM (Combined) 
print("\n  Training SVM (Combined)...")
svm_combined = LinearSVC(
    C=1.0, max_iter=2000, random_state=42
)
svm_combined.fit(Xc_train_v, y_c_train)
svm_c_pred = svm_combined.predict(Xc_test_v)
svm_c_acc  = accuracy_score(y_c_test, svm_c_pred)
svm_c_cv   = cross_val_score(
    svm_combined, Xc_train_v, y_c_train,
    cv=5, scoring='accuracy'
)
print(f"   SVM Combined Accuracy: {svm_c_acc*100:.2f}%")
print(f"   SVM Combined CV      : "
      f"{svm_c_cv.mean()*100:.2f}% "
      f"(±{svm_c_cv.std()*100:.2f}%)")

# STEP 6 - TRAIN MODELS ON NEPAL DATA
print("\n STEP 6: Training Models — Nepal Dataset")
print("-"*60)

#  Naive Bayes (Nepal) 
print("\n  Training Naive Bayes (Nepal)...")
nb_nepal = MultinomialNB(alpha=0.1)
nb_nepal.fit(Xn_train_v, y_n_train)
nb_n_pred = nb_nepal.predict(Xn_test_v)
nb_n_acc  = accuracy_score(y_n_test, nb_n_pred)
nb_n_cv   = cross_val_score(
    nb_nepal, Xn_train_v, y_n_train,
    cv=5, scoring='accuracy'
)
print(f"   NB Nepal Accuracy    : {nb_n_acc*100:.2f}%")
print(f"   NB Nepal CV          : "
      f"{nb_n_cv.mean()*100:.2f}% "
      f"(±{nb_n_cv.std()*100:.2f}%)")

#  SVM (Nepal) 
print("\n  Training SVM (Nepal)...")
svm_nepal = LinearSVC(
    C=1.0, max_iter=2000, random_state=42
)
svm_nepal.fit(Xn_train_v, y_n_train)
svm_n_pred = svm_nepal.predict(Xn_test_v)
svm_n_acc  = accuracy_score(y_n_test, svm_n_pred)
svm_n_cv   = cross_val_score(
    svm_nepal, Xn_train_v, y_n_train,
    cv=5, scoring='accuracy'
)
print(f"   SVM Nepal Accuracy   : {svm_n_acc*100:.2f}%")
print(f"  SVM Nepal CV         : "
      f"{svm_n_cv.mean()*100:.2f}% "
      f"(±{svm_n_cv.std()*100:.2f}%)")


# STEP 7 - PHASE 1 vs PHASE 2 COMPARISON
print("\n STEP 7: Phase 1 vs Phase 2 Comparison")
print("-"*60)

# Phase 1 results (from previous run)
phase1_results = {
    'NB_accuracy' : 65.82,
    'NB_cv'       : 68.05,
    'SVM_accuracy': 71.52,
    'SVM_cv'      : 69.16,
}

print(f"\n{'Model':<20} {'Phase1':>10} "
      f"{'Phase2(Nepal)':>15} {'Phase2(Combined)':>18}")
print("-"*65)
print(f"{'Naive Bayes Acc':<20} "
      f"{phase1_results['NB_accuracy']:>9.2f}% "
      f"{nb_n_acc*100:>14.2f}% "
      f"{nb_c_acc*100:>17.2f}%")
print(f"{'NB CV Score':<20} "
      f"{phase1_results['NB_cv']:>9.2f}% "
      f"{nb_n_cv.mean()*100:>14.2f}% "
      f"{nb_c_cv.mean()*100:>17.2f}%")
print(f"{'SVM Acc':<20} "
      f"{phase1_results['SVM_accuracy']:>9.2f}% "
      f"{svm_n_acc*100:>14.2f}% "
      f"{svm_c_acc*100:>17.2f}%")
print(f"{'SVM CV Score':<20} "
      f"{phase1_results['SVM_cv']:>9.2f}% "
      f"{svm_n_cv.mean()*100:>14.2f}% "
      f"{svm_c_cv.mean()*100:>17.2f}%")

# Best model selection
best_acc = max(
    nb_c_acc, svm_c_acc,
    nb_n_acc, svm_n_acc
)
if svm_c_acc == best_acc:
    best_name = "SVM (Combined Dataset)"
    best_pred = svm_c_pred
    best_test = y_c_test
elif svm_n_acc == best_acc:
    best_name = "SVM (Nepal Dataset)"
    best_pred = svm_n_pred
    best_test = y_n_test
else:
    best_name = "Naive Bayes (Combined)"
    best_pred = nb_c_pred
    best_test = y_c_test

print(f"\n BEST MODEL: {best_name}")
print(f"   Accuracy: {best_acc*100:.2f}%")

# STEP 8 - CLASSIFICATION REPORTS

print("\n STEP 8: Classification Reports")
print("-"*60)

print(f"\n  SVM — Combined Dataset:")
print(classification_report(
    y_c_test, svm_c_pred,
    target_names=['Negative','Neutral','Positive']
))

print(f"\n  SVM — Nepal Dataset:")
print(classification_report(
    y_n_test, svm_n_pred,
    target_names=['Negative','Neutral','Positive']
))


# STEP 9 - FASHION CATEGORY MODEL

print("\n STEP 9: Fashion Category Model")
print("-"*60)

# Train on combined
y_cat_combined = combined['fashion_category']
X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
    X_combined, y_cat_combined,
    test_size=0.2, random_state=42,
    stratify=y_cat_combined
)

tfidf_category = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english',
    sublinear_tf=True
)
X_cat_train_v = tfidf_category.fit_transform(X_cat_train)
X_cat_test_v  = tfidf_category.transform(X_cat_test)

cat_model = LinearSVC(
    C=1.0, max_iter=2000, random_state=42
)
cat_model.fit(X_cat_train_v, y_cat_train)
cat_pred = cat_model.predict(X_cat_test_v)
cat_acc  = accuracy_score(y_cat_test, cat_pred)

cat_cv = cross_val_score(
    cat_model, X_cat_train_v, y_cat_train,
    cv=5, scoring='accuracy'
)

print(f"Category Model Accuracy : {cat_acc*100:.2f}%")
print(f"Category Model CV       : "
      f"{cat_cv.mean()*100:.2f}% "
      f"(±{cat_cv.std()*100:.2f}%)")
print(f"\nPhase 1 Category Acc    : 38.61%")
print(f"Phase 2 Category Acc    : {cat_acc*100:.2f}%")
print(f"Improvement             : "
      f"+{cat_acc*100 - 38.61:.2f}%")
print(f"Vocabulary size: {len(tfidf_category.vocabulary_)}")
print(f"Category classes: {cat_model.classes_}")
os.makedirs('../models', exist_ok=True)
joblib.dump(tfidf_category, '../models/tfidf_category.pkl')
joblib.dump(cat_model, '../models/svm_category_model.pkl')
print("✅ Category model saved to ../models/")
# STEP 10 - VISUALIZATIONS

print("\n STEP 10: Creating Visualizations")
print("-"*60)

os.makedirs('../outputs/phase2', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'Phase 2 Machine Learning Results\n'
    'Fashion Trend Prediction — Nepal (18-26)',
    fontsize=14, fontweight='bold'
)

#  Chart 1: Model Comparison 
models = [
    'NB\nPhase1', 'SVM\nPhase1',
    'NB\nP2-Nepal', 'SVM\nP2-Nepal',
    'NB\nP2-Combined', 'SVM\nP2-Combined'
]
accs = [
    phase1_results['NB_accuracy'],
    phase1_results['SVM_accuracy'],
    nb_n_acc*100, svm_n_acc*100,
    nb_c_acc*100, svm_c_acc*100
]
bar_colors = [
    '#95a5a6','#95a5a6',
    '#3498db','#3498db',
    '#e91e8c','#e91e8c'
]
bars = axes[0].bar(
    models, accs,
    color=bar_colors,
    edgecolor='white', width=0.6
)
axes[0].set_title(
    'Model Accuracy Comparison\nPhase 1 vs Phase 2',
    fontweight='bold', pad=12
)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 105)
for bar, acc in zip(bars, accs):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{acc:.1f}%',
        ha='center', fontsize=8,
        fontweight='bold'
    )

# Add legend
from matplotlib.patches import Patch
legend = [
    Patch(color='#95a5a6', label='Phase 1'),
    Patch(color='#3498db', label='Phase 2 Nepal'),
    Patch(color='#e91e8c', label='Phase 2 Combined'),
]
axes[0].legend(handles=legend, fontsize=8)

#  Chart 2: Confusion Matrix (Best SVM) 
if svm_c_acc >= svm_n_acc:
    cm = confusion_matrix(
        y_c_test, svm_c_pred,
        labels=['Positive','Neutral','Negative']
    )
    cm_title = 'SVM Confusion Matrix\n(Combined Dataset)'
else:
    cm = confusion_matrix(
        y_n_test, svm_n_pred,
        labels=['Positive','Neutral','Negative']
    )
    cm_title = 'SVM Confusion Matrix\n(Nepal Dataset)'

sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='Blues', ax=axes[1],
    xticklabels=['Pos','Neu','Neg'],
    yticklabels=['Pos','Neu','Neg'],
    annot_kws={'size': 12}
)
axes[1].set_title(cm_title, fontweight='bold', pad=12)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

#  Chart 3: Improvement Chart 
improvements = {
    'NB\nAccuracy' : nb_c_acc*100 - phase1_results['NB_accuracy'],
    'SVM\nAccuracy': svm_c_acc*100 - phase1_results['SVM_accuracy'],
    'Category\nModel': cat_acc*100 - 38.61,
}
imp_colors = [
    '#2ecc71' if v >= 0 else '#e74c3c'
    for v in improvements.values()
]
bars3 = axes[2].bar(
    improvements.keys(),
    improvements.values(),
    color=imp_colors,
    edgecolor='white', width=0.5
)
axes[2].axhline(y=0, color='black',
                linewidth=1, linestyle='-')
axes[2].set_title(
    'Accuracy Improvement\nPhase 1 → Phase 2',
    fontweight='bold', pad=12
)
axes[2].set_ylabel('Improvement (%)')
for bar, val in zip(bars3, improvements.values()):
    axes[2].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.1 if val >= 0
        else bar.get_height() - 0.5,
        f'{val:+.2f}%',
        ha='center', fontsize=10,
        fontweight='bold'
    )

plt.tight_layout(pad=3.0)
plt.savefig(
    '../outputs/phase2/phase2_ml_results.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print("Chart saved: phase2_ml_results.png")


# STEP 11 - SAVE RESULTS

print("\n STEP 11: Saving Results")
print("-"*60)

# Save comparison table
results_df = pd.DataFrame({
    'Model': [
        'Naive Bayes', 'SVM',
        'Naive Bayes', 'SVM',
        'Naive Bayes', 'SVM'
    ],
    'Phase': [
        'Phase1', 'Phase1',
        'Phase2_Nepal', 'Phase2_Nepal',
        'Phase2_Combined', 'Phase2_Combined'
    ],
    'Test_Accuracy': [
        phase1_results['NB_accuracy'],
        phase1_results['SVM_accuracy'],
        nb_n_acc*100, svm_n_acc*100,
        nb_c_acc*100, svm_c_acc*100
    ],
    'CV_Accuracy': [
        phase1_results['NB_cv'],
        phase1_results['SVM_cv'],
        nb_n_cv.mean()*100,
        svm_n_cv.mean()*100,
        nb_c_cv.mean()*100,
        svm_c_cv.mean()*100
    ]
})

results_df.to_csv(
    '../data/cleaned/phase2_ml_comparison.csv',
    index=False
)
print(" phase2_ml_comparison.csv saved")


# FINAL SUMMARY
print(f"\n{'='*60}")
print(f"PHASE 2 ML COMPLETE — FINAL SUMMARY")
print(f"{'='*60}")

print(f"\n MODEL PERFORMANCE COMPARISON:")
print(f"{'':5}{'Model':<20} {'P1':>8} "
      f"{'P2 Nepal':>10} {'P2 Combined':>13}")
print(f"  {'-'*55}")
print(f"{'':5}{'Naive Bayes':<20} "
      f"{phase1_results['NB_accuracy']:>7.2f}% "
      f"{nb_n_acc*100:>9.2f}% "
      f"{nb_c_acc*100:>12.2f}%")
print(f"{'':5}{'SVM':<20} "
      f"{phase1_results['SVM_accuracy']:>7.2f}% "
      f"{svm_n_acc*100:>9.2f}% "
      f"{svm_c_acc*100:>12.2f}%")
print(f"{'':5}{'Category Model':<20} "
      f"{'38.61':>7}% "
      f"{'N/A':>9} "
      f"{cat_acc*100:>12.2f}%")

print(f"\n BEST MODEL: {best_name}")
print(f"   Accuracy: {best_acc*100:.2f}%")

print(f"\n KEY IMPROVEMENTS:")
nb_imp  = nb_c_acc*100 - phase1_results['NB_accuracy']
svm_imp = svm_c_acc*100 - phase1_results['SVM_accuracy']
cat_imp = cat_acc*100 - 38.61
print(f"   NB improvement   : {nb_imp:+.2f}%")
print(f"   SVM improvement  : {svm_imp:+.2f}%")
print(f"   Category improve : {cat_imp:+.2f}%")

print(f"\n🇳🇵 NEPAL FINDINGS UNCHANGED:")
print(f"   Top Trend: Traditional/Ethnic")
print(f"   Positive Sentiment: 62.9%")

print(f"\n Ready for Deep Learning (LSTM)!")
print(f"{'='*60}")