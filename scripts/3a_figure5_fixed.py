# ============================================
# FIXED CHART 5 - NO OVERLAPPING
# By: Aruna Guragain
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('../data/cleaned/fashion_data_with_sentiment.csv')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

df['text_clean'] = df['text_clean'].fillna('')
X_text     = df['text_clean']
y_sent     = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_sent, test_size=0.2,
    random_state=42, stratify=y_sent
)

tfidf = TfidfVectorizer(
    max_features=5000, ngram_range=(1,2),
    min_df=2, stop_words='english', sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

nb_model  = MultinomialNB(alpha=0.1)
svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
nb_model.fit(X_train_tfidf,  y_train)
svm_model.fit(X_train_tfidf, y_train)

nb_cv  = cross_val_score(nb_model,  X_train_tfidf, y_train, cv=5) * 100
svm_cv = cross_val_score(svm_model, X_train_tfidf, y_train, cv=5) * 100

os.makedirs('../outputs', exist_ok=True)

# ============================================
# CHART 5a — Cross Validation Box Plot (FIXED)
# ============================================

fig1, ax1 = plt.subplots(figsize=(8, 6))
fig1.suptitle(
    'Cross Validation Score Distribution\n'
    'Fashion Trend Prediction — Nepali Female (18-26)',
    fontsize=13, fontweight='bold'
)

cv_data   = [nb_cv, svm_cv]
cv_labels = ['Naive Bayes', 'SVM (LinearSVC)']
cv_colors = ['#3498db', '#e67e22']

bp = ax1.boxplot(
    cv_data,
    labels=cv_labels,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker='o', markerfacecolor='gray',
                    markersize=6)
)
for patch, color in zip(bp['boxes'], cv_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Add mean value labels
for i, (cv, label) in enumerate(zip(cv_data, cv_labels), 1):
    ax1.text(
        i, cv.mean() + 1.5,
        f'Mean: {cv.mean():.1f}%',
        ha='center', fontsize=10, fontweight='bold'
    )

ax1.set_title('5-Fold Cross Validation Accuracy',
              fontweight='bold', pad=12)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_ylim(40, 100)
ax1.grid(axis='y', alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('../outputs/chart5a_crossvalidation.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 5a saved: chart5a_crossvalidation.png")

# ============================================
# CHART 5b — Trend Score by Platform (FIXED)
# Separated into its own figure, legend outside
# ============================================

platform_trend = df.groupby(
    ['platform', 'fashion_category']
)['trend_score'].mean().unstack(fill_value=0)

fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.suptitle(
    'Average Trend Score by Platform & Fashion Category\n'
    'Nepali Female Fashion (18-26)',
    fontsize=13, fontweight='bold'
)

category_colors = [
    '#3498db', '#e67e22', '#2ecc71',
    '#e74c3c', '#9b59b6', '#1abc9c'
]

bars = platform_trend.plot(
    kind='bar',
    ax=ax2,
    color=category_colors[:len(platform_trend.columns)],
    edgecolor='white',
    width=0.65
)

ax2.set_xlabel('Platform', fontsize=11)
ax2.set_ylabel('Average Trend Score', fontsize=11)
ax2.set_title(
    'TikTok shows higher trend scores across most categories',
    fontsize=10, style='italic', pad=8
)
ax2.tick_params(axis='x', rotation=0, labelsize=11)
ax2.set_ylim(0, 0.65)
ax2.grid(axis='y', alpha=0.3)

# Legend OUTSIDE the plot — no overlap!
ax2.legend(
    title='Fashion Category',
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    fontsize=9,
    title_fontsize=10,
    framealpha=0.9
)

# Add value labels on bars
for container in ax2.containers:
    ax2.bar_label(
        container,
        fmt='%.2f',
        fontsize=7.5,
        padding=2,
        rotation=90
    )

plt.tight_layout(pad=2.0)
plt.savefig('../outputs/chart5b_platform_trend.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 5b saved: chart5b_platform_trend.png")

print("\n✅ Both fixed charts saved successfully!")
print("   outputs/chart5a_crossvalidation.png")
print("   outputs/chart5b_platform_trend.png")