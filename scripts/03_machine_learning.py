import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("FASHION TREND PREDICTION — MACHINE LEARNING")
print("By: Aruna Guragain")
print("="*55)

# STEP 1 - LOAD DATA

print("\n" + "="*55)
print("STEP 1: LOADING SENTIMENT DATA")
print("="*55)

df = pd.read_csv('../data/cleaned/fashion_data_with_sentiment.csv')
print(f" Loaded {len(df)} posts")
print(f"   Columns: {list(df.columns)}")

# Check class distribution
print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts().to_string())
print(f"\nFashion Category Distribution:")
print(df['fashion_category'].value_counts().to_string())


# STEP 2 - PREPARE FEATURES AND LABELS

print("\n" + "="*55)
print("STEP 2: PREPARING FEATURES AND LABELS")
print("="*55)

# Fill any missing text
df['text_clean'] = df['text_clean'].fillna('')

# TARGET 1: Sentiment Prediction 
# X = text features
# y = sentiment label (Positive/Neutral/Negative)
X_text = df['text_clean']
y_sentiment = df['sentiment']

# TARGET 2: Fashion Category Prediction 
y_category = df['fashion_category']

# ADDITIONAL FEATURES 
# Engagement features for trend scoring
df['engagement_score'] = (
    df['likes'] * 0.4 +
    df['comments'] * 0.3 +
    df['shares'] * 0.2 +
    df['views'].clip(upper=1000000) / 1000000 * 100 * 0.1
)

print(f" Text features prepared: {len(X_text)} samples")
print(f" Sentiment labels: {y_sentiment.nunique()} classes")
print(f" Category labels: {y_category.nunique()} classes")


# STEP 3 - TRAIN/TEST SPLIT

print("\n" + "="*55)
print("STEP 3: SPLITTING DATA — TRAIN/TEST")
print("="*55)

# 80% training, 20% testing
# random_state=42 ensures reproducibility
X_train, X_test, y_train_sent, y_test_sent = train_test_split(
    X_text, y_sentiment,
    test_size=0.2,
    random_state=42,
    stratify=y_sentiment  # ensures balanced split
)

_, _, y_train_cat, y_test_cat = train_test_split(
    X_text, y_category,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

print(f"Training set  : {len(X_train)} posts (80%)")
print(f"Testing set   : {len(X_test)} posts  (20%)")
print(f"\nTraining sentiment distribution:")
print(y_train_sent.value_counts().to_string())

# STEP 4 - TFIDF VECTORIZER

print("\n" + "="*55)
print("STEP 4: TFIDF VECTORIZATION")
print("="*55)

# TF-IDF converts text to numbers
# TF = Term Frequency (how often word appears)
# IDF = Inverse Document Frequency (how unique word is)
tfidf = TfidfVectorizer(
    max_features=5000,    # top 5000 most important words
    ngram_range=(1, 2),   # single words + word pairs
    min_df=2,             # word must appear at least twice
    stop_words='english', # remove common words (the, is, at)
    sublinear_tf=True     # apply log normalization
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f" Vocabulary size : {len(tfidf.vocabulary_)} words")
print(f" Training matrix : {X_train_tfidf.shape}")
print(f" Testing matrix  : {X_test_tfidf.shape}")

# Show top features
feature_names = tfidf.get_feature_names_out()
print(f"\nTop 20 most important words:")
print(', '.join(feature_names[:20]))


# STEP 5 - MODEL 1: NAIVE BAYES

print("\n" + "="*55)
print("STEP 5: MODEL 1 — NAIVE BAYES")
print("="*55)

# Naive Bayes — works well with text classification
# Based on probability (Bayes theorem)
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, y_train_sent)

# Predictions
nb_predictions = nb_model.predict(X_test_tfidf)
nb_accuracy    = accuracy_score(y_test_sent, nb_predictions)

# Cross validation (5-fold)
nb_cv_scores = cross_val_score(
    nb_model, X_train_tfidf, y_train_sent,
    cv=5, scoring='accuracy'
)

print(f" Naive Bayes Training Complete!")
print(f"\n   Test Accuracy     : {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")
print(f"   CV Accuracy (5-fold): {nb_cv_scores.mean():.4f} "
      f"(±{nb_cv_scores.std():.4f})")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test_sent, nb_predictions))

# STEP 6 - MODEL 2: SVM

print("\n" + "="*55)
print("STEP 6: MODEL 2 — SUPPORT VECTOR MACHINE (SVM)")
print("="*55)

# SVM — finds the best boundary between classes
# LinearSVC is fastest version for text data
svm_model = LinearSVC(
    C=1.0,           # regularization parameter
    max_iter=2000,   # maximum iterations
    random_state=42
)
svm_model.fit(X_train_tfidf, y_train_sent)

# Predictions
svm_predictions = svm_model.predict(X_test_tfidf)
svm_accuracy    = accuracy_score(y_test_sent, svm_predictions)

# Cross validation
svm_cv_scores = cross_val_score(
    svm_model, X_train_tfidf, y_train_sent,
    cv=5, scoring='accuracy'
)

print(f" SVM Training Complete!")
print(f"\n   Test Accuracy      : {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
print(f"   CV Accuracy (5-fold): {svm_cv_scores.mean():.4f} "
      f"(±{svm_cv_scores.std():.4f})")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test_sent, svm_predictions))

# STEP 7 - MODEL COMPARISON

print("\n" + "="*55)
print("STEP 7: MODEL COMPARISON")
print("="*55)

print(f"{'Model':<20} {'Test Acc':>10} {'CV Acc':>10} {'CV Std':>10}")
print("-"*55)
print(f"{'Naive Bayes':<20} "
      f"{nb_accuracy*100:>9.2f}% "
      f"{nb_cv_scores.mean()*100:>9.2f}% "
      f"{nb_cv_scores.std()*100:>9.2f}%")
print(f"{'SVM (LinearSVC)':<20} "
      f"{svm_accuracy*100:>9.2f}% "
      f"{svm_cv_scores.mean()*100:>9.2f}% "
      f"{svm_cv_scores.std()*100:>9.2f}%")

# Select best model
if svm_accuracy >= nb_accuracy:
    best_model       = svm_model
    best_model_name  = "SVM"
    best_predictions = svm_predictions
    best_accuracy    = svm_accuracy
else:
    best_model       = nb_model
    best_model_name  = "Naive Bayes"
    best_predictions = nb_predictions
    best_accuracy    = nb_accuracy

print(f"\n BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy*100:.2f}%")

# STEP 8 - FASHION CATEGORY PREDICTION

print("\n" + "="*55)
print("STEP 8: FASHION CATEGORY PREDICTION MODEL")
print("="*55)

# Train best algorithm on fashion categories
cat_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
cat_model.fit(
    tfidf.transform(X_train),
    y_train_cat
)

cat_predictions = cat_model.predict(tfidf.transform(X_test))
cat_accuracy    = accuracy_score(y_test_cat, cat_predictions)

cat_cv_scores = cross_val_score(
    cat_model,
    tfidf.transform(X_train),
    y_train_cat,
    cv=5, scoring='accuracy'
)

print(f" Category Prediction Model Complete!")
print(f"   Test Accuracy      : {cat_accuracy:.4f} ({cat_accuracy*100:.2f}%)")
print(f"   CV Accuracy (5-fold): {cat_cv_scores.mean():.4f} "
      f"(±{cat_cv_scores.std():.4f})")
print(f"\nDetailed Report:")
print(classification_report(y_test_cat, cat_predictions))

# STEP 9 - TREND PREDICTION

print("\n" + "="*55)
print("STEP 9: FASHION TREND PREDICTION")
print("="*55)

# Combine sentiment + category + engagement
# to predict overall trending score

trend_analysis = df.groupby('fashion_category').agg(
    total_posts    = ('text_clean', 'count'),
    avg_sentiment  = ('vader_compound', 'mean'),
    positive_ratio = ('sentiment', lambda x:
                      (x == 'Positive').sum() / len(x)),
    avg_likes      = ('likes', 'mean'),
    avg_comments   = ('comments', 'mean'),
    avg_shares     = ('shares', 'mean'),
    avg_views      = ('views', 'mean'),
    avg_trend_score= ('trend_score', 'mean'),
).reset_index()

# Calculate final trend rank
trend_analysis['trend_rank'] = (
    trend_analysis['avg_trend_score'] * 0.5 +
    trend_analysis['positive_ratio']  * 0.3 +
    (trend_analysis['total_posts'] /
     trend_analysis['total_posts'].max()) * 0.2
)

trend_analysis = trend_analysis.sort_values(
    'trend_rank', ascending=False
).reset_index(drop=True)

trend_analysis['rank'] = range(1, len(trend_analysis) + 1)

print(" FASHION TREND RANKINGS FOR NEPALI FEMALES (18-26):")
print("-"*60)
for _, row in trend_analysis.iterrows():
    medal = ['🥇','🥈','🥉','4️⃣ ','5️⃣ ','6️⃣ '][int(row['rank'])-1]
    print(f"\n{medal} {row['fashion_category']}")
    print(f"   Trend Rank Score : {row['trend_rank']:.4f}")
    print(f"   Total Posts      : {int(row['total_posts'])}")
    print(f"   Positive Ratio   : {row['positive_ratio']*100:.1f}%")
    print(f"   Avg Likes        : {row['avg_likes']:.0f}")
    print(f"   Avg Views        : {row['avg_views']:.0f}")

# STEP 10 - VISUALIZATIONS

print("\n" + "="*55)
print("STEP 10: CREATING ML VISUALIZATIONS")
print("="*55)

os.makedirs('../outputs', exist_ok=True)

#  Chart 4: Model Comparison 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Machine Learning Model Results\n'
    'Fashion Trend Prediction — Nepali Female (18-26)',
    fontsize=14, fontweight='bold'
)

# Accuracy comparison bar
models      = ['Naive Bayes', 'SVM (LinearSVC)']
accuracies  = [nb_accuracy * 100, svm_accuracy * 100]
bar_colors  = ['#3498db', '#e67e22']

bars = axes[0].bar(models, accuracies,
                   color=bar_colors, edgecolor='white',
                   width=0.4)
axes[0].set_title('Model Accuracy Comparison',
                  fontweight='bold', pad=12)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{acc:.2f}%',
        ha='center', fontweight='bold', fontsize=11
    )

# Confusion matrix — best model
cm = confusion_matrix(
    y_test_sent, best_predictions,
    labels=['Positive', 'Neutral', 'Negative']
)
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='Blues', ax=axes[1],
    xticklabels=['Positive', 'Neutral', 'Negative'],
    yticklabels=['Positive', 'Neutral', 'Negative'],
    annot_kws={'size': 12}
)
axes[1].set_title(f'Confusion Matrix\n({best_model_name})',
                  fontweight='bold', pad=12)
axes[1].set_xlabel('Predicted', fontsize=10)
axes[1].set_ylabel('Actual', fontsize=10)

# Trend ranking bar chart
colors_trend = [
    '#f1c40f','#95a5a6','#cd7f32',
    '#3498db','#2ecc71','#e74c3c'
]
bars2 = axes[2].barh(
    trend_analysis['fashion_category'],
    trend_analysis['trend_rank'],
    color=colors_trend[:len(trend_analysis)],
    edgecolor='white', height=0.5
)
axes[2].set_title('Fashion Trend Rankings\nNepali Female 18-26',
                  fontweight='bold', pad=12)
axes[2].set_xlabel('Trend Rank Score')
for bar, val in zip(bars2, trend_analysis['trend_rank']):
    axes[2].text(
        val + 0.002,
        bar.get_y() + bar.get_height()/2,
        f'{val:.4f}',
        va='center', fontsize=9, fontweight='bold'
    )

plt.tight_layout(pad=3.0)
plt.savefig('../outputs/chart4_ml_results.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(" Chart 4 saved: chart4_ml_results.png")

#  Chart 5: Cross Validation Scores 
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    'Cross Validation Analysis\n'
    'Fashion Trend Prediction — Nepali Female (18-26)',
    fontsize=14, fontweight='bold'
)

# CV scores comparison
cv_data   = [nb_cv_scores * 100, svm_cv_scores * 100]
cv_labels = ['Naive Bayes', 'SVM']
cv_colors = ['#3498db', '#e67e22']

bp = axes2[0].boxplot(
    cv_data, labels=cv_labels,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2)
)
for patch, color in zip(bp['boxes'], cv_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes2[0].set_title('Cross Validation Score Distribution',
                   fontweight='bold', pad=12)
axes2[0].set_ylabel('Accuracy (%)')
axes2[0].set_ylim(0, 100)

# Trend score by platform
platform_trend = df.groupby(
    ['platform', 'fashion_category']
)['trend_score'].mean().unstack(fill_value=0)

platform_trend.plot(
    kind='bar', ax=axes2[1],
    edgecolor='white', width=0.6
)
axes2[1].set_title('Trend Score by Platform & Category',
                   fontweight='bold', pad=12)
axes2[1].set_xlabel('Platform')
axes2[1].set_ylabel('Average Trend Score')
axes2[1].legend(
    title='Category', fontsize=8,
    title_fontsize=8, loc='upper right'
)
axes2[1].tick_params(axis='x', rotation=0)

plt.tight_layout(pad=3.0)
plt.savefig('../outputs/chart5_cv_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(" Chart 5 saved: chart5_cv_analysis.png")


# STEP 11 - LIVE PREDICTION DEMO

print("\n" + "="*55)
print("STEP 11: LIVE PREDICTION DEMO")
print("="*55)

def predict_fashion_trend(text):
    """
    Given any fashion post text,
    predict its sentiment and category
    """
    text_vec  = tfidf.transform([text])
    sentiment = best_model.predict(text_vec)[0]
    category  = cat_model.predict(text_vec)[0]

    from vaderSentiment.vaderSentiment import (
        SentimentIntensityAnalyzer
    )
    analyzer  = SentimentIntensityAnalyzer()
    score     = analyzer.polarity_scores(text)['compound']

    trend_potential = " HIGH" if score > 0.3 else \
                      " MEDIUM" if score > 0 else \
                      " LOW"

    return {
        'text'           : text[:80] + '...',
        'sentiment'      : sentiment,
        'vader_score'    : round(score, 4),
        'category'       : category,
        'trend_potential': trend_potential
    }

# Test with sample posts
test_posts = [
    "Beautiful nepali saree for dashain festival! "
    "Love this traditional look! ❤️ #nepalifashion",

    "New kurti collection available. "
    "DM to order. Delivery all over Nepal 🇳🇵",

    "Poor quality fabric, very disappointed "
    "with this dress. Not worth the price 😞",

    "Outfit under Rs 2000! Western style "
    "casual look for college girls nepal 🎓",

    "Indo western fusion wear perfect for "
    "weddings and parties in Nepal ✨💕"
]

print("Testing prediction model with sample posts:\n")
for i, post in enumerate(test_posts, 1):
    result = predict_fashion_trend(post)
    print(f"Test {i}:")
    print(f"  Text      : {result['text']}")
    print(f"  Sentiment : {result['sentiment']}")
    print(f"  Score     : {result['vader_score']}")
    print(f"  Category  : {result['category']}")
    print(f"  Trending  : {result['trend_potential']}")
    print()

# STEP 12 - SAVE RESULTS

print("="*55)
print("STEP 12: SAVING ALL RESULTS")
print("="*55)

# Save trend analysis
trend_analysis.to_csv(
    '../data/cleaned/trend_analysis_results.csv',
    index=False
)
print(" Saved: data/cleaned/trend_analysis_results.csv")

# Save final complete dataset
df.to_csv(
    '../data/cleaned/fashion_data_final.csv',
    index=False
)
print(" Saved: data/cleaned/fashion_data_final.csv")


# FINAL SUMMARY

print("\n" + "="*55)
print("MACHINE LEARNING COMPLETE — FINAL SUMMARY")
print("="*55)
print(f"Total posts used         : {len(df)}")
print(f"Training posts           : {len(X_train)}")
print(f"Testing posts            : {len(X_test)}")
print(f"\nModel Performance:")
print(f"  Naive Bayes Accuracy   : {nb_accuracy*100:.2f}%")
print(f"  SVM Accuracy           : {svm_accuracy*100:.2f}%")
print(f"  Best Model             : {best_model_name}")
print(f"  Best Accuracy          : {best_accuracy*100:.2f}%")
print(f"  Category Accuracy      : {cat_accuracy*100:.2f}%")
print(f"\nTop Fashion Trend       : "
      f"{trend_analysis.iloc[0]['fashion_category']}")
print(f"Trend Score             : "
      f"{trend_analysis.iloc[0]['trend_rank']:.4f}")
print(f"\n READY FOR THESIS WRITING!") 
print("="*55)