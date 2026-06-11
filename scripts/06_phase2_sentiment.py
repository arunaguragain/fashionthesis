# PHASE 2 - SENTIMENT ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
warnings.filterwarnings('ignore')

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

print("="*60)
print("PHASE 2 - SENTIMENT ANALYSIS")
print("Combined Dataset: Nepal + Secondary")
print("="*60)

# STEP 1 - LOAD COMBINED DATASET


print("\n STEP 1: Loading Combined Dataset")
print("-"*60)

df = pd.read_csv(
    '../data/cleaned/combined_all_datasets.csv'
)
print(f" Loaded: {len(df):,} rows")
print(f"   Columns: {list(df.columns)}")

# Also load Nepal primary for comparison
nepal = pd.read_csv(
    '../data/cleaned/nepal_primary_only.csv'
)
print(f" Nepal primary: {len(nepal):,} rows")


# STEP 2 - INITIALIZE VADER
print("\n STEP 2: Initializing VADER Analyzer")
print("-"*60)

analyzer = SentimentIntensityAnalyzer()
print("VADER ready!")


# STEP 3 - PREPROCESS TEXT

print("\n STEP 3: Preprocessing Text")
print("-"*60)

def preprocess(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[#@]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_sentiment'] = df['text_clean'].apply(
    preprocess
)
print(f" Preprocessed {len(df):,} posts")

# STEP 4 - APPLY VADER SENTIMENT

print("\n STEP 4: Applying VADER Sentiment")
print("-"*60)
print("  Processing... (may take 2-3 mins for 105k rows)")

def get_vader_scores(text):
    scores = analyzer.polarity_scores(str(text))
    return (
        scores['neg'],
        scores['neu'],
        scores['pos'],
        scores['compound']
    )

def get_sentiment_label(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply VADER
results = df['text_sentiment'].apply(get_vader_scores)
df['vader_neg']      = results.apply(lambda x: x[0])
df['vader_neu']      = results.apply(lambda x: x[1])
df['vader_pos']      = results.apply(lambda x: x[2])
df['vader_compound'] = results.apply(lambda x: x[3])

# For secondary data — use pre_sentiment if available
# but override with VADER for consistency
df['sentiment'] = df['vader_compound'].apply(
    get_sentiment_label
)

print(f" Sentiment analysis complete!")


# STEP 5 - TREND SCORE CALCULATION

print("\n STEP 5: Calculating Trend Scores")
print("-"*60)

def calc_trend_score(row):
    sent_weight = {
        'Positive': 1.0,
        'Neutral':  0.5,
        'Negative': 0.0
    }.get(row['sentiment'], 0.5)

    likes    = min(float(row.get('likes', 0)),
                   10000) / 10000
    comments = min(float(row.get('comments', 0)),
                   1000) / 1000
    shares   = min(float(row.get('shares', 0)),
                   1000) / 1000
    views    = min(float(row.get('views', 0)),
                   1000000) / 1000000

    engagement = (
        likes * 0.4 +
        comments * 0.3 +
        shares * 0.2 +
        views * 0.1
    )
    return round(sent_weight * 0.4 +
                 engagement * 0.6, 4)

df['trend_score'] = df.apply(calc_trend_score, axis=1)
print(f" Trend scores calculated")

# STEP 6 - ANALYSIS BY SOURCE

print("\n STEP 6: Analysis by Source")
print("-"*60)

for source in df['source'].unique():
    sub = df[df['source'] == source]
    print(f"\n  {source} ({len(sub):,} rows):")
    sent = sub['sentiment'].value_counts()
    for s, c in sent.items():
        print(f"    {s}: {c:,} ({c/len(sub)*100:.1f}%)")

# STEP 7 - NEPAL PRIMARY SPECIFIC ANALYSIS

print("\n STEP 7: Nepal Primary Analysis")
print("-"*60)

nepal_df = df[df['source'] == 'Nepal_Primary'].copy()

print(f"\n  Total Nepal posts: {len(nepal_df):,}")
print(f"\n  Sentiment Distribution (Nepal only):")
nepal_sent = nepal_df['sentiment'].value_counts()
for s, c in nepal_sent.items():
    pct = c/len(nepal_df)*100
    bar = '█' * int(pct/3)
    print(f"    {s:10}: {c:4} ({pct:.1f}%) {bar}")

print(f"\n  By Platform (Nepal):")
plat_sent = pd.crosstab(
    nepal_df['platform'],
    nepal_df['sentiment'],
    normalize='index'
) * 100
print(plat_sent.round(1).to_string())

print(f"\n  Fashion Categories (Nepal):")
cat_counts = nepal_df['fashion_category'].value_counts()
for cat, cnt in cat_counts.items():
    pct = cnt/len(nepal_df)*100
    print(f"    {cat:25}: {cnt:4} ({pct:.1f}%)")

# STEP 8 - PHASE 1 VS PHASE 2 COMPARISON

print("\n STEP 8: Phase 1 vs Phase 2 Comparison")
print("-"*60)

phase1 = df[df['phase'] == 'Phase1']
phase2_nepal = df[
    (df['phase'] == 'Phase2') &
    (df['source'] == 'Nepal_Primary')
]

print(f"\n  Phase 1 Nepal posts : {len(phase1):,}")
print(f"  Phase 2 Nepal posts : {len(phase2_nepal):,}")

if len(phase1) > 0 and len(phase2_nepal) > 0:
    print(f"\n  Sentiment Comparison:")
    print(f"  {'':25} {'Phase1':>10} {'Phase2':>10}")
    print(f"  {'-'*45}")
    for sent in ['Positive', 'Neutral', 'Negative']:
        p1_pct = (phase1['sentiment'] == sent
                  ).mean() * 100
        p2_pct = (phase2_nepal['sentiment'] == sent
                  ).mean() * 100
        print(f"  {sent:25} {p1_pct:>9.1f}% "
              f"{p2_pct:>9.1f}%")

# STEP 9 - TREND RANKINGS

print("\n STEP 9: Trend Rankings")
print("-"*60)

# Nepal specific trend rankings
print("\n  🇳🇵 NEPAL SPECIFIC TREND RANKINGS:")
nepal_trend = nepal_df.groupby(
    'fashion_category'
).agg(
    total_posts    = ('text_clean', 'count'),
    avg_sentiment  = ('vader_compound', 'mean'),
    positive_ratio = ('sentiment',
                      lambda x: (x == 'Positive'
                                 ).sum() / len(x)),
    avg_likes      = ('likes', 'mean'),
    avg_views      = ('views', 'mean'),
    avg_trend_score= ('trend_score', 'mean'),
).reset_index()

nepal_trend['trend_rank'] = (
    nepal_trend['avg_trend_score'] * 0.5 +
    nepal_trend['positive_ratio']  * 0.3 +
    (nepal_trend['total_posts'] /
     nepal_trend['total_posts'].max()) * 0.2
)
nepal_trend = nepal_trend.sort_values(
    'trend_rank', ascending=False
).reset_index(drop=True)

medals = ['🥇','🥈','🥉','4️⃣ ','5️⃣ ','6️⃣ ']
for i, row in nepal_trend.iterrows():
    print(f"\n  {medals[i]} {row['fashion_category']}")
    print(f"     Score    : {row['trend_rank']:.4f}")
    print(f"     Posts    : {int(row['total_posts'])}")
    print(f"     Positive : {row['positive_ratio']*100:.1f}%")
    print(f"     Avg Likes: {row['avg_likes']:.0f}")

# Combined dataset trend rankings
print("\n\n  COMBINED DATASET TREND RANKINGS:")
combined_trend = df.groupby(
    'fashion_category'
).agg(
    total_posts    = ('text_clean', 'count'),
    positive_ratio = ('sentiment',
                      lambda x: (x == 'Positive'
                                 ).sum() / len(x)),
    avg_compound   = ('vader_compound', 'mean'),
).reset_index()
combined_trend = combined_trend.sort_values(
    'positive_ratio', ascending=False
)
for _, row in combined_trend.iterrows():
    print(f"    {row['fashion_category']:25}: "
          f"{row['positive_ratio']*100:.1f}% positive "
          f"({int(row['total_posts']):,} posts)")


# STEP 10 - VISUALIZATIONS

print("\n STEP 10: Creating Visualizations")
print("-"*60)

os.makedirs('../outputs/phase2', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'Phase 2 Sentiment Analysis — Combined Dataset\n'
    'Nepal Fashion Trend Prediction (105,443 records)',
    fontsize=14, fontweight='bold'
)

colors = {
    'Positive': '#2ecc71',
    'Neutral':  '#3498db',
    'Negative': '#e74c3c'
}

# Plot 1 — Overall sentiment pie
sent_counts = df['sentiment'].value_counts()
axes[0,0].pie(
    sent_counts.values,
    labels=sent_counts.index,
    colors=[colors[s] for s in sent_counts.index],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75
)
axes[0,0].set_title(
    'Overall Sentiment\n(Combined 105k dataset)',
    fontweight='bold'
)

# Plot 2 — Nepal sentiment pie
nepal_sent_counts = nepal_df['sentiment'].value_counts()
axes[0,1].pie(
    nepal_sent_counts.values,
    labels=nepal_sent_counts.index,
    colors=[colors[s] for s in nepal_sent_counts.index],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75
)
axes[0,1].set_title(
    'Nepal Primary Sentiment\n(2,172 posts)',
    fontweight='bold'
)

# Plot 3 — Sentiment by source
source_sent = pd.crosstab(
    df['source'], df['sentiment'],
    normalize='index'
) * 100
for col in ['Positive','Neutral','Negative']:
    if col not in source_sent.columns:
        source_sent[col] = 0
source_sent[['Positive','Neutral','Negative']].plot(
    kind='bar', ax=axes[0,2],
    color=['#2ecc71','#3498db','#e74c3c'],
    edgecolor='white', width=0.6
)
axes[0,2].set_title(
    'Sentiment by Data Source (%)',
    fontweight='bold'
)
axes[0,2].set_xlabel('')
axes[0,2].tick_params(axis='x', rotation=20)
axes[0,2].legend(title='Sentiment', fontsize=8)

# Plot 4 — Nepal trend rankings
axes[1,0].barh(
    nepal_trend['fashion_category'],
    nepal_trend['trend_rank'],
    color='#e91e8c', edgecolor='white'
)
axes[1,0].set_title(
    'Nepal Fashion Trend Rankings',
    fontweight='bold'
)
axes[1,0].set_xlabel('Trend Score')
for i, v in enumerate(nepal_trend['trend_rank']):
    axes[1,0].text(
        v + 0.002, i,
        f'{v:.4f}', va='center', fontsize=9
    )

# Plot 5 — VADER distribution
axes[1,1].hist(
    df['vader_compound'], bins=40,
    color='#9c27b0', edgecolor='white', alpha=0.8
)
axes[1,1].axvline(
    x=0.05, color='#2ecc71',
    linestyle='--', label='Positive (0.05)'
)
axes[1,1].axvline(
    x=-0.05, color='#e74c3c',
    linestyle='--', label='Negative (-0.05)'
)
axes[1,1].set_title(
    'VADER Compound Score Distribution',
    fontweight='bold'
)
axes[1,1].set_xlabel('Compound Score')
axes[1,1].legend(fontsize=8)

# Plot 6 — Category distribution Nepal
cat_colors = [
    '#9b59b6','#e67e22','#1abc9c',
    '#e74c3c','#3498db','#f39c12'
]
bars = axes[1,2].bar(
    range(len(cat_counts)),
    cat_counts.values,
    color=cat_colors[:len(cat_counts)],
    edgecolor='white'
)
axes[1,2].set_xticks(range(len(cat_counts)))
axes[1,2].set_xticklabels(
    cat_counts.index, rotation=25,
    ha='right', fontsize=8
)
axes[1,2].set_title(
    'Fashion Categories (Nepal Primary)',
    fontweight='bold'
)
for bar, val in zip(bars, cat_counts.values):
    axes[1,2].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        str(val), ha='center',
        fontsize=8, fontweight='bold'
    )

plt.tight_layout()
plt.savefig(
    '../outputs/phase2/phase2_sentiment_analysis.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print(" Chart saved: phase2_sentiment_analysis.png")

# STEP 11 - SAVE FINAL DATASET WITH SENTIMENT

print("\n STEP 11: Saving Final Dataset")
print("-"*60)

df.to_csv(
    '../data/cleaned/combined_with_sentiment.csv',
    index=False
)
print(f" combined_with_sentiment.csv — {len(df):,} rows")

nepal_df.to_csv(
    '../data/cleaned/nepal_with_sentiment.csv',
    index=False
)
print(f" nepal_with_sentiment.csv — {len(nepal_df):,} rows")

nepal_trend.to_csv(
    '../data/cleaned/nepal_trend_rankings.csv',
    index=False
)
print(f" nepal_trend_rankings.csv saved")


# FINAL SUMMARY

print(f"\n{'='*60}")
print(f"PHASE 2 SENTIMENT ANALYSIS COMPLETE")
print(f"{'='*60}")

print(f"\n COMBINED DATASET (105,443 rows):")
for s, c in sent_counts.items():
    print(f"  {s:10}: {c:,} ({c/len(df)*100:.1f}%)")

print(f"\n🇳🇵 NEPAL PRIMARY (2,172 rows):")
for s, c in nepal_sent_counts.items():
    print(f"  {s:10}: {c:,} ({c/len(nepal_df)*100:.1f}%)")

print(f"\n TOP NEPAL FASHION TREND:")
top = nepal_trend.iloc[0]
print(f"  {top['fashion_category']}")
print(f"  Score: {top['trend_rank']:.4f}")
print(f"  Positive: {top['positive_ratio']*100:.1f}%")

print(f"\n Ready for Machine Learning (Phase 2)!")
print(f"{'='*60}")