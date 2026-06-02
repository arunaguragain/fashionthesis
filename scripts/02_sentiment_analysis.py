import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import warnings
warnings.filterwarnings('ignore')

# STEP 1 - LOAD CLEANED DATA
print("="*55)
print("STEP 1: LOADING CLEANED FASHION DATA")
print("="*55)

df = pd.read_csv('../data/cleaned/fashion_data_cleaned.csv')
print(f" Loaded {len(df)} clean fashion posts")
print(f"   Columns: {list(df.columns)}")

# STEP 2 - INITIALIZE VADER ANALYZER

print("\n" + "="*55)
print("STEP 2: INITIALIZING VADER SENTIMENT ANALYZER")
print("="*55)

analyzer = SentimentIntensityAnalyzer()
print(" VADER Analyzer ready!")


# STEP 3 - PREPROCESS TEXT FOR SENTIMENT

print("\n" + "="*55)
print("STEP 3: PREPROCESSING TEXT")
print("="*55)

def preprocess_for_sentiment(text):
    """
    Clean text specifically for sentiment analysis.
    We keep emojis because VADER understands them!
    Example: ❤️ = positive, 😡 = negative
    """
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove excessive punctuation but keep ! and ?
    # because they carry sentiment meaning
    text = re.sub(r'[#@]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_sentiment'] = df['text_clean'].apply(preprocess_for_sentiment)
print(f" Text preprocessed for {len(df)} posts")
print(f"\nExample original : {df['text_clean'].iloc[0][:100]}")
print(f"Example processed: {df['text_sentiment'].iloc[0][:100]}")

# STEP 4 - APPLY VADER SENTIMENT ANALYSIS

print("\n" + "="*55)
print("STEP 4: APPLYING VADER SENTIMENT ANALYSIS")
print("="*55)

def get_sentiment_scores(text):
    """
    VADER returns 4 scores:
    - neg   : negative sentiment (0 to 1)
    - neu   : neutral sentiment (0 to 1)
    - pos   : positive sentiment (0 to 1)
    - compound: overall score (-1 to +1)
      > 0.05  = Positive
      < -0.05 = Negative
      else    = Neutral
    """
    scores = analyzer.polarity_scores(str(text))
    return scores

def get_sentiment_label(compound_score):
    """
    Convert compound score to human readable label
    Threshold based on VADER documentation
    """
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply to all posts
print("Analyzing sentiment for all posts...")
scores_list = df['text_sentiment'].apply(get_sentiment_scores)

# Extract individual scores
df['vader_neg']      = scores_list.apply(lambda x: x['neg'])
df['vader_neu']      = scores_list.apply(lambda x: x['neu'])
df['vader_pos']      = scores_list.apply(lambda x: x['pos'])
df['vader_compound'] = scores_list.apply(lambda x: x['compound'])

# Add sentiment label
df['sentiment'] = df['vader_compound'].apply(get_sentiment_label)

print(f" Sentiment analysis complete for {len(df)} posts!")

# Show examples
print("\n--- SAMPLE RESULTS ---")
for sentiment in ['Positive', 'Neutral', 'Negative']:
    sample = df[df['sentiment'] == sentiment]['text_sentiment'].iloc[0]
    score  = df[df['sentiment'] == sentiment]['vader_compound'].iloc[0]
    print(f"\n{sentiment} (score: {score:.3f}):")
    print(f"  {sample[:120]}...")


# STEP 5 - SENTIMENT DISTRIBUTION ANALYSIS

print("\n" + "="*55)
print("STEP 5: SENTIMENT DISTRIBUTION ANALYSIS")
print("="*55)

# Overall distribution
sentiment_counts = df['sentiment'].value_counts()
sentiment_pct    = df['sentiment'].value_counts(normalize=True) * 100

print("\nOverall Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    pct = sentiment_pct[sentiment]
    bar = '█' * int(pct / 2)
    print(f"  {sentiment:10} : {count:4} posts ({pct:.1f}%) {bar}")

# By Platform
print("\nSentiment by Platform:")
platform_sentiment = pd.crosstab(
    df['platform'],
    df['sentiment'],
    normalize='index'
) * 100

print(platform_sentiment.round(1).to_string())

# By Language
print("\nSentiment by Language:")
language_sentiment = pd.crosstab(
    df['language'],
    df['sentiment'],
    normalize='index'
) * 100
print(language_sentiment.round(1).to_string())

# STEP 6 - FASHION CATEGORY CLASSIFICATION

print("\n" + "="*55)
print("STEP 6: FASHION CATEGORY CLASSIFICATION")
print("="*55)

def classify_fashion_category(text):
    """
    Classify posts into fashion categories
    based on keywords found in the text
    """
    text = str(text).lower()

    categories = {
        'Traditional/Ethnic': [
            'saree', 'sari', 'lehenga', 'kurta', 'kurti',
            'ethnic', 'traditional', 'daura', 'suruwal',
            'gunyo', 'cholo', 'dhaka', 'madise', 'nepali dress',
            'dashain', 'tihar', 'teej', 'festival', 'cultural',
            'पोशाक', 'लुगा', 'साडी', 'कुर्ता'
        ],
        'Western/Casual': [
            'jeans', 'top', 'tshirt', 't-shirt', 'casual',
            'western', 'hoodie', 'jacket', 'sneaker', 'denim',
            'shorts', 'skirt', 'crop', 'streetwear', 'urban',
            'street style', 'street fashion'
        ],
        'Indo-Western/Fusion': [
            'indo western', 'indo-western', 'fusion',
            'modern', 'contemporary', 'blend', 'mix',
            'indo', 'bollywood', 'semi formal'
        ],
        'Formal/Professional': [
            'formal', 'office', 'professional', 'business',
            'suit', 'blazer', 'workwear', 'corporate'
        ],
        'Accessories': [
            'handbag', 'bag', 'jewel', 'jewelry', 'jewellery',
            'necklace', 'earring', 'bracelet', 'shoes', 'heel',
            'sandal', 'accessories', 'watch', 'scarf', 'dupatta'
        ]
    }

    for category, keywords in categories.items():
        if any(kw in text for kw in keywords):
            return category

    return 'General Fashion'

df['fashion_category'] = df['text_sentiment'].apply(
    classify_fashion_category
)

print("Fashion Category Distribution:")
category_counts = df['fashion_category'].value_counts()
for cat, count in category_counts.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {cat:25} : {count:4} ({pct:.1f}%) {bar}")

# STEP 7 - TREND SCORING

print("\n" + "="*55)
print("STEP 7: CALCULATING TREND SCORES")
print("="*55)

def calculate_trend_score(row):
    """
    Trend score combines:
    - Sentiment (positive = higher trend)
    - Engagement (likes, comments, shares, views)
    - Recency (newer posts score higher)

    Formula:
    trend_score = (sentiment_weight * 0.4) +
                  (engagement_weight * 0.6)
    """
    # Sentiment weight
    sentiment_weight = {
        'Positive': 1.0,
        'Neutral' : 0.5,
        'Negative': 0.0
    }.get(row['sentiment'], 0.5)

    # Engagement weight (normalized)
    likes    = min(row['likes'], 10000) / 10000
    comments = min(row['comments'], 1000) / 1000
    shares   = min(row['shares'], 1000) / 1000
    views    = min(row['views'], 1000000) / 1000000

    engagement_weight = (
        likes * 0.4 +
        comments * 0.3 +
        shares * 0.2 +
        views * 0.1
    )

    # Combined trend score
    trend_score = (sentiment_weight * 0.4) + \
                  (engagement_weight * 0.6)

    return round(trend_score, 4)

df['trend_score'] = df.apply(calculate_trend_score, axis=1)

print("Trend Score Statistics:")
print(f"  Average  : {df['trend_score'].mean():.4f}")
print(f"  Maximum  : {df['trend_score'].max():.4f}")
print(f"  Minimum  : {df['trend_score'].min():.4f}")

print("\n TOP 5 TRENDING POSTS:")
top_trending = df.nlargest(5, 'trend_score')[
    ['platform', 'text_sentiment', 'sentiment',
     'fashion_category', 'trend_score']
]
for i, row in top_trending.iterrows():
    print(f"\n{row['platform']} | {row['sentiment']} | "
          f"{row['fashion_category']} | Score: {row['trend_score']}")
    print(f"  {row['text_sentiment'][:100]}...")

# STEP 8 - VISUALIZATIONS

print("\n" + "="*55)
print("STEP 8: CREATING VISUALIZATIONS")
print("="*55)

os.makedirs('../outputs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'Fashion Trend Prediction — Sentiment Analysis Results\n'
    'Nepali Female Fashion (Age 18-26)',
    fontsize=16, fontweight='bold', y=0.98
)

# Colors
colors = {
    'Positive': '#2ecc71',
    'Neutral' : '#3498db',
    'Negative': '#e74c3c'
}
color_list = [colors[s] for s in sentiment_counts.index]

# Plot 1 — Sentiment Distribution Pie
axes[0,0].pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    colors=color_list,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 11}
)
axes[0,0].set_title('Overall Sentiment Distribution',
                     fontweight='bold')

# Plot 2 — Sentiment by Platform Bar
platform_sentiment.plot(
    kind='bar',
    ax=axes[0,1],
    color=['#e74c3c', '#3498db', '#2ecc71'],
    edgecolor='white'
)
axes[0,1].set_title('Sentiment by Platform (%)',
                     fontweight='bold')
axes[0,1].set_xlabel('Platform')
axes[0,1].set_ylabel('Percentage (%)')
axes[0,1].legend(title='Sentiment')
axes[0,1].tick_params(axis='x', rotation=0)

# Plot 3 — Fashion Category Distribution
category_colors = [
    '#9b59b6','#e67e22','#1abc9c',
    '#e74c3c','#3498db','#f39c12'
]
axes[0,2].bar(
    range(len(category_counts)),
    category_counts.values,
    color=category_colors[:len(category_counts)],
    edgecolor='white'
)
axes[0,2].set_xticks(range(len(category_counts)))
axes[0,2].set_xticklabels(
    category_counts.index,
    rotation=30, ha='right', fontsize=9
)
axes[0,2].set_title('Fashion Category Distribution',
                     fontweight='bold')
axes[0,2].set_ylabel('Number of Posts')

# Plot 4 — VADER Compound Score Distribution
axes[1,0].hist(
    df['vader_compound'],
    bins=30,
    color='#3498db',
    edgecolor='white',
    alpha=0.8
)
axes[1,0].axvline(x=0.05,  color='green', linestyle='--',
                   label='Positive threshold')
axes[1,0].axvline(x=-0.05, color='red',   linestyle='--',
                   label='Negative threshold')
axes[1,0].set_title('VADER Compound Score Distribution',
                     fontweight='bold')
axes[1,0].set_xlabel('Compound Score')
axes[1,0].set_ylabel('Number of Posts')
axes[1,0].legend()

# Plot 5 — Trend Score by Category
category_trend = df.groupby('fashion_category')['trend_score'].mean()
axes[1,1].barh(
    category_trend.index,
    category_trend.values,
    color='#e67e22',
    edgecolor='white'
)
axes[1,1].set_title('Average Trend Score by Category',
                     fontweight='bold')
axes[1,1].set_xlabel('Average Trend Score')

# Plot 6 — Language Distribution
language_counts = df['language'].value_counts()
axes[1,2].pie(
    language_counts.values,
    labels=language_counts.index,
    colors=['#3498db', '#e67e22'],
    autopct='%1.1f%%',
    startangle=90
)
axes[1,2].set_title('Language Distribution',
                     fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/sentiment_analysis_results.png',
            dpi=150, bbox_inches='tight')
print(" Chart saved to: outputs/sentiment_analysis_results.png")
plt.show()


# STEP 9 - SAVE FINAL DATASET

print("\n" + "="*55)
print("STEP 9: SAVING FINAL DATASET WITH SENTIMENT")
print("="*55)

df.to_csv(
    '../data/cleaned/fashion_data_with_sentiment.csv',
    index=False
)
print("Saved: data/cleaned/fashion_data_with_sentiment.csv")


# FINAL SUMMARY

print("\n" + "="*55)
print("SENTIMENT ANALYSIS COMPLETE — FINAL SUMMARY")
print("="*55)
print(f"Total posts analyzed     : {len(df)}")
print(f"\nSentiment Breakdown:")
for s, c in sentiment_counts.items():
    print(f"  {s:10} : {c} posts ({c/len(df)*100:.1f}%)")
print(f"\nTop Trending Category    : "
      f"{category_trend.idxmax()}")
print(f"Avg Trend Score          : "
      f"{df['trend_score'].mean():.4f}")
print(f"\n Ready for Machine Learning step!")
print("="*55)