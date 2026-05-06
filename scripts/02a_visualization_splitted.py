# ============================================
# FASHION TREND PREDICTION - FIXED VISUALIZATIONS
# By: Aruna Guragain
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Load data
df = pd.read_csv('../data/cleaned/fashion_data_with_sentiment.csv')
print(f"✅ Loaded {len(df)} posts for visualization")

os.makedirs('../outputs', exist_ok=True)

# ============================================
# CHART 1 - SENTIMENT OVERVIEW (3 charts)
# ============================================

fig = plt.figure(figsize=(20, 6))
fig.suptitle(
    'Fashion Trend Prediction — Sentiment Analysis\n'
    'Nepali Female Fashion (Age 18–26)',
    fontsize=15, fontweight='bold', y=1.02
)

colors_sentiment = {
    'Positive': '#2ecc71',
    'Neutral':  '#3498db',
    'Negative': '#e74c3c'
}

sentiment_counts = df['sentiment'].value_counts()

# --- Plot 1: Pie Chart ---
ax1 = fig.add_subplot(1, 3, 1)
wedge_colors = [colors_sentiment[s] for s in sentiment_counts.index]
wedges, texts, autotexts = ax1.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    colors=wedge_colors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.75,
    labeldistance=1.15,
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax1.set_title('Overall Sentiment\nDistribution',
              fontweight='bold', fontsize=12, pad=15)

# --- Plot 2: Sentiment by Platform ---
ax2 = fig.add_subplot(1, 3, 2)
platform_sentiment = pd.crosstab(
    df['platform'], df['sentiment'],
    normalize='index'
) * 100

platform_colors = [
    colors_sentiment.get(c, '#95a5a6')
    for c in platform_sentiment.columns
]
platform_sentiment.plot(
    kind='bar',
    ax=ax2,
    color=platform_colors,
    edgecolor='white',
    width=0.5
)
ax2.set_title('Sentiment by Platform (%)',
              fontweight='bold', fontsize=12, pad=15)
ax2.set_xlabel('Platform', fontsize=10)
ax2.set_ylabel('Percentage (%)', fontsize=10)
ax2.legend(title='Sentiment', fontsize=9,
           title_fontsize=9, loc='upper right')
ax2.tick_params(axis='x', rotation=0, labelsize=10)
ax2.set_ylim(0, 100)

# --- Plot 3: Sentiment by Language ---
ax3 = fig.add_subplot(1, 3, 3)
language_sentiment = pd.crosstab(
    df['language'], df['sentiment'],
    normalize='index'
) * 100

language_colors = [
    colors_sentiment.get(c, '#95a5a6')
    for c in language_sentiment.columns
]
language_sentiment.plot(
    kind='bar',
    ax=ax3,
    color=language_colors,
    edgecolor='white',
    width=0.5
)
ax3.set_title('Sentiment by Language (%)',
              fontweight='bold', fontsize=12, pad=15)
ax3.set_xlabel('Language', fontsize=10)
ax3.set_ylabel('Percentage (%)', fontsize=10)
ax3.legend(title='Sentiment', fontsize=9,
           title_fontsize=9, loc='upper right')
ax3.tick_params(axis='x', rotation=0, labelsize=10)
ax3.set_ylim(0, 100)

plt.tight_layout(pad=3.0)
plt.savefig('../outputs/chart1_sentiment_overview.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved: chart1_sentiment_overview.png")

# ============================================
# CHART 2 - FASHION CATEGORIES (2 charts)
# ============================================

fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle(
    'Fashion Category Analysis\nNepali Female Fashion (Age 18–26)',
    fontsize=15, fontweight='bold'
)

category_counts = df['fashion_category'].value_counts()
category_colors = [
    '#9b59b6', '#e67e22', '#1abc9c',
    '#e74c3c', '#3498db', '#f39c12'
]

# --- Plot 4: Category Bar Chart ---
bars = ax4.bar(
    range(len(category_counts)),
    category_counts.values,
    color=category_colors[:len(category_counts)],
    edgecolor='white',
    width=0.6
)
ax4.set_xticks(range(len(category_counts)))
ax4.set_xticklabels(
    category_counts.index,
    rotation=25, ha='right', fontsize=9
)
ax4.set_title('Fashion Category Distribution',
              fontweight='bold', fontsize=12, pad=15)
ax4.set_ylabel('Number of Posts', fontsize=10)

# Add value labels on bars
for bar, val in zip(bars, category_counts.values):
    ax4.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 3,
        str(val),
        ha='center', va='bottom',
        fontsize=9, fontweight='bold'
    )

# --- Plot 5: Trend Score by Category ---
category_trend = df.groupby(
    'fashion_category')['trend_score'].mean().sort_values()

bars2 = ax5.barh(
    range(len(category_trend)),
    category_trend.values,
    color='#e67e22',
    edgecolor='white',
    height=0.5
)
ax5.set_yticks(range(len(category_trend)))
ax5.set_yticklabels(category_trend.index, fontsize=9)
ax5.set_title('Average Trend Score by Category',
              fontweight='bold', fontsize=12, pad=15)
ax5.set_xlabel('Average Trend Score', fontsize=10)

# Add value labels
for bar, val in zip(bars2, category_trend.values):
    ax5.text(
        val + 0.005,
        bar.get_y() + bar.get_height()/2,
        f'{val:.3f}',
        va='center', fontsize=9, fontweight='bold'
    )

plt.tight_layout(pad=3.0)
plt.savefig('../outputs/chart2_fashion_categories.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved: chart2_fashion_categories.png")

# ============================================
# CHART 3 - VADER SCORE & LANGUAGE (2 charts)
# ============================================

fig3, (ax6, ax7) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(
    'VADER Score Distribution & Language Analysis\n'
    'Nepali Female Fashion (Age 18–26)',
    fontsize=15, fontweight='bold'
)

# --- Plot 6: VADER Histogram ---
ax6.hist(
    df['vader_compound'],
    bins=30,
    color='#3498db',
    edgecolor='white',
    alpha=0.85
)
ax6.axvline(x=0.05,  color='#2ecc71',
            linestyle='--', linewidth=2,
            label='Positive threshold (0.05)')
ax6.axvline(x=-0.05, color='#e74c3c',
            linestyle='--', linewidth=2,
            label='Negative threshold (-0.05)')
ax6.set_title('VADER Compound Score Distribution',
              fontweight='bold', fontsize=12, pad=15)
ax6.set_xlabel('Compound Score', fontsize=10)
ax6.set_ylabel('Number of Posts', fontsize=10)
ax6.legend(fontsize=9)

# --- Plot 7: Language Pie ---
language_counts = df['language'].value_counts()
ax7.pie(
    language_counts.values,
    labels=language_counts.index,
    colors=['#3498db', '#e67e22'],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    labeldistance=1.15,
    textprops={'fontsize': 11}
)
ax7.set_title('Language Distribution',
              fontweight='bold', fontsize=12, pad=15)

plt.tight_layout(pad=3.0)
plt.savefig('../outputs/chart3_vader_language.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 3 saved: chart3_vader_language.png")

print("\n✅ ALL 3 CHARTS SAVED SUCCESSFULLY!")
print("   outputs/chart1_sentiment_overview.png")
print("   outputs/chart2_fashion_categories.png")
print("   outputs/chart3_vader_language.png")