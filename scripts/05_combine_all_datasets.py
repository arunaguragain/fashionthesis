# COMBINED DATASET BUILDER

import pandas as pd
import numpy as np
import re
import os
import glob
import warnings
warnings.filterwarnings('ignore') 

print("="*60)
print("COMBINED DATASET BUILDER")
print("Nepal Primary + Secondary Datasets")
print("="*60)

# STEP 1 — LOAD NEPAL PRIMARY DATA

print("\n STEP 1: Loading Nepal Primary Data")
print("-"*60)

# Phase 1 Instagram files
p1_ig_files = [
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch1).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch2).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch3).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch4).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch5).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch6).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch7).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch8).csv',
]

# Phase 1 TikTok files
p1_tk_files = [
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch1).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch2).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch3).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch4).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch5).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch6).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch7).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch8).csv',
]

# Phase 2 files (all CSVs in phase2 folder)
p2_ig_files = sorted(glob.glob('../data/raw/instagram_phase2/dataset_instagram*.csv'))
p2_tk_files = sorted(glob.glob('../data/raw/tiktok_phase2/dataset_tiktok*.csv'))

def load_instagram(files, phase):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['platform'] = 'Instagram'
            df['phase']    = phase
            df['source']   = 'Nepal_Primary'
            df = df.rename(columns={
                'caption':'text',
                'likesCount':'likes',
                'commentsCount':'comments',
                'timestamp':'date'
            })
            df['shares'] = 0
            df['views']  = 0
            keep = ['platform','phase','source',
                    'text','likes','comments',
                    'shares','views','date']
            dfs.append(df[[c for c in keep
                           if c in df.columns]])
            print(f" {os.path.basename(f)}: {len(df)} posts")
        except Exception as e:
            print(f" {os.path.basename(f)}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs \
           else pd.DataFrame()

def load_tiktok(files, phase):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['platform'] = 'TikTok'
            df['phase']    = phase
            df['source']   = 'Nepal_Primary'
            df = df.rename(columns={
                'diggCount':'likes',
                'commentCount':'comments',
                'shareCount':'shares',
                'playCount':'views',
                'createTimeISO':'date'
            })
            keep = ['platform','phase','source',
                    'text','likes','comments',
                    'shares','views','date']
            dfs.append(df[[c for c in keep
                           if c in df.columns]])
            print(f" {os.path.basename(f)}: {len(df)} posts")
        except Exception as e:
            print(f" {os.path.basename(f)}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs \
           else pd.DataFrame()

print("\n Phase 1 Instagram:")
p1_ig = load_instagram(p1_ig_files, 'Phase1')
print("\n Phase 1 TikTok:")
p1_tk = load_tiktok(p1_tk_files, 'Phase1')
print("\n Phase 2 Instagram:")
p2_ig = load_instagram(p2_ig_files, 'Phase2')
print("\n Phase 2 TikTok:")
p2_tk = load_tiktok(p2_tk_files, 'Phase2')

nepal_df = pd.concat(
    [p1_ig, p1_tk, p2_ig, p2_tk],
    ignore_index=True
)
nepal_df = nepal_df.dropna(subset=['text'])
nepal_df = nepal_df.drop_duplicates(subset=['text'])

print(f"\n Total Nepal posts (clean): {len(nepal_df)}")

# STEP 2 — LOAD WOMEN'S CLOTHING REVIEWS

print("\n STEP 2: Loading Women's Clothing Reviews")
print("-"*60)

try:
    wc = pd.read_csv(
    "D:/fashionthesis/data/raw/secondary/Womens Clothing E-Commerce Reviews.csv")
    print(f"  Loaded: {len(wc)} rows")
    print(f"  Columns: {list(wc.columns)}")

    # Convert rating to sentiment-like text
    def rating_to_text(row):
        title   = str(row.get('Title', ''))
        review  = str(row.get('Review Text', ''))
        dept    = str(row.get('Department Name', ''))
        cls     = str(row.get('Class Name', ''))
        rating  = row.get('Rating', 3)
        # Build descriptive text
        text = f"{title}. {review} {dept} {cls}".strip()
        return text

    wc['text'] = wc.apply(rating_to_text, axis=1)

    # Convert rating to sentiment
    def rating_to_sentiment(rating):
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'

    wc['pre_sentiment'] = wc['Rating'].apply(
        rating_to_sentiment
    )

    # Standardize columns
    wc['platform'] = 'Ecommerce_Review'
    wc['phase']    = 'Secondary'
    wc['source']   = 'Womens_Clothing_Reviews'
    wc['likes']    = wc.get(
        'Positive Feedback Count', 0
    ).fillna(0).astype(int)
    wc['comments'] = 0
    wc['shares']   = 0
    wc['views']    = 0
    wc['date']     = pd.NaT
    wc['age']      = wc.get('Age', np.nan)

    wc_clean = wc[[
        'platform','phase','source','text',
        'likes','comments','shares','views',
        'date','pre_sentiment','age'
    ]].copy()

    # Filter for 18-26 age group (thesis specific!)
    wc_1826 = wc_clean[
        (wc_clean['age'] >= 18) &
        (wc_clean['age'] <= 26)
    ].copy()

    # Remove empty reviews
    wc_clean = wc_clean.dropna(subset=['text'])
    wc_clean = wc_clean[
        wc_clean['text'].str.len() > 10
    ]

    print(f" Total rows (all ages): {len(wc_clean)}")
    print(f" Rows aged 18-26: {len(wc_1826)}")
    print(f"  Sentiment distribution:")
    print(wc_clean['pre_sentiment'].value_counts().to_string())

except Exception as e:
    print(f"  Error: {e}")
    wc_clean = pd.DataFrame()

# STEP 3 — LOAD AMAZON APPAREL REVIEWS

print("\n STEP 3: Loading Amazon Apparel Reviews")
print("-"*60)

amazon_path = (
    '../data/raw/secondary/'
    'amazon_reviews_us_Apparel_v1_00.tsv'
)

try:
    print("  Loading TSV file (large file — may take 1-2 mins)...")

    # Load in chunks to handle large file
    chunks = []
    chunk_size = 10000
    total_loaded = 0
    target = 100000  # We only need 100k rows

    for chunk in pd.read_csv(
        amazon_path,
        sep='\t',
        on_bad_lines='skip',
        chunksize=chunk_size,
        low_memory=False
    ):
        chunks.append(chunk)
        total_loaded += len(chunk)
        if total_loaded >= target:
            break

    amazon_raw = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded: {len(amazon_raw)} rows")
    print(f"  Columns: {list(amazon_raw.columns[:8])}")

    # Filter apparel only (already apparel file but double check)
    if 'product_category' in amazon_raw.columns:
        amazon_raw = amazon_raw[
            amazon_raw['product_category'] == 'Apparel'
        ]
        print(f"  After Apparel filter: {len(amazon_raw)} rows")

    # Build text column
    def build_amazon_text(row):
        title  = str(row.get('review_headline', ''))
        body   = str(row.get('review_body', ''))
        product = str(row.get('product_title', ''))
        return f"{title}. {body}".strip()

    amazon_raw['text'] = amazon_raw.apply(
        build_amazon_text, axis=1
    )

    # Convert star_rating to sentiment
    def star_to_sentiment(star):
        try:
            star = int(star)
            if star >= 4:
                return 'Positive'
            elif star == 3:
                return 'Neutral'
            else:
                return 'Negative'
        except:
            return 'Neutral'

    amazon_raw['pre_sentiment'] = amazon_raw[
        'star_rating'
    ].apply(star_to_sentiment)

    # Standardize
    amazon_raw['platform'] = 'Ecommerce_Review'
    amazon_raw['phase']    = 'Secondary'
    amazon_raw['source']   = 'Amazon_Apparel'
    amazon_raw['likes']    = amazon_raw.get(
        'helpful_votes', 0
    ).fillna(0)
    amazon_raw['comments'] = 0
    amazon_raw['shares']   = 0
    amazon_raw['views']    = 0
    amazon_raw['date']     = pd.to_datetime(
        amazon_raw.get('review_date', None),
        errors='coerce'
    )
    amazon_raw['age']      = np.nan

    amazon_clean = amazon_raw[[
        'platform','phase','source','text',
        'likes','comments','shares','views',
        'date','pre_sentiment','age'
    ]].copy()

    # Remove empty
    amazon_clean = amazon_clean.dropna(subset=['text'])
    amazon_clean = amazon_clean[
        amazon_clean['text'].str.len() > 10
    ]
    amazon_clean = amazon_clean.drop_duplicates(
        subset=['text']
    )

    # Sample 80,000 to keep manageable
    if len(amazon_clean) > 80000:
        amazon_clean = amazon_clean.sample(
            80000, random_state=42
        )
        print(f"  Sampled to: 80,000 rows")

    print(f"  Final Amazon rows: {len(amazon_clean)}")
    print(f"  Sentiment distribution:")
    print(amazon_clean['pre_sentiment'].value_counts().to_string())

except Exception as e:
    print(f"   Error loading Amazon TSV: {e}")
    amazon_clean = pd.DataFrame()

# STEP 4 — COMBINE ALL DATASETS

print("\n STEP 4: Combining All Datasets")
print("-"*60)

# Add missing columns to Nepal data
nepal_df['pre_sentiment'] = np.nan
nepal_df['age']           = np.nan

# Combine
all_datasets = []
if len(nepal_df) > 0:
    all_datasets.append(nepal_df)
if len(wc_clean) > 0:
    all_datasets.append(wc_clean)
if len(amazon_clean) > 0:
    all_datasets.append(amazon_clean)

combined = pd.concat(all_datasets, ignore_index=True)

print(f"\n  Combined total rows: {len(combined)}")
print(f"\n  By Source:")
for src, cnt in combined['source'].value_counts().items():
    pct = cnt/len(combined)*100
    print(f"    {src}: {cnt:,} ({pct:.1f}%)")

# STEP 5 — CLEAN COMBINED TEXT
print("\n STEP 5: Cleaning Combined Text")
print("-"*60)

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

combined['text_clean'] = combined['text'].apply(clean_text)

# Remove too short texts
before = len(combined)
combined = combined[combined['text_clean'].str.len() > 15]
print(f"  Removed too-short texts: {before - len(combined)}")

# Language detection
def detect_lang(text):
    if any('\u0900' <= c <= '\u097F'
           for c in str(text)):
        return 'Nepali'
    return 'English'

combined['language'] = combined['text_clean'].apply(
    detect_lang
)
print(f" Language detected")

# Fashion category classification
def classify_fashion(text):
    t = str(text).lower()
    if any(k in t for k in [
        'saree','sari','lehenga','kurta','kurti',
        'ethnic','traditional','daura','dashain',
        'tihar','teej','festival','cultural',
        'gunyo','cholo','dhaka','pahiran','bride',
        'wedding','handloom','पोशाक','साडी'
    ]):
        return 'Traditional/Ethnic'
    elif any(k in t for k in [
        'jeans','casual','western','hoodie',
        'jacket','sneaker','denim','shorts',
        'skirt','crop','streetwear','grwm',
        'tshirt','t-shirt','blouse','top',
        'sweater','knit','fine gauge'
    ]):
        return 'Western/Casual'
    elif any(k in t for k in [
        'indo western','indo-western','fusion',
        'modern','contemporary','blend'
    ]):
        return 'Indo-Western/Fusion'
    elif any(k in t for k in [
        'formal','office','professional',
        'suit','blazer','workwear','corporate',
        'dress','gown','evening'
    ]):
        return 'Formal/Professional'
    elif any(k in t for k in [
        'bag','jewel','jewelry','necklace',
        'earring','shoes','heel','sandal',
        'accessories','watch','intimate',
        'lingerie','bra','underwear'
    ]):
        return 'Accessories'
    return 'General Fashion'

combined['fashion_category'] = combined[
    'text_clean'
].apply(classify_fashion)

print(f"  Fashion categories assigned")

# Fix numeric columns
for col in ['likes','comments','shares','views']:
    if col in combined.columns:
        combined[col] = pd.to_numeric(
            combined[col], errors='coerce'
        ).fillna(0).astype(int)

# STEP 6 — SAVE ALL VERSIONS
print("\n STEP 6: Saving Files")
print("-"*60)

os.makedirs('../data/cleaned', exist_ok=True)

# 1. Save complete combined dataset
combined.to_csv(
    '../data/cleaned/combined_all_datasets.csv',
    index=False
)
print(f"  combined_all_datasets.csv — {len(combined):,} rows")

# 2. Save Nepal primary only (for findings)
nepal_primary = combined[
    combined['source'] == 'Nepal_Primary'
].copy()
nepal_primary.to_csv(
    '../data/cleaned/nepal_primary_only.csv',
    index=False
)
print(f"   nepal_primary_only.csv — {len(nepal_primary):,} rows")

# 3. Save secondary only (for model training context)
secondary = combined[
    combined['source'] != 'Nepal_Primary'
].copy()
secondary.to_csv(
    '../data/cleaned/secondary_datasets.csv',
    index=False
)
print(f"   secondary_datasets.csv — {len(secondary):,} rows")

# 4. Save 18-26 age filtered secondary
if 'age' in combined.columns:
    age_filtered = combined[
        (combined['age'] >= 18) &
        (combined['age'] <= 26)
    ].copy()
    age_filtered.to_csv(
        '../data/cleaned/secondary_age_18_26.csv',
        index=False
    )
    print(f"  secondary_age_18_26.csv — "
          f"{len(age_filtered):,} rows")


# FINAL SUMMARY
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"\nTotal combined dataset : {len(combined):,} rows")
print(f"\nBy Source:")
for src, cnt in combined['source'].value_counts().items():
    pct = cnt/len(combined)*100
    print(f"  {src:<30}: {cnt:>8,} ({pct:.1f}%)")

print(f"\nBy Phase:")
for ph, cnt in combined['phase'].value_counts().items():
    print(f"  {ph:<15}: {cnt:>8,}")

print(f"\nBy Language:")
for lg, cnt in combined['language'].value_counts().items():
    pct = cnt/len(combined)*100
    print(f"  {lg:<10}: {cnt:>8,} ({pct:.1f}%)")

print(f"\nBy Fashion Category:")
for cat, cnt in combined['fashion_category'].value_counts().items():
    pct = cnt/len(combined)*100
    print(f"  {cat:<25}: {cnt:>8,} ({pct:.1f}%)")

print(f"\n{'='*60}")
print(f" Nepal findings from: nepal_primary_only.csv")
print(f" ML training from: combined_all_datasets.csv")
print(f"{'='*60}")