
import pandas as pd
import numpy as np
import re
import os


# STEP 1 - LOAD ALL FILES
print("="*50)
print("STEP 1: LOADING ALL DATA FILES")
print("="*50)

# Instagram Files 
instagram_files = [
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch1).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch2).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch3).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch4).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch5).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch6).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch7).csv',
    '../data/raw/instagram/dataset_instagram-hashtag-scraper(Batch8).csv',
]

# TikTok Files 
tiktok_files = [
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch1).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch2).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch3).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch4).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch5).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch6).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch7).csv',
    '../data/raw/tiktok/dataset_tiktok-hashtag-scraper(Batch8).csv',
]

# Load Instagram
instagram_dfs = []
for f in instagram_files:
    try:
        df = pd.read_csv(f)
        instagram_dfs.append(df)
        print(f" Loaded: {os.path.basename(f)} — {len(df)} posts")
    except Exception as e:
        print(f" Error loading {f}: {e}")

# Load TikTok
tiktok_dfs = []
for f in tiktok_files:
    try:
        df = pd.read_csv(f)
        tiktok_dfs.append(df)
        print(f" Loaded: {os.path.basename(f)} — {len(df)} posts")
    except Exception as e:
        print(f" Error loading {f}: {e}")

# STEP 2 - STANDARDIZE COLUMNS

print("\n" + "="*50)
print("STEP 2: STANDARDIZING COLUMNS")
print("="*50)

# Combine Instagram
instagram_df = pd.concat(instagram_dfs, ignore_index=True)
instagram_df = instagram_df.rename(columns={
    'caption': 'text',
    'likesCount': 'likes',
    'commentsCount': 'comments',
    'timestamp': 'date',
})
instagram_df['platform'] = 'Instagram'
instagram_df['shares'] = 0  # Instagram has no shares
instagram_df['views'] = 0

# Keep only needed columns
instagram_clean = instagram_df[[
    'platform', 'text', 'likes', 'comments',
    'shares', 'views', 'date'
]].copy()

print(f"Instagram posts loaded: {len(instagram_clean)}")

# Combine TikTok
tiktok_df = pd.concat(tiktok_dfs, ignore_index=True)
tiktok_df = tiktok_df.rename(columns={
    'diggCount': 'likes',
    'commentCount': 'comments',
    'shareCount': 'shares',
    'playCount': 'views',
    'createTimeISO': 'date',
})
tiktok_df['platform'] = 'TikTok'

tiktok_clean = tiktok_df[[
    'platform', 'text', 'likes', 'comments',
    'shares', 'views', 'date'
]].copy()

print(f"TikTok posts loaded: {len(tiktok_clean)}")

# STEP 3 - COMBINE ALL PLATFORMS
print("\n" + "="*50)
print("STEP 3: COMBINING ALL PLATFORMS")
print("="*50)

combined_df = pd.concat(
    [instagram_clean, tiktok_clean],
    ignore_index=True
)
print(f"Total combined posts: {len(combined_df)}")

# STEP 4 - CLEAN THE DATA
print("\n" + "="*50)
print("STEP 4: CLEANING DATA")
print("="*50)

# 4a - Remove empty text
before = len(combined_df)
combined_df = combined_df.dropna(subset=['text'])
combined_df = combined_df[combined_df['text'].str.strip() != '']
print(f"Removed empty posts: {before - len(combined_df)}")

# 4b - Remove duplicates
before = len(combined_df)
combined_df = combined_df.drop_duplicates(subset=['text'])
print(f"Removed duplicates: {before - len(combined_df)}")

# 4c - Clean text function
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)             # remove mentions
    text = re.sub(r'\s+', ' ', text)             # remove extra spaces
    text = text.strip()
    return text

combined_df['text_clean'] = combined_df['text'].apply(clean_text)
print("Text cleaned — URLs and mentions removed")

# 4d - Filter fashion relevant posts
fashion_keywords = [
    'fashion', 'style', 'outfit', 'wear', 'dress', 'cloth',
    'saree', 'lehenga', 'kurta', 'ethnic', 'traditional',
    'फेसन', 'लुगा', 'पोशाक', 'ootd', 'look', 'collection',
    'design', 'trending', 'kurti', 'sari', 'dupatta',
    'blouse', 'skirt', 'shop', 'order', 'nepali', 'nepal',
    'dashain', 'tihar', 'teej', 'festival', 'ethnic',
    'indo-western', 'fusion', 'modern', 'casual', 'formal'
]

combined_df['is_fashion'] = combined_df['text_clean'].str.lower().apply(
    lambda x: any(kw in x for kw in fashion_keywords)
)

before = len(combined_df)
fashion_df = combined_df[combined_df['is_fashion']].copy()
print(f"Non-fashion posts removed: {before - len(fashion_df)}")
print(f"Fashion relevant posts kept: {len(fashion_df)}")

# 4e - Fix data types
fashion_df['likes'] = pd.to_numeric(
    fashion_df['likes'], errors='coerce').fillna(0).astype(int)
fashion_df['comments'] = pd.to_numeric(
    fashion_df['comments'], errors='coerce').fillna(0).astype(int)
fashion_df['shares'] = pd.to_numeric(
    fashion_df['shares'], errors='coerce').fillna(0).astype(int)
fashion_df['views'] = pd.to_numeric(
    fashion_df['views'], errors='coerce').fillna(0).astype(int)
fashion_df['date'] = pd.to_datetime(
    fashion_df['date'], errors='coerce')
print(" Data types fixed")

# 4f - Add language detection
def detect_language(text):
    if any('\u0900' <= c <= '\u097F' for c in str(text)):
        return 'Nepali'
    return 'English'

fashion_df['language'] = fashion_df['text_clean'].apply(detect_language)
print("Language detected")

# STEP 5 - SAVE CLEANED DATA
print("\n" + "="*50)
print("STEP 5: SAVING CLEANED DATA")
print("="*50)

os.makedirs('../data/cleaned', exist_ok=True)

fashion_df.to_csv(
    '../data/cleaned/fashion_data_cleaned.csv',
    index=False
)
print(f" Saved to: data/cleaned/fashion_data_cleaned.csv")

# STEP 6 - FINAL SUMMARY
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Total posts after cleaning : {len(fashion_df)}")
print(f"\nBy Platform:")
print(fashion_df['platform'].value_counts().to_string())
print(f"\nBy Language:")
print(fashion_df['language'].value_counts().to_string())
print(f"\nDate Range:")
print(f"  From : {fashion_df['date'].min()}")
print(f"  To   : {fashion_df['date'].max()}")
print(f"\nEngagement Stats:")
print(f"  Avg Likes    : {fashion_df['likes'].mean():.1f}")
print(f"  Avg Comments : {fashion_df['comments'].mean():.1f}")
print(f"  Avg Shares   : {fashion_df['shares'].mean():.1f}")
print(f"  Avg Views    : {fashion_df['views'].mean():.1f}")
print("\n DATA CLEANING COMPLETE!")
print("="*50)