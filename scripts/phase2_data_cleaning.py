# PHASE 2 - COMBINED DATA CLEANING SCRIPT
# Combines Phase 1 + Phase 2 data

import pandas as pd
import numpy as np
import re
import os
import glob

print("="*55)
print("PHASE 2 - COMBINED DATA CLEANING")
print("="*55)

# STEP 1 - DEFINE ALL FILES

print("\nSTEP 1: LOADING ALL FILES")
print("-"*55)

#  Phase 1 Instagram 
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

#  Phase 1 TikTok 
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

#  Phase 2 Instagram (new files) 
p2_ig_files = sorted(glob.glob(
    '../data/raw/instagram_phase2/dataset_instagram*.csv'
))

#  Phase 2 TikTok (new files) 
p2_tk_files = sorted(glob.glob(
    '../data/raw/tiktok_phase2/dataset_tiktok*.csv'
))

# STEP 2 - LOAD FUNCTIONS
def load_instagram(files, phase_label):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['platform'] = 'Instagram'
            df['phase']    = phase_label
            df = df.rename(columns={
                'caption'      : 'text',
                'likesCount'   : 'likes',
                'commentsCount': 'comments',
                'timestamp'    : 'date'
            })
            df['shares'] = 0
            df['views']  = 0
            keep = ['platform','phase','text',
                    'likes','comments','shares',
                    'views','date']
            dfs.append(df[[c for c in keep
                           if c in df.columns]])
            print(f" {os.path.basename(f)}: "
                  f"{len(df)} posts")
        except Exception as e:
            print(f"{os.path.basename(f)}: {e}")
    return pd.concat(dfs, ignore_index=True) \
           if dfs else pd.DataFrame()

def load_tiktok(files, phase_label):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['platform'] = 'TikTok'
            df['phase']    = phase_label
            df = df.rename(columns={
                'diggCount'   : 'likes',
                'commentCount': 'comments',
                'shareCount'  : 'shares',
                'playCount'   : 'views',
                'createTimeISO': 'date'
            })
            keep = ['platform','phase','text',
                    'likes','comments','shares',
                    'views','date']
            dfs.append(df[[c for c in keep
                           if c in df.columns]])
            print(f" {os.path.basename(f)}: "
                  f"{len(df)} posts")
        except Exception as e:
            print(f" {os.path.basename(f)}: {e}")
    return pd.concat(dfs, ignore_index=True) \
           if dfs else pd.DataFrame()

# Load all
print("\n Phase 1 Instagram:")
p1_ig = load_instagram(p1_ig_files, 'Phase1')
print("\n Phase 1 TikTok:")
p1_tk = load_tiktok(p1_tk_files,    'Phase1')
print("\n Phase 2 Instagram:")
p2_ig = load_instagram(p2_ig_files, 'Phase2')
print("\n Phase 2 TikTok:")
p2_tk = load_tiktok(p2_tk_files,    'Phase2')


# STEP 3 - COMBINE ALL

print(f"\nSTEP 3: COMBINING ALL DATA")
print("-"*55)

combined = pd.concat(
    [p1_ig, p1_tk, p2_ig, p2_tk],
    ignore_index=True
)

print(f"Phase 1 Instagram : {len(p1_ig)} posts")
print(f"Phase 1 TikTok    : {len(p1_tk)} posts")
print(f"Phase 2 Instagram : {len(p2_ig)} posts")
print(f"Phase 2 TikTok    : {len(p2_tk)} posts")
print(f"Total combined    : {len(combined)} posts")

# STEP 4 - CLEAN DATA

print(f"\nSTEP 4: CLEANING DATA")
print("-"*55)

# Remove empty
before = len(combined)
combined = combined.dropna(subset=['text'])
combined = combined[combined['text'].str.strip() != '']
print(f"Removed empty posts: {before - len(combined)}")

# Remove duplicates
before = len(combined)
combined = combined.drop_duplicates(subset=['text'])
print(f"Removed duplicates : {before - len(combined)}")

# Clean text function
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

combined['text_clean'] = combined['text'].apply(clean_text)
print(f" Text cleaned")

# STEP 5 - FASHION FILTER
print(f"\nSTEP 5: FILTERING FASHION POSTS")
print("-"*55)

fashion_keywords = [
    # English fashion terms
    'fashion','style','outfit','wear','dress','cloth',
    'saree','lehenga','kurta','ethnic','traditional',
    'ootd','look','collection','design','trending',
    'shop','order','kurti','sari','dupatta','blouse',
    'skirt','nepal','dashain','tihar','teej','festival',
    'modern','casual','formal','grwm','fits','bride',
    'wedding','party','cultural','model','miss','pageant',
    'tailored','festive','pahiran','cholo','dhaka','gunyo',
    'daura','suruwal','thamel','newari','rai','tamang',
    'gurung','limbu','tharu','kurticollection','fashionista',
    'streetstyle','lookoftheday','ootd','instafashion',
    'fashionblogger','styleinspo','outfitinspo','clothingstore',
    'handloom','handmade','localfashion','nepalicraft',
    # Nepali terms (Devanagari)
    'फेसन','लुगा','पोशाक','साडी','कुर्ता','फ्याशन',
    'स्टाइल','नारी','तिहार','दशैं','पार्टी','ढाका',
    'पहिरन','क्लोथ','सारी','लेहेंगा'
]

combined['text_lower'] = combined['text_clean'].str.lower()
combined['is_fashion'] = combined['text_lower'].apply(
    lambda x: any(kw.lower() in x
                  for kw in fashion_keywords)
)

before = len(combined)
fashion_df = combined[combined['is_fashion']].copy()
print(f"Non-fashion removed: {before - len(fashion_df)}")
print(f" Fashion posts kept: {len(fashion_df)}")

# STEP 6 - FEATURE ENGINEERING

print(f"\nSTEP 6: FEATURE ENGINEERING")
print("-"*55)

# Fix numeric types
for col in ['likes','comments','shares','views']:
    fashion_df[col] = pd.to_numeric(
        fashion_df[col], errors='coerce'
    ).fillna(0).astype(int)

# Date parsing
fashion_df['date'] = pd.to_datetime(
    fashion_df['date'], errors='coerce', utc=True
)

# Language detection
def detect_lang(text):
    if any('\u0900' <= c <= '\u097F'
           for c in str(text)):
        return 'Nepali'
    return 'English'

fashion_df['language'] = fashion_df['text_clean'].apply(
    detect_lang
)

print(f" Numeric types fixed")
print(f" Language detected")

# Fashion category classification
def classify_category(text):
    t = str(text).lower()
    if any(k in t for k in [
        'saree','sari','lehenga','kurta','kurti',
        'ethnic','traditional','daura','gunyo','cholo',
        'dhaka','dashain','tihar','teej','festival',
        'cultural','pahiran','newari','ढाका','साडी',
        'पोशाक','दशैं','तिहार','पहिरन','फ्याशन',
        'bride','wedding','handloom','handmade'
    ]):
        return 'Traditional/Ethnic'
    elif any(k in t for k in [
        'jeans','top','tshirt','t-shirt','casual',
        'western','hoodie','jacket','sneaker','denim',
        'shorts','skirt','crop','streetwear','urban',
        'grwm','streetstyle','fits','gym','sporty'
    ]):
        return 'Western/Casual'
    elif any(k in t for k in [
        'indo western','indo-western','fusion',
        'modern','contemporary','blend','mix',
        'semi formal','bollywood'
    ]):
        return 'Indo-Western/Fusion'
    elif any(k in t for k in [
        'formal','office','professional','business',
        'suit','blazer','workwear','corporate'
    ]):
        return 'Formal/Professional'
    elif any(k in t for k in [
        'handbag','bag','jewel','jewelry','jewellery',
        'necklace','earring','bracelet','shoes','heel',
        'sandal','accessories','watch','scarf','dupatta'
    ]):
        return 'Accessories'
    return 'General Fashion'

fashion_df['fashion_category'] = (
    fashion_df['text_clean'].apply(classify_category)
)
print(f"Fashion categories assigned")

# Engagement score
fashion_df['engagement_score'] = (
    fashion_df['likes'].clip(upper=10000)   / 10000 * 0.4 +
    fashion_df['comments'].clip(upper=1000) / 1000  * 0.3 +
    fashion_df['shares'].clip(upper=1000)   / 1000  * 0.2 +
    fashion_df['views'].clip(upper=1000000) / 1000000 * 0.1
)

# STEP 7 - SAVE FILES

print(f"\nSTEP 7: SAVING FILES")
print("-"*55)

os.makedirs('../data/cleaned', exist_ok=True)

# Save combined cleaned data
fashion_df.to_csv(
    '../data/cleaned/phase2_combined_clean.csv',
    index=False
)
print(f" Saved: phase2_combined_clean.csv")

# Save Phase 1 only (for comparison)
p1_only = fashion_df[
    fashion_df['phase'] == 'Phase1'
].copy()
p1_only.to_csv(
    '../data/cleaned/phase1_only_clean.csv',
    index=False
)
print(f" Saved: phase1_only_clean.csv")

# Save Phase 2 only
p2_only = fashion_df[
    fashion_df['phase'] == 'Phase2'
].copy()
p2_only.to_csv(
    '../data/cleaned/phase2_only_clean.csv',
    index=False
)
print(f" Saved: phase2_only_clean.csv")

# STEP 8 - FINAL SUMMARY


print(f"\n{'='*55}")
print(f"CLEANING COMPLETE — FINAL SUMMARY")
print(f"{'='*55}")
print(f"\nTotal posts (combined) : {len(fashion_df)}")
print(f"\nBy Phase:")
for ph, cnt in fashion_df['phase'].value_counts().items():
    print(f"  {ph}: {cnt} posts")
print(f"\nBy Platform:")
for pl, cnt in fashion_df['platform'].value_counts().items():
    pct = cnt/len(fashion_df)*100
    print(f"  {pl}: {cnt} ({pct:.1f}%)")
print(f"\nBy Language:")
for lg, cnt in fashion_df['language'].value_counts().items():
    pct = cnt/len(fashion_df)*100
    print(f"  {lg}: {cnt} ({pct:.1f}%)")
print(f"\nBy Fashion Category:")
for cat, cnt in fashion_df[
    'fashion_category'
].value_counts().items():
    pct = cnt/len(fashion_df)*100
    print(f"  {cat}: {cnt} ({pct:.1f}%)")
print(f"\nDate Range:")
print(f"  From: {fashion_df['date'].min()}")
print(f"  To  : {fashion_df['date'].max()}")
print(f"\n Ready for Phase 2 Sentiment Analysis!")
print(f"{'='*55}")