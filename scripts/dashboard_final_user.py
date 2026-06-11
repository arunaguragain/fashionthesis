# ============================================
# NEPAL FASHION TREND PREDICTOR
# User-Facing Live Dashboard
# By: Aruna Guragain
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================
# COMPATIBILITY FIXES
# ============================================

if not hasattr(st, 'cache_data'):
    st.cache_data = st.cache
if not hasattr(st, 'cache_resource'):
    st.cache_resource = st.cache

_st_btn_orig = st.button
def _safe_btn(label, **kw):
    kw.pop('type', None)
    kw.pop('use_container_width', None)
    try:
        return _st_btn_orig(label, **kw)
    except TypeError:
        return _st_btn_orig(label)
st.button = _safe_btn

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Nepal Fashion Trend Predictor",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# DESIGN SYSTEM — Nepal Inspired
# Crimson red from Nepal flag + deep navy + gold
# ============================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
    background: #0a0a0f;
    color: #f0ede8;
    font-family: 'Inter', sans-serif;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 48px 24px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 40px;
}
.hero-flag {
    font-size: 2.4rem;
    letter-spacing: 6px;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 5vw, 3.6rem);
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #dc143c 0%, #f5a623 50%, #dc143c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(240,237,232,0.55);
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Trend Ticker ── */
.ticker-wrap {
    background: linear-gradient(90deg,
        rgba(220,20,60,0.15), rgba(245,166,35,0.1),
        rgba(220,20,60,0.15));
    border: 1px solid rgba(220,20,60,0.3);
    border-radius: 8px;
    padding: 10px 20px;
    margin-bottom: 36px;
    text-align: center;
    font-size: 0.85rem;
    color: rgba(240,237,232,0.7);
    letter-spacing: 1px;
}
.ticker-wrap b { color: #f5a623; }

/* ── Input Card ── */
.input-card {
    background: linear-gradient(145deg,
        rgba(255,255,255,0.04) 0%,
        rgba(220,20,60,0.04) 100%);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 28px;
}
.input-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(240,237,232,0.45);
    margin-bottom: 12px;
}

/* ── Pill Buttons (examples) ── */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}
.pill {
    background: rgba(220,20,60,0.12);
    border: 1px solid rgba(220,20,60,0.35);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.8rem;
    color: rgba(240,237,232,0.8);
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.pill:hover {
    background: rgba(220,20,60,0.28);
    border-color: #dc143c;
    color: #fff;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f0ede8 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: rgba(220,20,60,0.6) !important;
    box-shadow: 0 0 0 3px rgba(220,20,60,0.12) !important;
}
.stTextArea textarea::placeholder { color: rgba(240,237,232,0.3) !important; }

/* ── Predict Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #dc143c, #a50e2b) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e8173f, #dc143c) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(220,20,60,0.4) !important;
}

/* ── Result Cards ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
}
.result-card {
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.result-card.positive {
    background: linear-gradient(145deg,
        rgba(34,197,94,0.12), rgba(16,185,129,0.06));
    border-color: rgba(34,197,94,0.3);
}
.result-card.neutral {
    background: linear-gradient(145deg,
        rgba(59,130,246,0.12), rgba(99,102,241,0.06));
    border-color: rgba(59,130,246,0.3);
}
.result-card.negative {
    background: linear-gradient(145deg,
        rgba(239,68,68,0.12), rgba(220,20,60,0.06));
    border-color: rgba(239,68,68,0.3);
}
.result-card.category {
    background: linear-gradient(145deg,
        rgba(245,166,35,0.1), rgba(220,20,60,0.05));
    border-color: rgba(245,166,35,0.3);
}
.result-card.trend-high {
    background: linear-gradient(145deg,
        rgba(34,197,94,0.15), rgba(16,185,129,0.08));
    border-color: rgba(34,197,94,0.4);
}
.result-card.trend-mid {
    background: linear-gradient(145deg,
        rgba(245,166,35,0.12), rgba(251,191,36,0.06));
    border-color: rgba(245,166,35,0.4);
}
.result-card.trend-low {
    background: linear-gradient(145deg,
        rgba(239,68,68,0.1), rgba(220,20,60,0.05));
    border-color: rgba(239,68,68,0.3);
}
.card-icon { font-size: 2rem; margin-bottom: 8px; }
.card-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.card-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(240,237,232,0.45);
}

/* ── Score Bar ── */
.score-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}
.score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.score-name {
    width: 80px;
    font-size: 0.82rem;
    color: rgba(240,237,232,0.6);
    flex-shrink: 0;
}
.score-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 50px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.6s ease;
}
.score-val {
    width: 48px;
    text-align: right;
    font-size: 0.82rem;
    font-weight: 600;
    color: rgba(240,237,232,0.8);
    flex-shrink: 0;
}

/* ── Trend Rankings ── */
.rank-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px;
    margin-top: 20px;
}
.rank-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #f5a623;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rank-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.rank-item:last-child { border-bottom: none; }
.rank-num {
    font-size: 1.1rem;
    width: 28px;
    flex-shrink: 0;
}
.rank-name {
    flex: 1;
    font-size: 0.92rem;
    font-weight: 500;
}
.rank-bar-wrap {
    width: 120px;
    flex-shrink: 0;
}
.rank-bar-bg {
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 50px;
    overflow: hidden;
}
.rank-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #dc143c, #f5a623);
}
.rank-score {
    width: 52px;
    text-align: right;
    font-size: 0.8rem;
    color: rgba(240,237,232,0.5);
    flex-shrink: 0;
}

/* ── Insight Box ── */
.insight-box {
    background: linear-gradient(135deg,
        rgba(220,20,60,0.08), rgba(245,166,35,0.05));
    border: 1px solid rgba(220,20,60,0.2);
    border-left: 3px solid #dc143c;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-top: 20px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: rgba(240,237,232,0.8);
}
.insight-box b { color: #f5a623; }

/* ── Footer ── */
.footer-bar {
    text-align: center;
    padding: 32px 0 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 48px;
    font-size: 0.78rem;
    color: rgba(240,237,232,0.3);
    line-height: 1.8;
}

/* ── Section label ── */
.section-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #dc143c;
    margin-bottom: 8px;
}
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 20px;
    color: #f0ede8;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent.parent
    path = (base / 'data' / 'cleaned' /
            'combined_with_sentiment.csv')
    df = pd.read_csv(str(path))
    df['text_clean'] = df['text_clean'].fillna('')
    return df

@st.cache_resource
def train_model(_df):
    X = _df['text_clean']
    y = _df['sentiment']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1,2),
        min_df=2, stop_words='english',
        sublinear_tf=True
    )
    X_train_v = tfidf.fit_transform(X_train)
    svm = LinearSVC(C=1.0, max_iter=2000,
                    random_state=42)
    svm.fit(X_train_v, y_train)
    return tfidf, svm

@st.cache_resource
def train_cat_model(_df):
    X = _df['text_clean']
    y = _df['fashion_category']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1,2),
        min_df=2, stop_words='english',
        sublinear_tf=True
    )
    X_v = tfidf.fit_transform(X_train)
    cat = LinearSVC(C=1.0, max_iter=2000,
                    random_state=42)
    cat.fit(X_v, y_train)
    return tfidf, cat

@st.cache_data
def get_trends(_df):
    nepal = _df[_df['source'] == 'Nepal_Primary']
    t = nepal.groupby('fashion_category').agg(
        posts    = ('text_clean','count'),
        pos_ratio= ('sentiment',
                    lambda x: (x=='Positive').sum()/len(x)),
        avg_score= ('trend_score','mean'),
    ).reset_index()
    t['trend_rank'] = (
        t['avg_score']*0.5 +
        t['pos_ratio']*0.3 +
        (t['posts']/t['posts'].max())*0.2
    )
    return t.sort_values(
        'trend_rank', ascending=False
    ).reset_index(drop=True)

# Load everything
with st.spinner("Loading model..."):
    df        = load_data()
    tfidf, svm        = train_model(df)
    tfidf_cat, cat    = train_cat_model(df)
    trend_df          = get_trends(df)
    vader             = SentimentIntensityAnalyzer()

# ============================================
# HERO
# ============================================

st.markdown("""
<div class="hero">
    <div class="hero-flag">🇳🇵</div>
    <div class="hero-title">Nepal Fashion Trend Predictor</div>
    <div class="hero-sub">
        Discover what's trending in Nepali fashion —
        powered by AI & social media analysis
    </div>
</div>
""", unsafe_allow_html=True)

# Trend ticker
top3 = trend_df.head(3)['fashion_category'].tolist()
st.markdown(f"""
<div class="ticker-wrap">
    🔥 Currently Trending in Nepal →
    <b>#{top3[0]}</b> &nbsp;·&nbsp;
    <b>#{top3[1]}</b> &nbsp;·&nbsp;
    <b>#{top3[2]}</b>
    &nbsp;· Based on {len(df[df['source']=='Nepal_Primary']):,}
    Nepal-specific social media posts
</div>
""", unsafe_allow_html=True)

# ============================================
# MAIN LAYOUT
# ============================================

left_col, right_col = st.columns([3, 2], gap="large")

# ============================================
# LEFT — PREDICTOR
# ============================================

with left_col:

    st.markdown("""
    <div class="section-eyebrow">Live Predictor</div>
    <div class="section-heading">
        Analyse any fashion post
    </div>
    """, unsafe_allow_html=True)

    # Example pills
    examples = [
        "Beautiful nepali saree for dashain! ❤️",
        "New kurti collection — DM to order 🇳🇵",
        "Outfit under Rs 2000 for college girls 🎓",
        "Daura suruwal for wedding — love it! ✨",
        "Indo western fusion look for party 💕",
        "साडी र कुर्ता सेट — एकदम राम्रो छ!",
        "Poor quality dress, disappointed 😞",
        "Dhaka fabric collection — handmade Nepal",
    ]

    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    # Show pills as streamlit buttons
    # Use selectbox to avoid nested columns issue
    st.markdown(
        "<div class='input-label'>Try an example</div>",
        unsafe_allow_html=True
    )

    example_labels = ["— Pick an example —"] + [
        ex[:55] + "…" if len(ex) > 55 else ex
        for ex in examples
    ]

    selected = st.selectbox(
        "example_selector",
        options=example_labels,
        index=0,
        # label_visibility="collapsed"
    )

    if selected != "— Pick an example —":
        idx = example_labels.index(selected) - 1
        st.session_state.input_text = examples[idx]

    st.markdown("<br>", unsafe_allow_html=True)

    # Text area
    with st.form("predict_form", clear_on_submit=False):
        st.markdown(
            "<div class='input-label'>"
            "Or type your own fashion post:"
            "</div>",
            unsafe_allow_html=True
        )
        user_input = st.text_area(
            "post_input",
            value=st.session_state.input_text,
            height=130,
            placeholder=(
                "Paste or type any fashion-related "
                "social media post here…\n"
                "Works in English and Nepali! 🇳🇵"
            ),
            # label_visibility="collapsed"
            # if hasattr(st, 'cache_data')
            # else None
        )
        submitted = st.form_submit_button(
            "✦ Analyse this post"
        )

# ============================================
# PREDICTION LOGIC
# ============================================

if submitted and user_input.strip():

    clean = re.sub(
        r'http\S+|www\S+|@\w+|[#]', '', user_input
    ).strip()

    # VADER
    vs    = vader.polarity_scores(user_input)
    score = vs['compound']

    # SVM sentiment
    vec       = tfidf.transform([clean])
    sentiment = svm.predict(vec)[0]

    # Category
    cat_vec  = tfidf_cat.transform([clean])
    category = cat.predict(cat_vec)[0]

    # Trend potential
    if score > 0.4:
        trend_label = "High Trend Potential"
        trend_cls   = "trend-high"
        trend_icon  = "🔥"
    elif score > 0.05:
        trend_label = "Rising Trend"
        trend_cls   = "trend-mid"
        trend_icon  = "📈"
    else:
        trend_label = "Low Trend Signal"
        trend_cls   = "trend-low"
        trend_icon  = "📉"

    # Sentiment style
    if sentiment == 'Positive':
        sent_cls  = 'positive'
        sent_icon = '😊'
        sent_col  = '#22c55e'
    elif sentiment == 'Neutral':
        sent_cls  = 'neutral'
        sent_icon = '😐'
        sent_col  = '#3b82f6'
    else:
        sent_cls  = 'negative'
        sent_icon = '😞'
        sent_col  = '#ef4444'

    # Category rank
    cat_row = trend_df[
        trend_df['fashion_category'] == category
    ]
    cat_rank = int(cat_row.index[0]) + 1 \
               if not cat_row.empty else "—"
    medals   = ['🥇','🥈','🥉','4th','5th','6th']
    cat_medal= medals[cat_rank-1] \
               if isinstance(cat_rank, int) \
               and cat_rank <= 6 else ""

    # ── Results in LEFT col ──
    with left_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-eyebrow'>Results</div>",
            unsafe_allow_html=True
        )

        # 3-card row
        st.markdown(f"""
        <div class="result-grid">
            <div class="result-card {sent_cls}">
                <div class="card-icon">{sent_icon}</div>
                <div class="card-value"
                     style="color:{sent_col};">
                    {sentiment}
                </div>
                <div class="card-label">Sentiment</div>
                <div style="font-size:0.8rem;
                            margin-top:6px;
                            color:rgba(240,237,232,0.5);">
                    Score: {score:.3f}
                </div>
            </div>

            <div class="result-card category">
                <div class="card-icon">👗</div>
                <div class="card-value"
                     style="color:#f5a623;font-size:1.05rem;">
                    {category}
                </div>
                <div class="card-label">Fashion Category</div>
                <div style="font-size:0.8rem;
                            margin-top:6px;
                            color:rgba(240,237,232,0.5);">
                    {cat_medal} Rank #{cat_rank} in Nepal
                </div>
            </div>

            <div class="result-card {trend_cls}">
                <div class="card-icon">{trend_icon}</div>
                <div class="card-value"
                     style="font-size:1rem;">
                    {trend_label}
                </div>
                <div class="card-label">Trend Signal</div>
                <div style="font-size:0.8rem;
                            margin-top:6px;
                            color:rgba(240,237,232,0.5);">
                    Based on sentiment + engagement
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # VADER score bars
        st.markdown(f"""
        <div class="score-section">
            <div style="font-size:0.78rem;
                        letter-spacing:1.5px;
                        text-transform:uppercase;
                        color:rgba(240,237,232,0.4);
                        margin-bottom:16px;">
                Sentiment Breakdown
            </div>

            <div class="score-row">
                <div class="score-name">Positive</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"
                         style="width:{vs['pos']*100:.0f}%;
                                background:#22c55e;">
                    </div>
                </div>
                <div class="score-val">{vs['pos']:.3f}</div>
            </div>

            <div class="score-row">
                <div class="score-name">Neutral</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"
                         style="width:{vs['neu']*100:.0f}%;
                                background:#3b82f6;">
                    </div>
                </div>
                <div class="score-val">{vs['neu']:.3f}</div>
            </div>

            <div class="score-row">
                <div class="score-name">Negative</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"
                         style="width:{vs['neg']*100:.0f}%;
                                background:#ef4444;">
                    </div>
                </div>
                <div class="score-val">{vs['neg']:.3f}</div>
            </div>

            <div class="score-row" style="margin-top:4px;
                 border-top:1px solid rgba(255,255,255,0.06);
                 padding-top:14px;">
                <div class="score-name">Compound</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"
                         style="width:{(score+1)/2*100:.0f}%;
                                background:linear-gradient(
                                90deg,#dc143c,#f5a623);">
                    </div>
                </div>
                <div class="score-val">{score:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Contextual insight
        if not cat_row.empty:
            row     = cat_row.iloc[0]
            pos_pct = row['pos_ratio']*100
            st.markdown(f"""
            <div class="insight-box">
                <b>{category}</b> is currently ranked
                <b>#{cat_rank}</b> in Nepal fashion trends
                with <b>{pos_pct:.1f}%</b> positive sentiment
                across <b>{int(row['posts'])}</b> Nepal-specific
                social media posts. This category shows a trend
                score of <b>{row['trend_rank']:.4f}</b> — higher
                scores indicate stronger trending momentum.
            </div>
            """, unsafe_allow_html=True)

elif submitted and not user_input.strip():
    with left_col:
        st.warning("Please enter a fashion post to analyse.")

# ============================================
# RIGHT — TREND RANKINGS (always visible)
# ============================================

with right_col:

    st.markdown("""
    <div class="section-eyebrow">Nepal Insights</div>
    <div class="section-heading">
        Fashion Trend Rankings
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    nepal_df = df[df['source'] == 'Nepal_Primary']
    pos_pct  = (
        nepal_df['sentiment'] == 'Positive'
    ).mean() * 100

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:rgba(220,20,60,0.08);
                    border:1px solid rgba(220,20,60,0.2);
                    border-radius:12px;padding:16px;
                    text-align:center;margin-bottom:16px;">
            <div style="font-family:'Playfair Display',serif;
                        font-size:1.8rem;font-weight:900;
                        color:#dc143c;">
                {len(nepal_df):,}
            </div>
            <div style="font-size:0.72rem;
                        text-transform:uppercase;
                        letter-spacing:1.5px;
                        color:rgba(240,237,232,0.4);
                        margin-top:4px;">
                Nepal Posts
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:rgba(245,166,35,0.08);
                    border:1px solid rgba(245,166,35,0.2);
                    border-radius:12px;padding:16px;
                    text-align:center;margin-bottom:16px;">
            <div style="font-family:'Playfair Display',serif;
                        font-size:1.8rem;font-weight:900;
                        color:#f5a623;">
                {pos_pct:.0f}%
            </div>
            <div style="font-size:0.72rem;
                        text-transform:uppercase;
                        letter-spacing:1.5px;
                        color:rgba(240,237,232,0.4);
                        margin-top:4px;">
                Positive Sentiment
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Rankings
    medals_full = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣']
    max_score   = trend_df['trend_rank'].max()

    rank_items = ""
    for i, row in trend_df.iterrows():
        pct = row['trend_rank'] / max_score * 100
        rank_items += f"""
        <div class="rank-item">
            <div class="rank-num">{medals_full[i]}</div>
            <div class="rank-name">{row['fashion_category']}</div>
            <div class="rank-bar-wrap">
                <div class="rank-bar-bg">
                    <div class="rank-bar-fill"
                         style="width:{pct:.0f}%;">
                    </div>
                </div>
            </div>
            <div class="rank-score">
                {row['trend_rank']:.4f}
            </div>
        </div>
        """

    st.markdown(f"""
    <div class="rank-section">
        <div class="rank-title">
            🇳🇵 Nepali Females (18–26)
        </div>
        {rank_items}
    </div>
    """, unsafe_allow_html=True)

    # Platform breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow"
         style="margin-top:8px;">Platforms</div>
    """, unsafe_allow_html=True)

    plat = nepal_df['platform'].value_counts()
    for platform, count in plat.items():
        pct  = count / len(nepal_df) * 100
        icon = "📸" if platform == 'Instagram' else "🎵"
        st.markdown(f"""
        <div style="display:flex;align-items:center;
                    gap:12px;margin-bottom:10px;">
            <div style="font-size:1.1rem;">{icon}</div>
            <div style="flex:1;">
                <div style="display:flex;
                            justify-content:space-between;
                            margin-bottom:5px;">
                    <span style="font-size:0.85rem;">
                        {platform}
                    </span>
                    <span style="font-size:0.8rem;
                                 color:rgba(240,237,232,0.5);">
                        {count:,} posts ({pct:.1f}%)
                    </span>
                </div>
                <div style="height:5px;
                            background:rgba(255,255,255,0.07);
                            border-radius:50px;overflow:hidden;">
                    <div style="width:{pct:.0f}%;height:100%;
                                background:linear-gradient(
                                90deg,#dc143c,#f5a623);
                                border-radius:50px;">
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model info
    st.markdown(f"""
    <div style="margin-top:20px;
                background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.06);
                border-radius:12px;padding:16px;">
        <div style="font-size:0.72rem;letter-spacing:1.5px;
                    text-transform:uppercase;
                    color:rgba(240,237,232,0.35);
                    margin-bottom:12px;">
            Model Info
        </div>
        <div style="display:grid;
                    grid-template-columns:1fr 1fr;
                    gap:10px;font-size:0.82rem;">
            <div style="color:rgba(240,237,232,0.5);">
                Best Model</div>
            <div style="color:#f5a623;font-weight:600;">
                LSTM 95.42%</div>
            <div style="color:rgba(240,237,232,0.5);">
                SVM Accuracy</div>
            <div style="color:#f0ede8;">92.51%</div>
            <div style="color:rgba(240,237,232,0.5);">
                Total Dataset</div>
            <div style="color:#f0ede8;">105,443 records</div>
            <div style="color:rgba(240,237,232,0.5);">
                Nepal Specific</div>
            <div style="color:#f0ede8;">2,172 posts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="footer-bar">
    Nepal Fashion Trend Predictor &nbsp;·&nbsp;
    Aruna Guragain &nbsp;·&nbsp;
    Supervisor: Manoj Shrestha<br>
    Data: Instagram + TikTok (Nepal-specific) +
    Secondary Datasets &nbsp;·&nbsp;
    Model: SVM + LSTM &nbsp;·&nbsp; © 2026
</div>
""", unsafe_allow_html=True)