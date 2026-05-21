# ============================================
# FASHION TREND PREDICTION DASHBOARD
# FIXED VERSION v2.0
# By: Aruna Guragain
# Supervisor: Manoj Shrestha
# Fixes: Pie chart data, trend chart text,
#        live predictor button error
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STREAMLIT VERSION COMPATIBILITY FIXES
# ============================================

# FIX 1: Cache compatibility for Streamlit < 1.18
if not hasattr(st, 'cache_data'):
    st.cache_data = st.cache
if not hasattr(st, 'cache_resource'):
    st.cache_resource = st.cache

# FIX 2: st.dataframe compatibility wrapper
# Removes unsupported kwargs on older Streamlit
_st_dataframe_orig = st.dataframe
def _safe_dataframe(df, **kwargs):
    try:
        return _st_dataframe_orig(df, **kwargs)
    except TypeError:
        return _st_dataframe_orig(df)
st.dataframe = _safe_dataframe

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Nepal Fashion Trend Predictor",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e91e8c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e91e8c;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0a0c0;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e91e8c;
        margin-bottom: 20px;
    }
    .trend-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #e91e8c;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 8px 0;
    }
    .result-positive {
        background: linear-gradient(135deg, #1a3a2a, #0d2d1a);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-neutral {
        background: linear-gradient(135deg, #1a2a3a, #0d1d2d);
        border: 2px solid #3498db;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #3a1a1a, #2d0d0d);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #606080;
        font-size: 0.8rem;
        padding: 20px 0;
        border-top: 1px solid #2a2a4a;
        margin-top: 40px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING — FIX 3: Correct absolute path
# ============================================

@st.cache_data
def load_data():
    """
    FIX 3: Use Path(__file__) to build absolute
    path — prevents 'file not found' when
    Streamlit runs from different working directory
    """
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = (base_dir / 'data' / 'cleaned' /
                'fashion_data_with_sentiment.csv')
    df = pd.read_csv(str(csv_path))
    df['text_clean'] = df['text_clean'].fillna('')
    df['date'] = pd.to_datetime(
        df['date'], errors='coerce', utc=True
    )
    return df

# ============================================
# MODEL TRAINING
# ============================================

@st.cache_resource
def train_models(_df):
    """
    FIX 4: Added underscore prefix to df parameter
    (_df) to prevent Streamlit from trying to hash
    the DataFrame — fixes cache_resource error
    with DataFrame arguments
    """
    X      = _df['text_clean']
    y_sent = _df['sentiment']
    y_cat  = _df['fashion_category']

    X_train, X_test, y_train_s, y_test_s = train_test_split(
        X, y_sent, test_size=0.2,
        random_state=42, stratify=y_sent
    )
    _, _, y_train_c, _ = train_test_split(
        X, y_cat, test_size=0.2,
        random_state=42, stratify=y_cat
    )

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=2, stop_words='english',
        sublinear_tf=True
    )
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v  = tfidf.transform(X_test)

    svm = LinearSVC(C=1.0, max_iter=2000,
                    random_state=42)
    svm.fit(X_train_v, y_train_s)

    cat_model = LinearSVC(C=1.0, max_iter=2000,
                          random_state=42)
    cat_model.fit(X_train_v, y_train_c)

    acc = svm.score(X_test_v, y_test_s)
    return tfidf, svm, cat_model, acc

# ============================================
# TREND ANALYSIS
# ============================================

@st.cache_data
def get_trend_analysis(_df):
    trend = _df.groupby('fashion_category').agg(
        total_posts    =('text_clean', 'count'),
        avg_sentiment  =('vader_compound', 'mean'),
        positive_ratio =('sentiment',
                         lambda x: (x == 'Positive').sum()
                         / len(x)),
        avg_likes      =('likes', 'mean'),
        avg_comments   =('comments', 'mean'),
        avg_shares     =('shares', 'mean'),
        avg_views      =('views', 'mean'),
        avg_trend_score=('trend_score', 'mean'),
    ).reset_index()

    trend['trend_rank'] = (
        trend['avg_trend_score'] * 0.5 +
        trend['positive_ratio']  * 0.3 +
        (trend['total_posts'] /
         trend['total_posts'].max()) * 0.2
    )
    return trend.sort_values(
        'trend_rank', ascending=False
    ).reset_index(drop=True)

# ============================================
# LOAD EVERYTHING
# ============================================

df       = load_data()
tfidf, svm_model, cat_model, svm_acc = train_models(df)
trend_df = get_trend_analysis(df)
analyzer = SentimentIntensityAnalyzer()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.markdown("""
<div style='text-align:center; padding:20px 0;'>
    <div style='font-size:2.5rem;'>👗</div>
    <div style='font-size:1.1rem; font-weight:700;
                color:#e91e8c;'>Nepal Fashion</div>
    <div style='font-size:0.85rem; color:#a0a0c0;'>
        Trend Predictor</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Navigate",
    ["🏠 Home", "📊 Data Overview",
     "💬 Sentiment Analysis",
     "🔥 Trend Prediction", "🤖 Live Predictor"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:0.8rem; color:#606080;'>
    <b>Submitted by:</b> Aruna Guragain<br>
    <b>Supervisor:</b> Manoj Shrestha<br>
    <b>Dataset:</b> {len(df)} posts<br>
    <b>Platforms:</b> Instagram + TikTok<br>
    <b>Best Model:</b> SVM ({svm_acc*100:.2f}%)
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE 1 — HOME
# ============================================

if page == "🏠 Home":

    st.markdown("""
    <h1 style='text-align:center; color:#e91e8c;
               font-size:2.2rem; font-weight:800;'>
        Fashion Trend Prediction System
    </h1>
    <p style='text-align:center; color:#a0a0c0;
              font-size:1.0rem; margin-bottom:30px;'>
        For Nepali Females Aged 18–26 |
        Sentiment Analysis & Machine Learning
    </p>
    """, unsafe_allow_html=True)

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-value'>{len(df)}</p>
            <p class='metric-label'>Posts Analyzed</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-value'>{svm_acc*100:.1f}%</p>
            <p class='metric-label'>SVM Accuracy</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        pos_pct = (df['sentiment'] == 'Positive').mean() * 100
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-value'>{pos_pct:.1f}%</p>
            <p class='metric-label'>Positive Sentiment</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        top_trend = trend_df.iloc[0]['fashion_category']
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-value'>#1</p>
            <p class='metric-label'>{top_trend}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🎯 Project Aim")
        st.info(
            "Design and develop a fashion trend prediction "
            "system using social media sentiment analysis "
            "and machine learning to forecast fashion "
            "trends for Nepali females aged 18–26."
        )
        st.markdown("### 📱 Data Sources")
        # FIX 5: Use real counts from loaded data
        platform_counts = df['platform'].value_counts()
        source_df = pd.DataFrame({
            'Platform':   platform_counts.index.tolist(),
            'Posts':      platform_counts.values.tolist(),
            'Percentage': [
                f"{v/len(df)*100:.1f}%"
                for v in platform_counts.values
            ]
        })
        st.dataframe(source_df, use_container_width=True)

    with col_r:
        st.markdown("### 🔄 System Pipeline")
        steps = [
            "1️⃣  Data Collection (Apify Scraper)",
            "2️⃣  Data Cleaning & Preprocessing",
            "3️⃣  NLP & TF-IDF Vectorization",
            "4️⃣  VADER Sentiment Analysis",
            "5️⃣  ML Model Training (NB + SVM)",
            "6️⃣  Fashion Trend Ranking",
            "7️⃣  Live Prediction Dashboard",
        ]
        for s in steps:
            st.success(s)

    st.markdown("---")
    st.markdown("### 🏆 Top Fashion Trends — Nepali Female (18–26)")
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣']
    cols   = st.columns(3)
    for i, (_, row) in enumerate(trend_df.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='trend-card'>
                <b>{medals[i]} {row['fashion_category']}</b><br>
                <small style='color:#a0a0c0;'>
                Score: {row['trend_rank']:.4f} |
                Posts: {int(row['total_posts'])} |
                Positive: {row['positive_ratio']*100:.1f}%
                </small>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# PAGE 2 — DATA OVERVIEW
# ============================================

elif page == "📊 Data Overview":

    st.markdown(
        "<h2 class='section-header'>📊 Dataset Overview</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        # FIX 6: Compute counts fresh from df
        # not from cached/stale variable
        plat = df['platform'].value_counts().reset_index()
        plat.columns = ['platform', 'count']
        fig = px.pie(
            plat, values='count', names='platform',
            title=f'Posts by Platform (Total: {len(df)})',
            color_discrete_sequence=['#e91e8c', '#9c27b0'],
            hole=0.4
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # FIX 7: Same fix for language pie
        lang = df['language'].value_counts().reset_index()
        lang.columns = ['language', 'count']
        fig2 = px.pie(
            lang, values='count', names='language',
            title=f'Posts by Language (Total: {len(df)})',
            color_discrete_sequence=['#3498db', '#e67e22'],
            hole=0.4
        )
        fig2.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Posts over time
    st.markdown("### 📅 Posts Over Time")
    df_time = df.dropna(subset=['date']).copy()
    df_time['month'] = (
        df_time['date'].dt.to_period('M').astype(str)
    )
    monthly = df_time.groupby(
        ['month', 'platform']
    ).size().reset_index(name='count')
    monthly = monthly[monthly['month'] >= '2024-01']

    fig3 = px.line(
        monthly, x='month', y='count',
        color='platform',
        title='Monthly Post Volume (2024–2026)',
        color_discrete_map={
            'Instagram': '#e91e8c',
            'TikTok':    '#9c27b0'
        },
        markers=True
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title='Month',
        yaxis_title='Number of Posts'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Engagement
    st.markdown("### 💪 Engagement Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Likes",    f"{df['likes'].mean():.0f}")
    with c2:
        st.metric("Avg Comments", f"{df['comments'].mean():.0f}")
    with c3:
        st.metric("Avg Shares",   f"{df['shares'].mean():.0f}")
    with c4:
        st.metric("Avg Views",    f"{df['views'].mean():.0f}")

    st.markdown("### 🗂️ Sample Data")
    st.dataframe(
        df[['platform', 'text_clean', 'likes',
            'comments', 'sentiment',
            'fashion_category']].head(20),
        use_container_width=True
    )

# ============================================
# PAGE 3 — SENTIMENT ANALYSIS
# ============================================

elif page == "💬 Sentiment Analysis":

    st.markdown(
        "<h2 class='section-header'>💬 Sentiment Analysis</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        # FIX 8: Compute sentiment counts fresh from df
        sent = df['sentiment'].value_counts().reset_index()
        sent.columns = ['sentiment', 'count']
        color_map = {
            'Positive': '#2ecc71',
            'Neutral':  '#3498db',
            'Negative': '#e74c3c'
        }
        fig = px.pie(
            sent, values='count', names='sentiment',
            title='Overall Sentiment Distribution',
            color='sentiment',
            color_discrete_map=color_map,
            hole=0.45
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        platform_sent = pd.crosstab(
            df['platform'],
            df['sentiment'],
            normalize='index'
        ) * 100

        # Ensure all 3 sentiment columns exist
        for col in ['Positive', 'Neutral', 'Negative']:
            if col not in platform_sent.columns:
                platform_sent[col] = 0

        fig2 = px.bar(
            platform_sent.reset_index(),
            x='platform',
            y=['Positive', 'Neutral', 'Negative'],
            title='Sentiment by Platform (%)',
            color_discrete_map=color_map,
            barmode='group'
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title='Platform',
            yaxis_title='Percentage (%)'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # VADER histogram
    st.markdown("### 📈 VADER Compound Score Distribution")
    fig3 = px.histogram(
        df, x='vader_compound', nbins=40,
        title='Distribution of VADER Sentiment Scores (-1 to +1)',
        color_discrete_sequence=['#9c27b0']
    )
    fig3.add_vline(
        x=0.05, line_dash='dash',
        line_color='#2ecc71',
        annotation_text='Positive threshold (0.05)'
    )
    fig3.add_vline(
        x=-0.05, line_dash='dash',
        line_color='#e74c3c',
        annotation_text='Negative threshold (-0.05)'
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Top positive
    st.markdown("### 🌟 Top 5 Most Positive Posts")
    top_pos = df.nlargest(5, 'vader_compound')[
        ['platform', 'text_clean', 'vader_compound', 'likes']
    ]
    for _, row in top_pos.iterrows():
        st.success(
            f"**{row['platform']}** | "
            f"Score: {row['vader_compound']:.3f} | "
            f"Likes: {int(row['likes'])}\n\n"
            f"{str(row['text_clean'])[:200]}..."
        )

    # Top negative
    st.markdown("### ⚠️ Top 5 Most Negative Posts")
    top_neg = df.nsmallest(5, 'vader_compound')[
        ['platform', 'text_clean', 'vader_compound', 'likes']
    ]
    for _, row in top_neg.iterrows():
        st.error(
            f"**{row['platform']}** | "
            f"Score: {row['vader_compound']:.3f} | "
            f"Likes: {int(row['likes'])}\n\n"
            f"{str(row['text_clean'])[:200]}..."
        )

# ============================================
# PAGE 4 — TREND PREDICTION
# ============================================

elif page == "🔥 Trend Prediction":

    st.markdown(
        "<h2 class='section-header'>🔥 Fashion Trend Prediction</h2>",
        unsafe_allow_html=True
    )

    st.info(
        "📌 Trend rankings: "
        "Sentiment Score (50%) + Positive Ratio (30%) "
        "+ Post Volume (20%)"
    )

    # FIX 9: Remove texttemplate — use customdata instead
    # This fixes the literal '%{text:.4f}' bug
    trend_sorted = trend_df.sort_values('trend_rank')

    fig = go.Figure(go.Bar(
        x=trend_sorted['trend_rank'],
        y=trend_sorted['fashion_category'],
        orientation='h',
        marker=dict(
            color=trend_sorted['trend_rank'],
            colorscale=[[0, '#16213e'],
                        [0.5, '#e91e8c'],
                        [1, '#9c27b0']],
            showscale=False
        ),
        # FIX: Use text parameter correctly
        text=[f"{v:.4f}" for v in
              trend_sorted['trend_rank']],
        textposition='outside',
        textfont=dict(color='white', size=11)
    ))

    fig.update_layout(
        title='Fashion Trend Rankings — Nepali Female (18–26)',
        xaxis_title='Trend Rank Score',
        yaxis_title='Fashion Category',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=380,
        margin=dict(r=80)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics
    st.markdown("### 📊 Detailed Category Metrics")
    col1, col2 = st.columns(2)

    with col1:
        fig2 = go.Figure(go.Bar(
            x=trend_df['fashion_category'],
            y=trend_df['positive_ratio'] * 100,
            marker_color='#2ecc71',
            text=[f"{v*100:.1f}%"
                  for v in trend_df['positive_ratio']],
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig2.update_layout(
            title='Positive Sentiment % by Category',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=25,
            yaxis_title='Percentage (%)'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure(go.Bar(
            x=trend_df['fashion_category'],
            y=trend_df['avg_views'],
            marker_color='#e91e8c',
            text=[f"{v/1000:.0f}K"
                  for v in trend_df['avg_views']],
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig3.update_layout(
            title='Average Views by Category',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=25,
            yaxis_title='Average Views'
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Platform comparison
    st.markdown("### 📱 Trend Score by Platform")
    platform_trend = df.groupby(
        ['platform', 'fashion_category']
    )['trend_score'].mean().reset_index()

    fig4 = px.bar(
        platform_trend,
        x='fashion_category', y='trend_score',
        color='platform', barmode='group',
        title='Trend Score: Instagram vs TikTok',
        color_discrete_map={
            'Instagram': '#e91e8c',
            'TikTok':    '#9c27b0'
        }
    )
    fig4.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=20
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Summary table
    st.markdown("### 📋 Complete Trend Summary Table")
    display_df = trend_df[[
        'fashion_category', 'total_posts',
        'positive_ratio', 'avg_likes',
        'avg_views', 'trend_rank'
    ]].copy()
    display_df.columns = [
        'Category', 'Total Posts', 'Positive %',
        'Avg Likes', 'Avg Views', 'Trend Score'
    ]
    display_df['Positive %'] = (
        display_df['Positive %'] * 100
    ).round(1).astype(str) + '%'
    display_df['Avg Likes']  = (
        display_df['Avg Likes'].round(0).astype(int)
    )
    display_df['Avg Views']  = (
        display_df['Avg Views'].round(0).astype(int)
    )
    display_df['Trend Score'] = (
        display_df['Trend Score'].round(4)
    )
    st.dataframe(display_df, use_container_width=True)

# ============================================
# PAGE 5 — LIVE PREDICTOR
# ============================================

elif page == "🤖 Live Predictor":

    st.markdown(
        "<h2 class='section-header'>🤖 Live Fashion Trend Predictor</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "Enter any fashion-related post to instantly "
        "predict **sentiment**, **category**, and "
        "**trend potential**."
    )

    # FIX 10: Remove use_container_width from buttons
    # Not supported in Streamlit < 1.20
    st.markdown("### 💡 Try These Examples")

    examples = [
        "Beautiful nepali saree for dashain! "
        "Love this traditional look ❤️ #nepalifashion",
        "New kurti collection available. "
        "DM to order. Delivery all over Nepal 🇳🇵",
        "Outfit under Rs 2000! Casual western "
        "style for college girls nepal 🎓",
        "Poor quality fabric, very disappointed "
        "with this dress 😞 not worth the price",
        "Indo western fusion wear perfect for "
        "weddings and parties in Nepal ✨",
    ]

    # FIX 11: Use session_state to handle
    # example button selection properly
    if 'selected_text' not in st.session_state:
        st.session_state.selected_text = ""

    col_a, col_b = st.columns(2)
    for i, ex in enumerate(examples):
        with (col_a if i % 2 == 0 else col_b):
            # FIX: No use_container_width argument
            if st.button(f"📌 Example {i+1}",
                         key=f"btn_ex_{i}"):
                st.session_state.selected_text = ex

    st.markdown("---")

    user_input = st.text_area(
        "✍️ Enter Fashion Post Text:",
        value=st.session_state.selected_text,
        height=120,
        placeholder=(
            "Type or paste a fashion post here...\n"
            "e.g. Beautiful nepali saree! Love ❤️"
        )
    )

    # FIX 12: Predict button without
    # use_container_width for compatibility
    predict_clicked = st.button(
        "🔮 Predict Now", type="primary"
    )

    if predict_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text first!")
        else:
            with st.spinner("Analyzing..."):

                clean_input = re.sub(
                    r'http\S+|www\S+|@\w+|[#]',
                    '', user_input
                ).strip()

                vader_score = analyzer.polarity_scores(
                    user_input
                )['compound']

                input_vec = tfidf.transform([clean_input])
                sentiment = svm_model.predict(input_vec)[0]
                category  = cat_model.predict(input_vec)[0]

                if vader_score > 0.3:
                    trend       = "🔥 HIGH TREND POTENTIAL"
                    trend_color = "#2ecc71"
                elif vader_score > 0:
                    trend       = "📈 MEDIUM TREND POTENTIAL"
                    trend_color = "#f39c12"
                else:
                    trend       = "📉 LOW TREND POTENTIAL"
                    trend_color = "#e74c3c"

                sent_styles = {
                    'Positive': ('result-positive',
                                 '😊', '#2ecc71'),
                    'Neutral':  ('result-neutral',
                                 '😐', '#3498db'),
                    'Negative': ('result-negative',
                                 '😞', '#e74c3c'),
                }
                box_class, emoji, s_color = \
                    sent_styles.get(
                        sentiment,
                        ('result-neutral', '😐', '#3498db')
                    )

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div class='{box_class}'>
                    <div style='font-size:2.5rem'>{emoji}</div>
                    <div style='color:{s_color};
                        font-size:1.4rem;
                        font-weight:800;'>{sentiment}</div>
                    <div style='color:#a0a0c0;
                        font-size:0.85rem;'>Sentiment</div>
                    <div style='color:{s_color};
                        font-size:1.1rem;
                        font-weight:600;
                        margin-top:8px;'>
                        Score: {vader_score:.4f}</div>
                </div>""", unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:2rem;'>👗</div>
                    <div style='color:#e91e8c;
                        font-size:1.0rem;
                        font-weight:800;'>{category}</div>
                    <div style='color:#a0a0c0;
                        font-size:0.85rem;'>
                        Fashion Category</div>
                </div>""", unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:2rem;'>📊</div>
                    <div style='color:{trend_color};
                        font-size:0.95rem;
                        font-weight:800;'>{trend}</div>
                    <div style='color:#a0a0c0;
                        font-size:0.85rem;
                        margin-top:4px;'>
                        Trend Potential</div>
                </div>""", unsafe_allow_html=True)

            # VADER breakdown
            st.markdown("### 📊 VADER Score Breakdown")
            vader_full = analyzer.polarity_scores(user_input)
            s1, s2, s3, s4 = st.columns(4)
            score_items = [
                (s1, 'Positive', vader_full['pos'],  '#2ecc71'),
                (s2, 'Neutral',  vader_full['neu'],  '#3498db'),
                (s3, 'Negative', vader_full['neg'],  '#e74c3c'),
                (s4, 'Compound', vader_full['compound'], '#e91e8c'),
            ]
            for col, label, val, color in score_items:
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p class='metric-value'
                           style='color:{color};
                           font-size:1.6rem;'>
                            {val:.3f}</p>
                        <p class='metric-label'>{label}</p>
                    </div>""", unsafe_allow_html=True)

            # Category context
            st.markdown("### 📈 How This Category Trends")
            cat_row = trend_df[
                trend_df['fashion_category'] == category
            ]
            if not cat_row.empty:
                row  = cat_row.iloc[0]
                rank = cat_row.index[0] + 1
                medals = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣']
                medal  = medals[min(rank-1, 5)]
                st.info(
                    f"{medal} **{category}** ranks "
                    f"**#{rank}** in overall fashion trends!\n\n"
                    f"📊 {int(row['total_posts'])} posts | "
                    f"✅ {row['positive_ratio']*100:.1f}% Positive"
                    f" | ❤️ Avg {int(row['avg_likes'])} likes"
                    f" | 👁️ Avg {int(row['avg_views'])} views"
                )

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class='footer'>
    Fashion Trend Prediction System |
    Aruna Guragain | Supervisor: Manoj Shrestha<br>
    Data: Instagram + TikTok |
    Model: SVM | Dashboard v2.0 | © 2026
</div>
""", unsafe_allow_html=True)