# ============================================
# FASHION TREND PREDICTION DASHBOARD
# By: Aruna Guragain
# Supervisor: Manoj Shrestha
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Compatibility shim: map newer Streamlit cache APIs to older `st.cache`
# when running with older Streamlit versions that don't provide
# `st.cache_data` / `st.cache_resource` (e.g., streamlit 1.12).
if not hasattr(st, 'cache_data'):
    st.cache_data = st.cache
if not hasattr(st, 'cache_resource'):
    st.cache_resource = st.cache

# Make `st.dataframe` tolerant of newer kwargs (e.g. `use_container_width`,
# `hide_index`) when running on older Streamlit versions that don't accept
# those keywords. It will fall back to calling the original function without
# extra kwargs if a TypeError is raised.
_st_dataframe_orig = st.dataframe
def _st_dataframe_compat(df, **kwargs):
    try:
        return _st_dataframe_orig(df, **kwargs)
    except TypeError:
        return _st_dataframe_orig(df)
st.dataframe = _st_dataframe_compat

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
    /* Main background */
    .main {
        background-color: #0f0f1a;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a2e;
    }

    /* Metric cards */
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

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #e91e8c, #9c27b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* Trend rank cards */
    .trend-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #e91e8c;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 8px 0;
    }

    .trend-rank {
        font-size: 1.5rem;
        font-weight: 800;
        color: #e91e8c;
    }

    .trend-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
    }

    .trend-score {
        font-size: 0.9rem;
        color: #a0a0c0;
    }

    /* Prediction result */
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

    /* Footer */
    .footer {
        text-align: center;
        color: #606080;
        font-size: 0.8rem;
        padding: 20px 0;
        border-top: 1px solid #2a2a4a;
        margin-top: 40px;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA AND TRAIN MODEL
# ============================================

@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / 'data' / 'cleaned' / 'fashion_data_with_sentiment.csv'
    df = pd.read_csv(csv_path)
    df['text_clean'] = df['text_clean'].fillna('')
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    return df

@st.cache_resource
def train_models(df):
    X = df['text_clean']
    y_sent = df['sentiment']
    y_cat  = df['fashion_category']

    X_train, X_test, y_train_s, y_test_s = train_test_split(
        X, y_sent, test_size=0.2,
        random_state=42, stratify=y_sent
    )
    _, _, y_train_c, y_test_c = train_test_split(
        X, y_cat, test_size=0.2,
        random_state=42, stratify=y_cat
    )

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1,2),
        min_df=2, stop_words='english', sublinear_tf=True
    )
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v  = tfidf.transform(X_test)

    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    svm.fit(X_train_v, y_train_s)

    cat_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    cat_model.fit(X_train_v, y_train_c)

    svm_acc = svm.score(X_test_v, y_test_s)
    return tfidf, svm, cat_model, svm_acc

@st.cache_data
def get_trend_analysis(df):
    trend = df.groupby('fashion_category').agg(
        total_posts     = ('text_clean', 'count'),
        avg_sentiment   = ('vader_compound', 'mean'),
        positive_ratio  = ('sentiment',
                          lambda x: (x=='Positive').sum()/len(x)),
        avg_likes       = ('likes', 'mean'),
        avg_comments    = ('comments', 'mean'),
        avg_shares      = ('shares', 'mean'),
        avg_views       = ('views', 'mean'),
        avg_trend_score = ('trend_score', 'mean'),
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

# Load everything
df          = load_data()
tfidf, svm_model, cat_model, svm_acc = train_models(df)
trend_df    = get_trend_analysis(df)
analyzer    = SentimentIntensityAnalyzer()

# ============================================
# SIDEBAR NAVIGATION
# ============================================

st.sidebar.markdown("""
<div style='text-align:center; padding: 20px 0;'>
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
    [
        "🏠 Home",
        "📊 Data Overview",
        "💬 Sentiment Analysis",
        "🔥 Trend Prediction",
        "🤖 Live Predictor",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.8rem; color:#606080;'>
    <b>Submitted by:</b> Aruna Guragain<br>
    <b>Supervisor:</b> Manoj Shrestha<br>
    <b>Dataset:</b> 790 posts<br>
    <b>Platforms:</b> Instagram + TikTok<br>
    <b>Best Model:</b> SVM (71.52%)
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE 1 — HOME
# ============================================

if page == "🏠 Home":

    st.markdown("""
    <h1 style='text-align:center; background:
        linear-gradient(90deg, #e91e8c, #9c27b0);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        font-size:2.5rem; font-weight:800;'>
        Fashion Trend Prediction System
    </h1>
    <p style='text-align:center; color:#a0a0c0;
              font-size:1.1rem; margin-bottom:30px;'>
        For Nepali Females Aged 18–26 |
        Using Sentiment Analysis & Machine Learning
    </p>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <p class='metric-value'>790</p>
            <p class='metric-label'>Total Posts Analyzed</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <p class='metric-value'>71.52%</p>
            <p class='metric-label'>SVM Model Accuracy</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <p class='metric-value'>65.6%</p>
            <p class='metric-label'>Positive Sentiment</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
            <p class='metric-value'>#1</p>
            <p class='metric-label'>Traditional/Ethnic Trend</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Project Overview
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🎯 Project Aim")
        st.info(
            "Design and develop a fashion trend prediction system "
            "using social media sentiment analysis and machine "
            "learning to forecast fashion trends for Nepali females "
            "aged 18–26."
        )

        st.markdown("### 📱 Data Sources")
        source_data = {
            'Platform': ['Instagram', 'TikTok'],
            'Posts':    [328, 462],
            'Percentage': ['41.5%', '58.5%']
        }
        st.dataframe(
            pd.DataFrame(source_data),
            use_container_width=True
        )

    with col_right:
        st.markdown("### 🔄 System Pipeline")
        pipeline_steps = [
            "1️⃣  Data Collection (Apify Scraper)",
            "2️⃣  Data Cleaning & Preprocessing",
            "3️⃣  NLP & TF-IDF Vectorization",
            "4️⃣  VADER Sentiment Analysis",
            "5️⃣  ML Model Training (NB + SVM)",
            "6️⃣  Fashion Trend Ranking",
            "7️⃣  Live Prediction Dashboard",
        ]
        for step in pipeline_steps:
            st.success(step)

    st.markdown("---")
    st.markdown("### 🏆 Top Fashion Trends — Nepali Female (18–26)")

    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣']
    cols   = st.columns(3)
    for i, (_, row) in enumerate(trend_df.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='trend-card'>
                <span class='trend-rank'>{medals[i]}</span>
                <span class='trend-name'>
                    &nbsp;{row['fashion_category']}
                </span><br>
                <span class='trend-score'>
                    Score: {row['trend_rank']:.4f} |
                    Posts: {int(row['total_posts'])} |
                    Positive: {row['positive_ratio']*100:.1f}%
                </span>
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

    # Platform distribution
    col1, col2 = st.columns(2)

    with col1:
        platform_counts = df['platform'].value_counts()
        fig = px.pie(
            values=platform_counts.values,
            names=platform_counts.index,
            title='Posts by Platform',
            color_discrete_sequence=['#e91e8c', '#9c27b0'],
            hole=0.4
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        lang_counts = df['language'].value_counts()
        fig2 = px.pie(
            values=lang_counts.values,
            names=lang_counts.index,
            title='Posts by Language',
            color_discrete_sequence=['#3498db', '#e67e22'],
            hole=0.4
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Posts over time
    st.markdown("### 📅 Posts Over Time")
    df_time = df.dropna(subset=['date']).copy()
    df_time['month'] = df_time['date'].dt.to_period('M').astype(str)
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

    # Engagement stats
    st.markdown("### 💪 Engagement Statistics")
    col3, col4, col5, col6 = st.columns(4)

    with col3:
        st.metric("Avg Likes",
                  f"{df['likes'].mean():.0f}")
    with col4:
        st.metric("Avg Comments",
                  f"{df['comments'].mean():.0f}")
    with col5:
        st.metric("Avg Shares",
                  f"{df['shares'].mean():.0f}")
    with col6:
        st.metric("Avg Views",
                  f"{df['views'].mean():.0f}")

    # Raw data preview
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

    # Overall sentiment
    col1, col2 = st.columns(2)

    with col1:
        sent_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            title='Overall Sentiment Distribution',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral':  '#3498db',
                'Negative': '#e74c3c'
            },
            hole=0.45
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

        fig2 = px.bar(
            platform_sent.reset_index(),
            x='platform',
            y=['Positive', 'Neutral', 'Negative'],
            title='Sentiment by Platform (%)',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral':  '#3498db',
                'Negative': '#e74c3c'
            },
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

    # VADER score distribution
    st.markdown("### 📈 VADER Compound Score Distribution")
    fig3 = px.histogram(
        df, x='vader_compound',
        nbins=40,
        title='Distribution of Sentiment Scores',
        color_discrete_sequence=['#9c27b0']
    )
    fig3.add_vline(
        x=0.05, line_dash='dash',
        line_color='#2ecc71',
        annotation_text='Positive threshold'
    )
    fig3.add_vline(
        x=-0.05, line_dash='dash',
        line_color='#e74c3c',
        annotation_text='Negative threshold'
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Top positive posts
    st.markdown("### 🌟 Top 5 Most Positive Posts")
    top_pos = df.nlargest(5, 'vader_compound')[
        ['platform', 'text_clean',
         'vader_compound', 'likes']
    ]
    for _, row in top_pos.iterrows():
        st.success(
            f"**{row['platform']}** | "
            f"Score: {row['vader_compound']:.3f} | "
            f"Likes: {int(row['likes'])}\n\n"
            f"{str(row['text_clean'])[:200]}..."
        )

    # Top negative posts
    st.markdown("### ⚠️ Top 5 Most Negative Posts")
    top_neg = df.nsmallest(5, 'vader_compound')[
        ['platform', 'text_clean',
         'vader_compound', 'likes']
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
        "📌 Trend rankings are calculated using a composite score: "
        "Sentiment Score (50%) + Positive Ratio (30%) + "
        "Post Volume (20%)"
    )

    # Trend ranking chart
    fig = px.bar(
        trend_df.sort_values('trend_rank'),
        x='trend_rank',
        y='fashion_category',
        orientation='h',
        title='Fashion Trend Rankings — Nepali Female (18–26)',
        color='trend_rank',
        color_continuous_scale=[
            '#1a1a2e', '#e91e8c', '#9c27b0'
        ],
        text='trend_rank'
    )
    fig.update_traces(
        texttemplate='%{text:.4f}',
        textposition='outside'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400,
        xaxis_title='Trend Rank Score',
        yaxis_title='Fashion Category',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics
    st.markdown("### 📊 Detailed Category Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.bar(
            trend_df,
            x='fashion_category',
            y='positive_ratio',
            title='Positive Sentiment Ratio by Category',
            color='positive_ratio',
            color_continuous_scale=['#1a1a2e', '#2ecc71'],
            text=trend_df['positive_ratio'].apply(
                lambda x: f'{x*100:.1f}%'
            )
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=30,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.bar(
            trend_df,
            x='fashion_category',
            y='avg_views',
            title='Average Views by Category',
            color='avg_views',
            color_continuous_scale=['#1a1a2e', '#e91e8c'],
            text=trend_df['avg_views'].apply(
                lambda x: f'{x/1000:.0f}K'
            )
        )
        fig3.update_traces(textposition='outside')
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=30,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Platform comparison
    st.markdown("### 📱 Trend Score by Platform")
    platform_trend = df.groupby(
        ['platform', 'fashion_category']
    )['trend_score'].mean().reset_index()

    fig4 = px.bar(
        platform_trend,
        x='fashion_category',
        y='trend_score',
        color='platform',
        barmode='group',
        title='Trend Score Comparison: Instagram vs TikTok',
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
        'Category', 'Total Posts',
        'Positive %', 'Avg Likes',
        'Avg Views', 'Trend Score'
    ]
    display_df['Positive %'] = (
        display_df['Positive %'] * 100
    ).round(1).astype(str) + '%'
    display_df['Avg Likes']  = display_df['Avg Likes'].round(0).astype(int)
    display_df['Avg Views']  = display_df['Avg Views'].round(0).astype(int)
    display_df['Trend Score']= display_df['Trend Score'].round(4)

    st.dataframe(
        display_df,
        use_container_width=True
    )

# ============================================
# PAGE 5 — LIVE PREDICTOR
# ============================================

elif page == "🤖 Live Predictor":

    st.markdown(
        "<h2 class='section-header'>🤖 Live Fashion Trend Predictor</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "Enter any fashion-related social media post below "
        "to instantly predict its **sentiment**, "
        "**fashion category**, and **trend potential**."
    )

    # Quick examples
    st.markdown("### 💡 Try These Examples")
    examples = [
        "Beautiful nepali saree for dashain! "
        "Love this traditional look ❤️ #nepalifashion",
        "New kurti collection available. "
        "DM to order. Delivery all over Nepal 🇳🇵",
        "Outfit under Rs 2000! Casual western "
        "style for college girls nepal 🎓",
        "Poor quality fabric very disappointed "
        "with this dress not worth the price 😞",
        "Indo western fusion wear perfect for "
        "weddings and parties in Nepal ✨",
    ]

    col_ex1, col_ex2 = st.columns(2)
    selected_example = None

    for i, ex in enumerate(examples):
        if i % 2 == 0:
            with col_ex1:
                if st.button(f"📌 Example {i+1}",
                             key=f"ex_{i}",
                             use_container_width=True):
                    selected_example = ex
        else:
            with col_ex2:
                if st.button(f"📌 Example {i+1}",
                             key=f"ex_{i}",
                             use_container_width=True):
                    selected_example = ex

    st.markdown("---")

    # Text input
    user_input = st.text_area(
        "✍️ Enter Fashion Post Text:",
        value=selected_example if selected_example else "",
        height=120,
        placeholder=(
            "Type or paste a fashion post here...\n"
            "e.g. Beautiful nepali saree for dashain! "
            "Love this look ❤️"
        )
    )

    # Predict button
    if st.button("🔮 Predict Now", type="primary",
                 use_container_width=True):

        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first!")
        else:
            with st.spinner("Analyzing..."):

                # Clean input
                clean_input = re.sub(
                    r'http\S+|www\S+|@\w+|[#]', '', user_input
                ).strip()

                # VADER score
                vader_score = analyzer.polarity_scores(
                    user_input
                )['compound']

                # ML predictions
                input_vec  = tfidf.transform([clean_input])
                sentiment  = svm_model.predict(input_vec)[0]
                category   = cat_model.predict(input_vec)[0]

                # Trend potential
                if vader_score > 0.3:
                    trend = "🔥 HIGH TREND POTENTIAL"
                    trend_color = "#2ecc71"
                elif vader_score > 0:
                    trend = "📈 MEDIUM TREND POTENTIAL"
                    trend_color = "#f39c12"
                else:
                    trend = "📉 LOW TREND POTENTIAL"
                    trend_color = "#e74c3c"

                # Sentiment styling
                if sentiment == 'Positive':
                    box_class = 'result-positive'
                    emoji     = '😊'
                    color     = '#2ecc71'
                elif sentiment == 'Neutral':
                    box_class = 'result-neutral'
                    emoji     = '😐'
                    color     = '#3498db'
                else:
                    box_class = 'result-negative'
                    emoji     = '😞'
                    color     = '#e74c3c'

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            # Results display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class='{box_class}'>
                    <div style='font-size:2.5rem'>{emoji}</div>
                    <div style='color:{color};
                                font-size:1.4rem;
                                font-weight:800;'>
                        {sentiment}
                    </div>
                    <div style='color:#a0a0c0;
                                font-size:0.85rem;'>
                        Sentiment
                    </div>
                    <div style='color:{color};
                                font-size:1.1rem;
                                font-weight:600;
                                margin-top:8px;'>
                        Score: {vader_score:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:2rem;'>👗</div>
                    <div style='color:#e91e8c;
                                font-size:1.1rem;
                                font-weight:800;'>
                        {category}
                    </div>
                    <div style='color:#a0a0c0;
                                font-size:0.85rem;'>
                        Fashion Category
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:2rem;'>📊</div>
                    <div style='color:{trend_color};
                                font-size:1rem;
                                font-weight:800;'>
                        {trend}
                    </div>
                    <div style='color:#a0a0c0;
                                font-size:0.85rem;
                                margin-top:4px;'>
                        Trend Potential
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # VADER breakdown
            st.markdown("### 📊 Detailed VADER Score Breakdown")
            vader_full = analyzer.polarity_scores(user_input)

            score_cols = st.columns(4)
            scores = {
                'Positive': (vader_full['pos'], '#2ecc71'),
                'Neutral':  (vader_full['neu'], '#3498db'),
                'Negative': (vader_full['neg'], '#e74c3c'),
                'Compound': (vader_full['compound'], '#e91e8c'),
            }
            for col, (label, (val, color)) in zip(
                score_cols, scores.items()
            ):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p class='metric-value'
                           style='color:{color};
                                  font-size:1.6rem;'>
                            {val:.3f}
                        </p>
                        <p class='metric-label'>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Category context
            st.markdown("### 📈 How This Category Trends")
            cat_row = trend_df[
                trend_df['fashion_category'] == category
            ]
            if not cat_row.empty:
                row = cat_row.iloc[0]
                rank = trend_df[
                    trend_df['fashion_category'] == category
                ].index[0] + 1
                medals = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣']
                st.info(
                    f"{medals[rank-1]} **{category}** ranks "
                    f"**#{rank}** in overall fashion trends!\n\n"
                    f"📊 Based on {int(row['total_posts'])} posts | "
                    f"✅ {row['positive_ratio']*100:.1f}% Positive | "
                    f"❤️ Avg {int(row['avg_likes'])} likes | "
                    f"👁️ Avg {int(row['avg_views'])} views"
                )

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class='footer'>
    Fashion Trend Prediction System |
    Aruna Guragain | Supervisor: Manoj Shrestha<br>
    Data: Instagram + TikTok |
    Model: SVM (71.52% Accuracy) |
    © 2026
</div>
""", unsafe_allow_html=True)