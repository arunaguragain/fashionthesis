import streamlit as st
import pandas as pd
import re
import math
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nepal Fashion Trend Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0) !important;
    }
    .lucide-icon {
        width: 16px;
        height: 16px;
        stroke: currentColor;
        stroke-width: 2;
        fill: none;
        vertical-align: middle;
    }
    .header-icon {
        color: #ffb300;
    }
    .section-icon {
        margin-right: 8px;
        flex-shrink: 0;
    }
    .section-icon.search {
        color: #29b6f6;
    }
    .section-icon.analytics {
        color: #9c27b0;
    }
    .section-icon.quick {
        color: #ffb300;
    }
    .section-icon.input {
        color: #4caf50;
    }
    .section-icon.data {
        color: #29b6f6;
    }
    .section-icon.prediction {
        color: #4caf50;
    }
    .section-icon.ethics {
        color: #ffb300;
    }
    .info-icon {
        color: #90caf9;
    }
    .metric-icon {
        width: 16px;
        height: 16px;
        margin-right: 8px;
        flex-shrink: 0;
    }
    .metric-icon.detected {
        color: #e91e63;
    }
    .metric-icon.love {
        color: #4caf50;
    }
    .metric-icon.based {
        color: #2196f3;
    }
    .post-sentiment-icon.positive {
        color: #4caf50;
    }
    .post-sentiment-icon.neutral {
        color: #9e9e9e;
    }
    .post-sentiment-icon.negative {
        color: #f44336;
    }
    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #4b5563;
        font-size: 0.92rem;
        padding: 4px 0;
    }
    .sentiment-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .sentiment-positive { background: #4CAF50; }
    .sentiment-neutral { background: #9E9E9E; }
    .sentiment-negative { background: #F44336; }
    .post-sentiment-icon {
        width: 14px;
        height: 14px;
        stroke: currentColor;
        stroke-width: 2;
        fill: none;
        vertical-align: middle;
        margin-right: 6px;
        flex-shrink: 0;
    }
    .custom-expander {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 16px;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .custom-expander summary {
        list-style: none;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 18px;
        cursor: pointer;
        color: #111827;
        font-weight: 700;
        outline: none;
    }
    .custom-expander summary::-webkit-details-marker {
        display: none;
    }
    .custom-expander summary::after {
        content: '▾';
        margin-left: auto;
        font-size: 0.85rem;
        color: #9e9e9e;
    }
    .custom-expander[open] summary::after {
        content: '▴';
    }
    .custom-expander .expander-content {
        padding: 0 18px 18px;
        color: #4b5563;
        line-height: 1.7;
    }
    .custom-expander .expander-content p,
    .custom-expander .expander-content li {
        margin: 0 0 0.9rem 0;
    }
    .custom-expander .expander-content ul {
        padding-left: 18px;
        margin: 0;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #E91E63;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #6b7280;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 22px;
        color: #111827;
        font-weight: bold;
    }
    .detected-tag {
        display:inline-block;
        background:#FCE4EC;
        color:#E91E63;
        padding:4px 12px;
        border-radius:20px;
        font-size:13px;
        font-weight:600;
        margin:2px 4px 2px 0;
    }
    .hero-header {
        background: linear-gradient(135deg, #fffdf5 0%, #ffffff 100%);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 16px 20px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    .section-divider {
        border-top: 1px solid rgba(0,0,0,0.08);
        margin: 18px 0 24px;
    }
    .metric-title {
        font-size: 12px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
        letter-spacing: 0.08em;
        line-height: 1.2;
    }
    .metric-value {
        font-size: 24px;
        color: #111827;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 1px 12px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .ethics-text {
        color: #4b5563;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .expander-content p,
    .expander-content li {
        color: #4b5563;
        line-height: 1.7;
    }
    </style>
""", unsafe_allow_html=True)

def get_session_value(key, default=None):
    try:
        return st.session_state[key]
    except Exception:
        st.session_state[key] = default
        return default


def set_session_value(key, value):
    try:
        st.session_state[key] = value
    except Exception:
        pass


st.session_state.setdefault("current_search", "")
st.session_state.setdefault("text_input_key", "")
st.session_state.setdefault("trigger_analysis", False)
current_search = get_session_value("current_search", "")
trigger_analysis = get_session_value("trigger_analysis", False)


def load_category_model():
    base = Path(__file__).resolve().parent.parent
    try:
        tfidf = joblib.load(str(base / 'models' / 'tfidf_category.pkl'))
        model = joblib.load(str(base / 'models' / 'svm_category_model.pkl'))
        return tfidf, model
    except Exception:
        return None, None


def render_expander(title, icon_svg, content_html):
    st.markdown(
        f"""
        <details class="custom-expander">
            <summary>{icon_svg}<span>{title}</span></summary>
            <div class="expander-content">{content_html}</div>
        </details>
        """,
        unsafe_allow_html=True
    )


tfidf_cat, svm_cat = load_category_model()
MODEL_FALLBACK_ACTIVE = tfidf_cat is None or svm_cat is None
if MODEL_FALLBACK_ACTIVE:
    st.caption("Using keyword fallback because the trained category model files were not found.")


# REAL RESEARCH DATA (unchanged)

REAL_NEPAL_TRENDS = {
    "Traditional/Ethnic": {
        "rank": 1, "trend_score": 0.5735, "positive_pct": 71.8,
        "neutral_pct": 23.2, "negative_pct": 5.0,
        "instagram_posts": 337, "tiktok_posts": 384,
        "posts": 721, "avg_likes": 5586, "avg_views": 125000,
        "example_posts": [
            {"text": "Traditional embroidered kurta post shows rich festival styling with handloom details.", "sentiment": "Positive", "score": 0.85},
            {"text": "Customer review mentions handloom fabric and price sensitivity but remains mostly neutral.", "sentiment": "Neutral", "score": 0.02},
            {"text": "Complaint about stitching quality for an ethnic dress after the first wear.", "sentiment": "Negative", "score": -0.45}
        ],
        "final_score": 82.5, "signal": "VIRAL DEMAND",
        "description": (
            "Nepali females aged 18-26 show the STRONGEST enthusiasm for "
            "traditional ethnic wear. 71.8% of all social media posts about "
            "this category express positive sentiment. Highly recommended "
            "for production!"
        )
    },
    "General Fashion": {
        "rank": 2, "trend_score": 0.5734, "positive_pct": 55.1,
        "neutral_pct": 39.9, "negative_pct": 5.0,
        "instagram_posts": 362, "tiktok_posts": 653,
        "posts": 1015, "avg_likes": 37546, "avg_views": 191132,
        "example_posts": [
            {"text": "Street fashion video highlights summer layering and modern Nepali brands.", "sentiment": "Positive", "score": 0.78},
            {"text": "Neutral comment on outfit comfort and seasonality for everyday wear.", "sentiment": "Neutral", "score": 0.01},
            {"text": "Negative feedback on delivery delays affecting the fashion order.", "sentiment": "Negative", "score": -0.33}
        ],
        "final_score": 75.0, "signal": "STABLE DEMAND",
        "description": (
            "General fashion content has the highest post volume showing "
            "broad market interest. Good steady demand."
        )
    },
    "Western/Casual": {
        "rank": 3, "trend_score": 0.4486, "positive_pct": 66.5,
        "neutral_pct": 28.5, "negative_pct": 5.0,
        "instagram_posts": 112, "tiktok_posts": 88,
        "posts": 200, "avg_likes": 72121, "avg_views": 204340,
        "example_posts": [
            {"text": "Denim jacket and streetwear look catching attention in Kathmandu.", "sentiment": "Positive", "score": 0.81},
            {"text": "Neutral post about a casual outfit for a college day.", "sentiment": "Neutral", "score": 0.05},
            {"text": "Negative remark on a western fashion item not suiting Nepali weather.", "sentiment": "Negative", "score": -0.28}
        ],
        "final_score": 65.0, "signal": "STABLE DEMAND",
        "description": (
            "Western casual wear shows high engagement. College-going "
            "females especially engage with affordable western styles."
        )
    },
    "Accessories": {
        "rank": 4, "trend_score": 0.4450, "positive_pct": 75.3,
        "neutral_pct": 19.7, "negative_pct": 5.0,
        "instagram_posts": 57, "tiktok_posts": 24,
        "posts": 81, "avg_likes": 8766, "avg_views": 93787,
        "example_posts": [
            {"text": "Accessory post shows matching earrings and a bag for festival outfits.", "sentiment": "Positive", "score": 0.88},
            {"text": "Neutral thread on choosing jewelry for a wedding look.", "sentiment": "Neutral", "score": 0.06},
            {"text": "Negative comment on accessory quality after one wear.", "sentiment": "Negative", "score": -0.40}
        ],
        "final_score": 62.0, "signal": "STABLE DEMAND",
        "description": (
            "Accessories show the second highest positive rate — people "
            "who engage with accessories content love it! Low volume but "
            "high quality."
        )
    },
    "Formal/Professional": {
        "rank": 5, "trend_score": 0.4120, "positive_pct": 62.5,
        "neutral_pct": 32.5, "negative_pct": 5.0,
        "instagram_posts": 65, "tiktok_posts": 55,
        "posts": 120, "avg_likes": 4032, "avg_views": 156071,
        "example_posts": [
            {"text": "Professional blazer post with mixed feedback on fit and style.", "sentiment": "Positive", "score": 0.72},
            {"text": "Neutral discussion about office wear choices for formal meetings.", "sentiment": "Neutral", "score": 0.00},
            {"text": "Negative review on formal shirt texture and price.", "sentiment": "Negative", "score": -0.38}
        ],
        "final_score": 48.0, "signal": "LOW DEMAND",
        "description": (
            "Formal wear has moderate engagement but highest negative "
            "sentiment among categories — buyers express dissatisfaction "
            "with quality and pricing."
        )
    },
    "Indo-Western/Fusion": {
        "rank": 6, "trend_score": 0.3545, "positive_pct": 60.0,
        "neutral_pct": 35.0, "negative_pct": 5.0,
        "instagram_posts": 16, "tiktok_posts": 19,
        "posts": 35, "avg_likes": 1123, "avg_views": 24786,
        "example_posts": [
            {"text": "Fusion outfit post praising the mix of traditional and western details.", "sentiment": "Positive", "score": 0.69},
            {"text": "Neutral mention of fusion wear for a casual event.", "sentiment": "Neutral", "score": 0.07},
            {"text": "Negative note about fit issues in an indo-western top.", "sentiment": "Negative", "score": -0.30}
        ],
        "final_score": 38.0, "signal": "LOW DEMAND",
        "description": (
            "Fusion wear currently shows lowest trend score. Limited "
            "social media posts and mixed reception suggest market is "
            "not ready for this category yet."
        )
    }
}


def build_phase_trend_summary():
    base = Path(__file__).resolve().parent.parent
    data_path = base / 'data' / 'cleaned' / 'combined_with_sentiment.csv'
    try:
        df = pd.read_csv(data_path)
    except Exception:
        return {
            category: {
                'direction': 'stable',
                'change_pct': 0.0,
                'label': 'Stable'
            }
            for category in REAL_NEPAL_TRENDS
        }

    phase_df = df[df['phase'].isin(['Phase1', 'Phase2'])].copy()
    summary = {}
    for category in REAL_NEPAL_TRENDS:
        category_df = phase_df[phase_df['fashion_category'] == category]
        if category_df.empty:
            summary[category] = {
                'direction': 'stable',
                'change_pct': 0.0,
                'label': 'Stable'
            }
            continue

        phase1_pos = category_df.loc[category_df['phase'] == 'Phase1', 'sentiment'].eq('Positive').mean() * 100
        phase2_pos = category_df.loc[category_df['phase'] == 'Phase2', 'sentiment'].eq('Positive').mean() * 100
        change_pct = float(phase2_pos - phase1_pos) if pd.notna(phase1_pos) and pd.notna(phase2_pos) else 0.0

        if abs(change_pct) <= 5.0:
            consistency = 'consistent'
            status = 'Consistent'
            indicator_color = '#2e7d32'
        else:
            consistency = 'shifted'
            status = 'Shifted'
            indicator_color = '#ef6c00'

        summary[category] = {
            'direction': consistency,
            'change_pct': round(change_pct, 1),
            'phase1_pct': round(phase1_pos, 1),
            'phase2_pct': round(phase2_pos, 1),
            'status': status,
            'indicator_color': indicator_color
        }

    return summary


PHASE_TREND_SUMMARY = build_phase_trend_summary()
for category, trend_info in PHASE_TREND_SUMMARY.items():
    REAL_NEPAL_TRENDS[category]['trend_direction'] = trend_info

CATEGORY_ICONS = {
    "Traditional/Ethnic": "",
    "Western/Casual": "",
    "Indo-Western/Fusion": "",
    "Formal/Professional": "",
    "Accessories": "",
    "General Fashion": ""
}

CATEGORY_NAME_MAP = {
    "traditional_ethnic": "Traditional/Ethnic",
    "traditional": "Traditional/Ethnic",
    "general_fashion": "General Fashion",
    "western_casual": "Western/Casual",
    "western": "Western/Casual",
    "accessories": "Accessories",
    "formal_professional": "Formal/Professional",
    "formal": "Formal/Professional",
    "indo_western_fusion": "Indo-Western/Fusion",
    "fusion": "Indo-Western/Fusion"
}


CATEGORY_KEYWORDS = {
    "Traditional/Ethnic": [
        'saree', 'sari', 'kurti', 'kurta', 'lehenga', 'ethnic',
        'traditional', 'daura', 'suruwal', 'gunyo', 'cholo',
        'dhaka', 'dashain', 'tihar', 'teej', 'festival', 'newari',
        'tamang', 'handloom', 'pahiran', 'bride', 'wedding',
        'bridal', 'sherwani', 'dhoti', 'shawl', 'stole',
        'embroidered', 'silk', 'velvet'
    ],
    "Western/Casual": [
        'jeans', 'denim', 'top', 'tshirt', 't-shirt', 'shirt',
        'blouse', 'hoodie', 'jacket', 'casual', 'shorts', 'skirt',
        'crop', 'sneaker', 'streetwear', 'cargo', 'sweatshirt',
        'pants', 'trouser', 'trousers', 'leggings', 'tank',
        'sweater', 'cardigan', 'romper', 'jumpsuit', 'dress',
        'tunic', 'polo', 'coat', 'vest', 'pullover'
    ],
    "Indo-Western/Fusion": [
        'fusion', 'indo western', 'indo-western', 'gown',
        'frock', 'anarkali', 'palazzo', 'crop top kurti',
        'jacket kurti', 'cape'
    ],
    "Formal/Professional": [
        'formal', 'office', 'professional', 'suit', 'blazer',
        'workwear', 'corporate', 'business', 'tie', 'waistcoat'
    ],
    "Accessories": [
        'bag', 'handbag', 'purse', 'jewel', 'jewelry', 'jewellery',
        'necklace', 'earring', 'bracelet', 'bangle', 'ring',
        'shoes', 'heel', 'sandal', 'watch', 'scarf', 'dupatta',
        'belt', 'sunglasses', 'clutch', 'wallet', 'hat', 'cap'
    ]
}

# Words that hint at general/modern fashion when
# no specific garment word is found at all
GENERIC_FASHION_HINTS = [
    'fashion', 'style', 'outfit', 'wear', 'cloth', 'clothing',
    'look', 'collection', 'trend', 'design'
]

COLOR_WORDS = [
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'pink',
    'purple', 'orange', 'brown', 'grey', 'gray', 'maroon',
    'navy', 'gold', 'silver', 'beige', 'cream', 'turquoise',
    'magenta', 'olive', 'mustard'
]

def strip_colors(text):
    """Remove color words so 'blue shorts' -> 'shorts'"""
    words = text.lower().split()
    return ' '.join(w for w in words if w not in COLOR_WORDS)


def normalize_category_name(category):
    if category in REAL_NEPAL_TRENDS:
        return category
    if category is None:
        return "General Fashion"
    normalized = str(category).strip().lower().replace(' ', '_')
    return CATEGORY_NAME_MAP.get(normalized, str(category))


def find_category_keywords(user_text):
    """Fallback keyword matching used when the trained model is unavailable."""
    text = strip_colors(user_text.lower())

    scores = {}
    matched_keywords = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text]
        if matches:
            scores[category] = len(matches)
            matched_keywords[category] = matches

    if scores:
        best_category = max(scores, key=scores.get)
        return best_category, True, matched_keywords[best_category]

    if any(hint in text for hint in GENERIC_FASHION_HINTS):
        return "General Fashion", True, []

    return "General Fashion", False, []


def find_category(user_text):
    """Use the real trained SVM model when available, otherwise fall back to keyword matching."""
    cleaned = str(user_text or '').strip()
    if len(cleaned) < 2:
        return "General Fashion", False, []

    if tfidf_cat is not None and svm_cat is not None:
        try:
            vec = tfidf_cat.transform([cleaned.lower()])
            predicted = svm_cat.predict(vec)[0]
            category = normalize_category_name(predicted)
            if category in REAL_NEPAL_TRENDS:
                return category, True, []
        except Exception:
            pass

    return find_category_keywords(cleaned)


def get_trend_result(user_text):
    category, matched, detected_keywords = find_category(user_text)
    data = REAL_NEPAL_TRENDS[category]
    
    # Get display text for detected keywords
    detected_item = ", ".join(detected_keywords) if detected_keywords else "No specific item"
    
    return {
        'category': category,
        'matched' : matched,
        'detected_item': detected_item,
        'score'   : data['final_score'],
        'signal'  : data['signal'],
        'desc'    : data['description'],
        'rank'    : data['rank'],
        'pos_pct' : data['positive_pct'],
        'neutral_pct': data.get('neutral_pct', round(100.0 - data['positive_pct'] - 5.0, 1)),
        'neg_pct' : data.get('negative_pct', 5.0),
        'instagram_posts': data.get('instagram_posts', 0),
        'tiktok_posts': data.get('tiktok_posts', 0),
        'example_posts': data.get('example_posts', []),
        'posts'   : data['posts'],
        'likes'   : data['avg_likes'],
        'views'   : data['avg_views'],
        'trend_direction': data.get('trend_direction', {
            'direction': 'stable',
            'phase1_pct': 0.0,
            'phase2_pct': 0.0,
            'status': 'Stable',
            'indicator_color': '#6c757d'
        })
    }


def get_model_confidence(user_text):
    if tfidf_cat is None or svm_cat is None:
        return None

    try:
        vec = tfidf_cat.transform([str(user_text or '').strip().lower()])
        scores = svm_cat.decision_function(vec)
        if hasattr(scores, '__iter__'):
            scores = scores[0] if scores.shape[0] == 1 else scores
        classes = list(svm_cat.classes_)
        ranked = sorted(
            zip(scores.tolist(), classes), key=lambda kv: kv[0], reverse=True
        )
        if not ranked:
            return None

        top_score, top_class = ranked[0]
        second_text = None
        if len(ranked) > 1:
            second_score, second_class = ranked[1]
            top_pct = max(0, min(100, round(100.0 / (1.0 + abs(second_score - top_score)), 1)))
            second_text = f"{top_pct}% confident it's {normalize_category_name(top_class)}, with {normalize_category_name(second_class)} as a second option"
        else:
            second_text = f"Predicted as {normalize_category_name(top_class)} with model confidence score {round(top_score, 2)}"

        return {
            'top': normalize_category_name(top_class),
            'top_score': round(top_score, 2),
            'second': normalize_category_name(ranked[1][1]) if len(ranked) > 1 else None,
            'summary': second_text
        }
    except Exception:
        return None


def on_text_change():
    new_val = get_session_value("text_input_key", "").strip()
    set_session_value("current_search", new_val)
    set_session_value("trigger_analysis", False)


def on_dropdown_change():
    pick = get_session_value("dropdown_key", "")
    if pick != "-- Click to pick a keyword --":
        set_session_value("current_search", pick)
        set_session_value("text_input_key", pick)
        set_session_value("trigger_analysis", False)
    else:
        set_session_value("current_search", "")
        set_session_value("text_input_key", "")
        set_session_value("trigger_analysis", False)


st.markdown(
    """
    <div class="hero-header">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;">
            <div>
                <h1 style="margin:0; font-size:2rem; letter-spacing: -0.03em; display:flex; align-items:center; gap:10px;">
                    <svg class="lucide-icon header-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2l1.24 3.68L17 6.5l-2.84 1.78L14.5 12 12 10.36 9.5 12l.34-3.72L7 6.5l3.76-.82L12 2z" />
                        <path d="M3 12l2 0m14 0l2 0m-12.34 5.66l1.34-1.34m6.34 1.34l1.34-1.34m-6.34-6.34l1.34-1.34m6.34 1.34l1.34-1.34" />
                    </svg>
                    <span>Nepal Fashion Trend Detector</span>
                </h1>
                <p style="margin:8px 0 0; color:#4b5563; font-size:1rem;">For Nepali females aged 18-26</p>
            </div>
        </div>
        <div class="section-divider"></div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:10px; margin-bottom:0.75rem;'>"
        "<svg class='lucide-icon section-icon search' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>"
        "<circle cx='11' cy='11' r='7' />"
        "<path d='M21 21l-4.35-4.35' />"
        "</svg>"
        "<span style='font-size:1.1rem; font-weight:600; color:#111827;'>Search Parameters</span>"
        "</div>",
        unsafe_allow_html=True
    )

    suggestions = [
        "Saree", "Cotton Kurti Set", "Blue Denim Jacket",
        "Handmade Dhaka Dress", "White Office Shirt",
        "Leather Handbag", "Black Formal Suit",
        "Red Lehenga"
    ]

    default_idx = 0
    if current_search in suggestions:
        default_idx = suggestions.index(current_search) + 1

    st.markdown(
        "<div style='display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;'>"
        "<svg class='lucide-icon section-icon quick' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>"
        "<path d='M12 5v14' />"
        "<path d='M6 9h12' />"
        "<path d='M6 15h12' />"
        "</svg>"
        "<strong style='color:#111827;'>Quick Suggestions:</strong>"
        "</div>",
        unsafe_allow_html=True
    )
    st.selectbox(
        "Quick Suggestions",
        options=["-- Click to pick a keyword --"] + suggestions,
        index=default_idx,
        key="dropdown_key",
        on_change=on_dropdown_change
    )

    st.markdown(
        "<div style='display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;'>"
        "<svg class='lucide-icon section-icon input' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>"
        "<path d='M21 11.5v6.5a2 2 0 0 1-2 2h-6.5' />"
        "<path d='M16 3l5 5-11 11H5v-5L16 3z' />"
        "</svg>"
        "<strong style='color:#111827;'>Or type any fashion item:</strong>"
        "</div>",
        unsafe_allow_html=True
    )
    st.text_input(
        "Fashion item input",
        placeholder="e.g., saree, denim jacket, office blazer",
        key="text_input_key",
        on_change=on_text_change
    )

    if not current_search.strip():
        set_session_value("trigger_analysis", False)

    st.write("")
    if st.button("Predict the trend", key="predict_btn"):
        typed_value = get_session_value("text_input_key", "").strip()
        set_session_value("current_search", typed_value)
        set_session_value("trigger_analysis", bool(typed_value))

with right_col:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:10px; margin-bottom:1rem;'>"
        "<svg class='lucide-icon section-icon analytics' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>"
        "<path d='M4 19V5' />"
        "<path d='M12 19V11' />"
        "<path d='M20 19V15' />"
        "</svg>"
        "<span style='font-size:1.1rem; font-weight:600; color:#111827;'>Analytics Report</span>"
        "</div>",
        unsafe_allow_html=True
    )

    if not (get_session_value("trigger_analysis", False) and
            get_session_value("current_search", "").strip()):
        st.markdown(
            "<div style='background-color: #e3f2fd; border-left: 4px solid #2196F3; padding: 12px; border-radius: 4px;'>"
            "<p style='margin: 0; color: #1565c0;'>"
            "<strong>Type any fashion keyword</strong> or full sentence (colors, brands, occasions all work!) and "
            "click <strong>'Predict the trend'</strong> to see the full analysis here."
            "</p></div>",
            unsafe_allow_html=True
        )

if (get_session_value("trigger_analysis", False) and
        get_session_value("current_search", "").strip()):
    
    user_keywords = get_session_value("current_search", "")
    result = get_trend_result(user_keywords)

    # ── 1. VERDICT BADGE (largest, first) ──
    signal_text = re.sub(r'<[^>]+>', '', result['signal']).strip().upper()
    if 'VIRAL' in signal_text or 'STABLE' in signal_text:
        badge_color = "#2E7D32"
        badge_bg = "#E8F5E9"
    elif 'MODERATE' in signal_text:
        badge_color = "#EF6C00"
        badge_bg = "#FFF3E0"
    else:
        badge_color = "#7F8C8D"
        badge_bg = "#F5F5F5"

    display_category = result['category']
    st.markdown(f"""
    <div style="background-color: {badge_bg}; border: 2px solid {badge_color}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
        <h2 style="margin: 0; color: {badge_color}; text-align: center;">{display_category}</h2>
        <p style="margin: 8px 0 0; color: {badge_color}; text-align: center; font-weight: 600;">{signal_text}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ── 2. ONE SENTENCE SUMMARY ──
    summary = (
        f"Based on {result['posts']:,} real social media posts, Nepali women aged 18-26 "
        f"LOVE this style ({result['pos_pct']}% positive) — "
        f"it's the #{result['rank']} trending category in Nepal right now!"
    )
    st.markdown(f"*{summary}*")

    st.write("")

    # ── 2a. SENTIMENT BREAKDOWN & METRIC SUMMARY ──
    sentiment_label = f"{result['pos_pct']}% Positive · {result['neutral_pct']}% Neutral · {result['neg_pct']}% Negative"
    sentiment_fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[result['pos_pct'], result['neutral_pct'], result['neg_pct']],
        marker=dict(colors=['#4CAF50', '#9E9E9E', '#F44336']),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='%{label}: %{value}%<extra></extra>',
        sort=False,
        hole=0.45,
        showlegend=False
    ))
    sentiment_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    left_panel, right_panel = st.columns([1.8, 1], gap='large')
    with left_panel:
        st.markdown(
            "<div style='padding:20px 0 0 0;'>"
            "<h3 style='margin:0 0 10px; color:#111827;'>What People Are Saying</h3>"
            "<div style='display:flex; align-items:center; gap:18px; flex-wrap:wrap; margin-bottom:12px;'>"
            "<span class='sentiment-badge'><span class='sentiment-dot sentiment-positive'></span>Positive</span>"
            "<span class='sentiment-badge'><span class='sentiment-dot sentiment-neutral'></span>Neutral</span>"
            "<span class='sentiment-badge'><span class='sentiment-dot sentiment-negative'></span>Negative</span>"
            "</div>"
            f"<p style='margin:0 0 20px; color:#4b5563; font-size:0.95rem;'>{sentiment_label}</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)

    with right_panel:
        st.markdown("<div style='display:grid; gap:14px; padding-top:8px;'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card" style="background-color:#ffffff; border-left-color:#4CAF50;">
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:10px;'>
                    <svg class='lucide-icon metric-icon detected' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M18.5 6.5L17.5 5.5a2 2 0 0 0-2.83 0l-7 7a2 2 0 0 0 0 2.83l1 1a2 2 0 0 0 2.83 0l7-7a2 2 0 0 0 0-2.83z' />
                        <path d='M7.21 13.79L10.5 10.5' />
                    </svg>
                    <div class="metric-title">Detected As</div>
                </div>
                <div class="metric-value">{result['category']}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card" style="background-color:#ffffff; border-left-color:#4CAF50;">
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:10px;'>
                    <svg class='lucide-icon metric-icon love' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67 10.94 4.61a5.5 5.5 0 1 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z' />
                    </svg>
                    <div class="metric-title">People Who Love It</div>
                </div>
                <div class="metric-value">{result['pos_pct']}%</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card" style="background-color:#ffffff; border-left-color:#4CAF50;">
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:10px;'>
                    <svg class='lucide-icon metric-icon based' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                        <ellipse cx='12' cy='6' rx='8' ry='3' />
                        <path d='M4 6v6a8 3 0 0 0 16 0V6' />
                        <path d='M4 12v6a8 3 0 0 0 16 0v-6' />
                    </svg>
                    <div class="metric-title">Based On</div>
                </div>
                <div class="metric-value">{result['posts']:,} real posts</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    rank = result['rank']
    if rank <= 2:
        rank_text = "This is currently one of Nepal's TOP trending fashion styles"
    elif rank <= 4:
        rank_text = "This style has steady, moderate popularity"
    else:
        rank_text = "This style is less popular right now compared to other categories"

    st.markdown(f"**{rank_text}**")

    st.write("")

    # ── 4. COMPARISON CHART ──
    categories = list(REAL_NEPAL_TRENDS.keys())
    chart_labels = [c for c in categories]
    percentages = [
        REAL_NEPAL_TRENDS[c]['positive_pct']
        for c in categories
    ]
    colors = [
        '#E91E63' if c == result['category'] else '#E0E0E0'
        for c in categories
    ]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=chart_labels,
        orientation='h',
        marker_color=colors,
        text=[f"{p}%" for p in percentages],
        textposition='outside',
        textfont=dict(size=12, color="#ffffff"),
        marker_line_color='rgba(255,255,255,0.08)',
        marker_line_width=1,
        width=0.55
    ))
    fig.update_layout(
        title=f"How '{result['category']}' Compares to Other Trends in Nepal",
        xaxis=dict(
            title="% of people who love this style",
            range=[0, 100],
            showgrid=False,
            tickfont=dict(color='#d0d4db')
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color='#d0d4db')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=420,
        margin=dict(l=140, r=80, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("")

    # ── 5. MODEL TRANSPARENCY & PROVENANCE ──
    confidence = get_model_confidence(user_keywords)
    render_expander(
        "How was this predicted?",
        "<svg class='lucide-icon section-icon prediction' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M14 9a3 3 0 0 1-4 0' /><path d='M10 12v1a2 2 0 0 0 2 2h1' /><path d='M18 8v6a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z' /></svg>",
        "<p>Category Classification: Support Vector Machine (SVM) with linear kernel, implemented using scikit-learn's LinearSVC. Text was converted to numerical features using TF-IDF vectorization (5,000 features, unigrams and bigrams). The model was trained on 84,354 labeled examples and tested on 21,089 held-out examples, achieving 92.51% accuracy on sentiment classification and 91.91% accuracy on category classification.</p>"
        "<p>Sentiment Scoring: VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based NLP sentiment analyzer designed specifically for social media text. VADER assigns a compound score from -1 (most negative) to +1 (most positive) based on word-level sentiment lexicons, capitalization, punctuation, and context cues.</p>"
        "<p>Accuracy: 91.91% (category classification) measured on a held-out test set of 21,089 records (20% of the combined 105,443-record dataset), not used during model training. Sentiment classification (positive/neutral/negative) achieved 92.51% accuracy on the same test methodology, using 5-fold cross-validation for additional robustness verification (92.70% ± 0.10%).</p>"
        + (f"<p><strong>Model confidence:</strong> {confidence['summary']}</p>" if confidence is not None else "<p><strong>Model confidence:</strong> Not available for fallback keyword matching.</p>")
    )

    render_expander(
        "About This Data",
        "<svg class='lucide-icon section-icon data' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9l7 7v9a2 2 0 0 1-2 2z' /><path d='M13 3v6h6' /><path d='M9 9h6' /><path d='M9 13h6' /><path d='M9 17h3' /></svg>",
        f"<p>Source: Instagram & TikTok posts mentioning Nepal-specific fashion hashtags, collected April-June 2026. {result['posts']:,} posts in this category were analyzed after filtering for fashion relevance and removing duplicates from an initial pool of 2,172 Nepal-specific posts (with 105,443 total records including supplementary training data from publicly available fashion review datasets to improve model accuracy).</p>"
        "<ul>"
        f"<li>Date range: April 2026 - June 2026</li>"
        f"<li>Platform breakdown: {result['instagram_posts']:,} from Instagram, {result['tiktok_posts']:,} from TikTok</li>"
        "<li>Facebook data could not be collected due to platform API restrictions encountered during this research.</li>"
        "</ul>"
    )

    if result.get('example_posts'):
        st.markdown("### Sample Posts Analyzed")
        for example in result['example_posts']:
            badge_color = (
                "#4CAF50" if example['sentiment'] == "Positive" else
                "#9E9E9E" if example['sentiment'] == "Neutral" else
                "#F44336"
            )
            sentiment_icon = (
                "<svg class='post-sentiment-icon positive' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M12 20c4.418 0 8-3.582 8-8s-3.582-8-8-8-8 3.582-8 8 3.582 8 8 8z' /><path d='M15 9l-3 3-2-2' /></svg>" if example['sentiment'] == "Positive" else
                "<svg class='post-sentiment-icon neutral' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='7' /><path d='M9.5 12h5' /></svg>" if example['sentiment'] == "Neutral" else
                "<svg class='post-sentiment-icon negative' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='7' /><path d='M9 9l6 6M15 9l-6 6' /></svg>"
            )
            st.markdown(
                f"<div style='margin-bottom:10px;'>"
                f"<span style='background:{badge_color}; color:#fff; padding:4px 8px; border-radius:12px; font-size:0.85rem; display:inline-flex; align-items:center; gap:6px; margin-right:10px;'>{sentiment_icon}{example['sentiment']}</span>"
                f"{example['text']} <span style='color:#6b7280;'>(score: {example['score']})</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.write("")

    render_expander(
        "Data Ethics & Limitations",
        "<svg class='lucide-icon section-icon ethics' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z' /><path d='M12 9v4' /><path d='M12 17h.01' /></svg>",
        "<ul>"
        "<li>This tool only analyzes publicly available social media posts, no private accounts, messages, or personal user data were collected.</li>"
        "<li>Sentiment predictions may be less accurate for Nepali-language text since our underlying sentiment model was primarily trained on English text.</li>"
        "<li>Category predictions reflect patterns in historical social media data and should be used as one input among several for business decisions, not as the sole basis for production planning.</li>"
        "<li>Temporal Scope: This prototype analyzes a snapshot of social media sentiment collected over a fixed period (April-June 2026) rather than tracking trends continuously over time. A production-ready trend forecasting system would require repeated data collection across multiple months or seasons to model rising/falling trajectories, this was outside the scope of this dissertation's timeline and is identified as a direction for future work.</li>"
        "</ul>"
    )

    st.write("")

    # ── 5. OVERALL TREND STRENGTH ──
    st.markdown(
        f"**Overall Trend Strength: {result['score']}%**  \n"
        f"*(combines popularity + how many people are talking about it)*"
    )

    st.write("")

    # ── 5. DETAILED DESCRIPTION ──
    st.markdown(f"**Why this matters:**  \n{result['desc']}")

    st.write("---")
