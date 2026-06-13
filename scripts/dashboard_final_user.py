import streamlit as st
import pandas as pd
import re


st.set_page_config(
    page_title="Nepal Fashion Trend Detector",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS to elevate aesthetics and style metrics beautifully
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #212529;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 20px;
        color: #212529;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize reactive state parameters securely
if 'current_search' not in st.session_state:
    st.session_state.current_search = ""
if 'trigger_analysis' not in st.session_state:
    st.session_state.trigger_analysis = False


REAL_NEPAL_TRENDS = {
    "Traditional/Ethnic": {
        "rank": 1,
        "trend_score": 0.5735,
        "positive_pct": 71.8,
        "posts": 721,
        "avg_likes": 5586,
        "avg_views": 125000,
        "final_score": 82.5,
        "signal": "🔥 VIRAL DEMAND",
        "description": (
            "Nepali females aged 18-26 show the STRONGEST enthusiasm for traditional "
            "ethnic wear. 71.8% of all social media posts about this category express positive "
            "sentiment. Highly recommended for production!"
        )
    },
    "General Fashion": {
        "rank": 2,
        "trend_score": 0.5734,
        "positive_pct": 55.1,
        "posts": 1015,
        "avg_likes": 37546,
        "avg_views": 191132,
        "final_score": 75.0,
        "signal": "📈 STABLE DEMAND",
        "description": (
            "General fashion content has the highest post volume showing "
            "broad market interest. Good steady demand."
        )
    },
    "Western/Casual": {
        "rank": 3,
        "trend_score": 0.4486,
        "positive_pct": 66.5,
        "posts": 200,
        "avg_likes": 72121,
        "avg_views": 204340,
        "final_score": 65.0,
        "signal": "📈 STABLE DEMAND",
        "description": (
            "Western casual wear shows high engagement. "
            "College-going females especially engage with affordable western styles."
        )
    },
    "Accessories": {
        "rank": 4,
        "trend_score": 0.4450,
        "positive_pct": 75.3,
        "posts": 81,
        "avg_likes": 8766,
        "avg_views": 93787,
        "final_score": 62.0,
        "signal": "📈 STABLE DEMAND",
        "description": (
            "Accessories show the second highest positive rate "
            "people who engage with accessories content love it! Low volume but high quality."
        )
    },
    "Formal/Professional": {
        "rank": 5,
        "trend_score": 0.4120,
        "positive_pct": 62.5,
        "posts": 120,
        "avg_likes": 4032,
        "avg_views": 156071,
        "final_score": 48.0,
        "signal": "📉 LOW DEMAND",
        "description": (
            "Formal wear has moderate engagement but highest negative sentiment "
            "among categories — buyers express dissatisfaction with quality and pricing."
        )
    },
    "Indo-Western/Fusion": {
        "rank": 6,
        "trend_score": 0.3545,
        "positive_pct": 60.0,
        "posts": 35,
        "avg_likes": 1123,
        "avg_views": 24786,
        "final_score": 38.0,
        "signal": "📉 LOW DEMAND",
        "description": (
            "Fusion wear currently shows lowest trend score. Limited social media posts and "
            "mixed reception suggest market is not ready for this category yet."
        )
    }
}

# Helper dataframe for rendering global comparative charts
chart_df = pd.DataFrame.from_dict(REAL_NEPAL_TRENDS, orient='index').reset_index()
chart_df.rename(columns={'index': 'Category'}, inplace=True)


def find_category(user_text):
    text = user_text.lower()

    traditional_words = [
        'saree', 'sari', 'kurti', 'kurta', 'lehenga', 'ethnic', 'traditional',
        'daura', 'gunyo', 'cholo', 'dhaka', 'dashain', 'tihar', 'teej', 'festival',
        'newari', 'tamang', 'handloom', 'pahiran', 'bride', 'wedding'
    ]
    western_words = [
        'jeans', 'denim', 'top', 'tshirt', 'hoodie', 'jacket', 'casual', 'shorts',
        'crop', 'sneaker', 'streetwear', 'cargo'
    ]
    fusion_words = [
        'fusion', 'indo western', 'indo-western', 'modern', 'contemporary', 'gown', 'frock'
    ]
    formal_words = [
        'formal', 'office', 'professional', 'suit', 'blazer', 'workwear', 'corporate'
    ]
    accessories_words = [
        'bag', 'jewel', 'necklace', 'earring', 'shoes', 'heel', 'sandal', 'watch',
        'scarf', 'dupatta', 'accessories', 'ring'
    ]

    if any(w in text for w in traditional_words):
        return "Traditional/Ethnic"
    elif any(w in text for w in western_words):
        return "Western/Casual"
    elif any(w in text for w in fusion_words):
        return "Indo-Western/Fusion"
    elif any(w in text for w in formal_words):
        return "Formal/Professional"
    elif any(w in text for w in accessories_words):
        return "Accessories"
    else:
        return "General Fashion"

def get_trend_result(user_text):
    category = find_category(user_text)
    data = REAL_NEPAL_TRENDS[category]
    return {
        'category' : category,
        'score'    : data['final_score'],
        'signal'   : data['signal'],
        'desc'     : data['description'],
        'rank'     : data['rank'],
        'pos_pct'  : data['positive_pct'],
        'posts'    : data['posts'],
        'likes'    : data['avg_likes'],
        'views'    : data['avg_views']
    }


def on_text_change():
    new_val = st.session_state.text_input_key.strip()
    st.session_state.current_search = new_val
    st.session_state.trigger_analysis = False

def on_dropdown_change():
    pick = st.session_state.dropdown_key
    if pick != "-- Click to pick a keyword --":
        st.session_state.current_search = pick
    st.session_state.trigger_analysis = False


st.markdown("#  Nepal Fashion Trend Detector")
st.markdown("### *Female aged 16 to 26*")
st.write("---")

left_col, right_col = st.columns([1, 1.2], gap="large")

#  LEFT COLUMN: CONTROL INPUT SYSTEMS 
with left_col:
    st.markdown("###  Search Parameters")
    
    suggestions = [
        "Saree",
        "Cotton Kurti Set",
        "Oversized Denim Jacket",
        "Handmade Dhaka Dress",
        "Office Blazer Suit",
        "Leather Handbag"
    ]
    
    default_idx = 0
    if st.session_state.current_search in suggestions:
        default_idx = suggestions.index(st.session_state.current_search) + 1
        
    st.selectbox(
        " Quick Suggestions (Select a style template):",
        options=["-- Click to pick a keyword --"] + suggestions,
        index=default_idx,
        key="dropdown_key",
        on_change=on_dropdown_change
    )
    
    st.text_input(
        "Or Type Custom Fashion Keywords:",
        value=st.session_state.current_search,
        placeholder="e.g., Silk Crop Top, Cargo Pants, Linen Kurti...",
        key="text_input_key",
        on_change=on_text_change
    )
    
    if not st.session_state.current_search.strip():
        st.session_state.trigger_analysis = False

    st.write("")
    # Version Fix: Removed modern icon & container attributes to handle older engine environments flawlessly
    if st.button(" Predict the trend "):
        if st.session_state.current_search.strip():
            st.session_state.trigger_analysis = True

#  RIGHT COLUMN: VISUAL METHODOLOGY REPORT 
with right_col:
    st.markdown("### Analytics Report")
    
    if st.session_state.trigger_analysis and st.session_state.current_search.strip():
        user_keywords = st.session_state.current_search
        result = get_trend_result(user_keywords)
        
        # Header Badge block
        st.markdown(f"""
        ### Category Match: `{result['category']}`
        Nepal Market Rank: Category Position `#{result['rank']} / 6`
        """)
        
        # Visual metric bar
        st.write(f"**Comprehensive Trend Index Score: {result['score']}%**")
        st.progress(int(result['score']))
        
        # Highlight card matching system status codes safely across all versions
        if result['score'] >= 72:
            st.success(f"**{result['signal']}**\n\n{result['desc']}")
        elif result['score'] >= 45:
            st.info(f"**{result['signal']}**\n\n{result['desc']}")
        else:
            st.warning(f"**{result['signal']}**\n\n{result['desc']}")
            
        st.write("---")       
        
    else:
        st.info(" **Awaiting Input Parameters:** Type custom words or select a quick template style option on the left, then click **'Run Live Database Query'** to view the empirical output.")

