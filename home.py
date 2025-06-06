import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Memecoin Mania Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #4ECDC4;
    }
    
    .nav-button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🚀 Memecoin Mania Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comprehensive Analysis of Cryptocurrency Market Trends & Sentiment</p>', unsafe_allow_html=True)

# Hero section with columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin: 2rem 0;">
        <h2>🎯 Welcome to the Ultimate Memecoin Analytics Platform</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Dive deep into the world of memecoins with our comprehensive data analysis platform. 
            Explore price trends, sentiment analysis, and predictive insights across multiple platforms.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Quick stats section
st.markdown("## 📊 Platform Overview")

col1, col2, col3, col4 = st.columns(4)

# Mock statistics - you can replace with actual data
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #FF6B6B;">10,000+</h3>
        <p>Data Points Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #4ECDC4;">6</h3>
        <p>Memecoins Tracked</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #45B7D1;">2</h3>
        <p>Social Platforms</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #96CEB4;">Real-time</h3>
        <p>Analysis Updates</p>
    </div>
    """, unsafe_allow_html=True)

# Features section
st.markdown("## 🎨 Platform Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📈 Data Analysis
    - **Comprehensive Price Data**: Track historical and real-time price movements
    - **Volume Analysis**: Monitor trading volumes across different memecoins  
    - **Statistical Insights**: Advanced metrics and trend analysis
    - **Interactive Filtering**: Analyze specific memecoins or compare all
    """)
    
    st.markdown("""
    ### 🤖 Machine Learning
    - **Predictive Modeling**: Forecast price movements using ML algorithms
    - **Pattern Recognition**: Identify recurring market patterns
    - **Risk Assessment**: Evaluate investment risks and opportunities
    """)

with col2:
    st.markdown("""
    ### 💭 Sentiment Analysis
    - **Multi-Platform Tracking**: Reddit and Twitter sentiment analysis
    - **Real-time Processing**: Live sentiment scoring using VADER
    - **Emotion Detection**: Positive, negative, and neutral sentiment classification
    - **Cross-Platform Comparison**: Compare sentiment across different social media
    """)
    
    st.markdown("""
    ### 📊 Visualization
    - **Interactive Charts**: Dynamic and responsive data visualizations
    - **Trend Analysis**: Time series analysis with filtering capabilities
    - **Comparative Views**: Side-by-side memecoin comparisons
    """)

# Navigation section
st.markdown("## 🧭 Navigation Guide")

st.markdown("""
<div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; margin: 2rem 0;">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #FF6B6B;">
            <h4>📊 Data Overview</h4>
            <p>Explore price data, trading volumes, and statistical insights across all tracked memecoins.</p>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4ECDC4;">
            <h4>💭 Sentiment Analysis</h4>
            <p>Analyze social media sentiment and compare emotions across platforms and cryptocurrencies.</p>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #45B7D1;">
            <h4>📈 Trend Visualization</h4>
            <p>Interactive time series analysis with advanced filtering and comparison capabilities.</p>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #96CEB4;">
            <h4>🔮 Predictive Modeling</h4>
            <p>Machine learning models for price prediction and market trend forecasting.</p>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #9b59b6;">
            <h4>⚖️ Comparison</h4>
            <p>Compare memecoins against traditional cryptocurrencies and market indices.</p>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #e74c3c;">
            <h4>📝 Conclusion</h4>
            <p>Summary of findings, insights, and future research directions.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick start section
st.markdown("## 🚀 Quick Start")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Start with Data Overview", key="data_btn"):
        st.switch_page("pages/data_overview.py")

with col2:
    if st.button("💭 Analyze Sentiment", key="sentiment_btn"):
        st.switch_page("pages/sentiment_analysis.py")

with col3:
    if st.button("📈 View Trends", key="trends_btn"):
        st.switch_page("pages/trend_visualization.py")

with col4:
    if st.button("🔮 Predict Prices", key="predict_btn"):
        st.switch_page("pages/predictive_modeling.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎯 Built with Streamlit | 📊 Data-Driven Insights | 🚀 Memecoin Analytics Platform</p>
    <p style="font-size: 0.9rem;">Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 🎮 Quick Actions")
    
    st.markdown("""
    ### 📋 Available Data
    - **DOGE**: Dogecoin data and sentiment
    - **PEPE**: Pepe coin analysis  
    - **SHIB**: Shiba Inu tracking
    - **FLOKI**: Floki coin insights
    - **BONK**: Bonk token data
    - **WOJAK**: Wojak coin analysis
    """)
    
    st.markdown("### 🎯 Key Metrics")
    st.info("Track price movements, volume changes, and sentiment shifts in real-time.")
    
    st.markdown("### 📱 Social Platforms")
    st.success("**Reddit** and **Twitter/X** sentiment analysis integrated.")
    
    st.markdown("### 🔄 Data Updates")
    st.warning("Data refreshes automatically. Check timestamps for latest updates.")