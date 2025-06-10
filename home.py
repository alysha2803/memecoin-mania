import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Memecoin Mania",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    body {
    }
    .main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #111111, #333333, #555555, #777777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
}

    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background: #ffffff;
        border-left: 5px solid #45B7D1;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .highlight-box {
    background: linear-gradient(135deg, #d3d3d3 0%, #a9a9a9 100%);
    color: black; /* Use black for best contrast on grey */
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
}

}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">MEMECOIN MANIA</h1>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">By Farah, Alysha, Melissa & Hakim</p>', unsafe_allow_html=True)

# Introduction to Memecoins
st.markdown("""
<div class="highlight-box">
    <h2>💲 What Are Memecoins?</h2>
    <p>Memecoins are cryptocurrencies inspired by internet memes and online communities. Unlike traditional cryptocurrencies like Bitcoin or Ethereum, memecoins such as <strong>Dogecoin</strong> and <strong>Shiba Inu</strong> were created more for fun and community engagement than technical innovation.</p>
    <p>Memecoins thrive on hype, social media, and viral culture. Their prices are often highly volatile, influenced by trends, influencers, and speculative trading rather than intrinsic value.</p>
</div>
""", unsafe_allow_html=True)

# Memecoin Highlights
st.markdown("""
<div class="info-box">
    <h3> Dogecoin (DOGE)</h3>
    <ul>
        <li>Launched in 2013 as a joke based on Shiba Inu meme.</li>
        <li>Uses Proof-of-Work with fast 1-minute blocks.</li>
        <li>Massive online community; used for tipping and charity.</li>
        <li>Gained fame from Elon Musk’s tweets.</li>
    </ul>
</div>

<div class="info-box">
    <h3> Shiba Inu (SHIB)</h3>
    <ul>
        <li>Ethereum-based; designed as a Dogecoin rival.</li>
        <li>Created 1 quadrillion tokens; 50% sent to Vitalik Buterin.</li>
        <li>Vitalik donated $1B to COVID relief and burned 40%.</li>
        <li>Part of a larger ecosystem with Shibarium and ShibaSwap.</li>
    </ul>
</div>

<div class="info-box">
    <h3> Bonk (BONK)</h3>
    <ul>
        <li>Launched on Solana in Dec 2022 as a fair token model.</li>
        <li>50% supply airdropped to Solana community.</li>
        <li>Most held dog-themed token on Solana.</li>
        <li>Revitalized interest in the Solana ecosystem.</li>
    </ul>
</div>

<div class="info-box">
    <h3> PEPE</h3>
    <ul>
        <li>Based on Pepe the Frog meme, launched in 2023.</li>
        <li>Ethereum-based with no utility and zero tax.</li>
        <li>Became a top memecoin by community power.</li>
        <li>Symbolizes viral success from internet culture.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Popularity and Impact
st.markdown("""
<div class="highlight-box">
    <h3>🔥 Why Memecoins Are Popular</h3>
    <ul>
        <li>Driven by strong online communities (Reddit, Twitter).</li>
        <li>Celebrity endorsements (e.g., Elon Musk).</li>
        <li>Low-cost entry for new investors.</li>
        <li>Psychological FOMO from viral gains.</li>
        <li>Appeal of meme culture and humor.</li>
    </ul>
</div>

<div class="highlight-box">
    <h3>📉 Positive & Negative Impacts</h3>
    <b>Positive:</b>
    <ul>
        <li>Brings awareness to crypto.</li>
        <li>Encourages community engagement.</li>
        <li>Potential for quick profits.</li>
        <li>Introduces fun and innovation.</li>
    </ul>
    <b>Negative:</b>
    <ul>
        <li>High volatility and risk of loss.</li>
        <li>Lack of real-world utility.</li>
        <li>Susceptible to pump-and-dump schemes.</li>
        <li>Potential for regulatory challenges.</li>
    </ul>
</div>
""", unsafe_allow_html=True)



# Analysis Overview
st.markdown("## 📊 Analysis Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <h3 style="color: #FF6B6B;">10,000+</h3>
        <p>Data Points Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <h3 style="color: #4ECDC4;">6</h3>
        <p>Memecoins Tracked</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <h3 style="color: #45B7D1;">2</h3>
        <p>Social Platforms</p>
    </div>
    """, unsafe_allow_html=True)



# Navigation Section
st.markdown("## 🧭 Page Navigation")

nav_cols = st.columns(3)

with nav_cols[0]:
    st.button("📊 Data Overview", key="data_btn")
    st.button("📈 Trend Visualization", key="trend_btn")

with nav_cols[1]:
    st.button("💭 Sentiment Analysis", key="sentiment_btn")
    st.button("🔮 Predictive Modeling", key="predict_btn")

with nav_cols[2]:
    st.button("⚖️ Comparative Analysis", key="compare_btn")
    st.button("📝 Conclusion", key="conclude_btn")



