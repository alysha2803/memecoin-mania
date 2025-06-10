import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Volatility Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data():
    """Load and process cryptocurrency data with caching for fast loading"""
    try:
        # Read datasets
        memecoin_df = pd.read_csv("data/merged_cleaned_price_data.csv")
        traditional_df = pd.read_csv("data/cleaned_Traditional_price.csv")
        
        # Load sentiment data
        traditional_sentiment = pd.read_csv("data/Traditional Crypto Twitter Sentiment - cleaned.csv")
        memecoin_sentiment = pd.read_csv("data/memecoin_sentiment_results_twitter.csv")

        # Load Reddit sentiment data (add after Twitter sentiment loading)
        traditional_reddit_sentiment = pd.read_csv("data/traditional_reddit_data.csv")
        memecoin_reddit_sentiment = pd.read_csv("data/new_memecoin_sentiment_reddit_data.csv")
        
        # Normalize column names to lowercase
        memecoin_df.columns = [col.lower() for col in memecoin_df.columns]
        traditional_df.columns = [col.lower() for col in traditional_df.columns]
        
        # Rename 'price' to 'open' in traditional_df if needed
        if 'price' in traditional_df.columns and 'open' not in traditional_df.columns:
            traditional_df['open'] = traditional_df['price']
        
        # Convert 'date' columns to datetime
        memecoin_df['date'] = pd.to_datetime(memecoin_df['date'])
        traditional_df['date'] = pd.to_datetime(traditional_df['date'])
        traditional_sentiment['date'] = pd.to_datetime(traditional_sentiment['date'])
        memecoin_sentiment['date'] = pd.to_datetime(memecoin_sentiment['date'])
        traditional_reddit_sentiment['date'] = pd.to_datetime(traditional_reddit_sentiment['date'])
        memecoin_reddit_sentiment['date'] = pd.to_datetime(memecoin_reddit_sentiment['date'])
        
        # Clean 'open', 'high', 'low' in traditional_df (remove commas)
        for col in ['open', 'high', 'low']:
            if col in traditional_df.columns:
                traditional_df[col] = traditional_df[col].astype(str).str.replace(',', '').astype(float)
        
        # Convert 'volume' in traditional_df (e.g., '1.08M', '40.64K') to numeric
        def parse_volume(x):
            x = str(x).upper().strip()
            if 'NAN' in x or x == 'NAN':
                return np.nan
            if x.endswith('K'):
                return float(x[:-1]) * 1e3
            elif x.endswith('M'):
                return float(x[:-1]) * 1e6
            elif x.endswith('B'):
                return float(x[:-1]) * 1e9
            else:
                try:
                    return float(x.replace(',', ''))
                except:
                    return np.nan
        
        if 'volume' in traditional_df.columns:
            traditional_df['volume'] = traditional_df['volume'].apply(parse_volume)
        
        # Add type_group labels
        memecoin_df['type_group'] = 'memecoin'
        traditional_df['type_group'] = 'traditional'
        
        # Add individual coin names (assuming 'type' column contains coin names)
        if 'type' in memecoin_df.columns:
            memecoin_df['coin'] = memecoin_df['type'].str.upper()
        else:
            memecoin_df['coin'] = 'UNKNOWN'
            
        if 'type' in traditional_df.columns:
            traditional_df['coin'] = traditional_df['type'].str.upper()
        else:
            traditional_df['coin'] = 'UNKNOWN'
        
        # Combine both datasets
        combined_df = pd.concat([memecoin_df, traditional_df], ignore_index=True)
        
        # Convert all relevant columns to numeric
        for col in ['open', 'high', 'low', 'volume']:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Remove rows with invalid price data
        combined_df = combined_df.dropna(subset=['open', 'date'])
        combined_df = combined_df[combined_df['open'] > 0]
        
        # Sort by coin and date for proper time series calculation
        combined_df = combined_df.sort_values(['coin', 'date'])
        
        # Calculate daily returns per coin
        combined_df['daily_return'] = combined_df.groupby('coin')['open'].pct_change()
        
        # Remove extreme outliers (likely data errors)
        combined_df = combined_df[
            (combined_df['daily_return'].abs() <= 10) | 
            (combined_df['daily_return'].isna())
        ]
        
        # Calculate 7-day rolling volatility per coin
        combined_df['volatility_7d'] = (
            combined_df.groupby('coin')['daily_return']
            .rolling(window=7, min_periods=3)
            .std()
            .reset_index(level=0, drop=True)
        )
        
        # Cap volatility at reasonable levels
        combined_df.loc[combined_df['volatility_7d'] > 5, 'volatility_7d'] = np.nan
        
        # Process sentiment data
        processed_sentiment = process_sentiment_data(traditional_sentiment, memecoin_sentiment)

        processed_reddit_sentiment = process_sentiment_data(traditional_reddit_sentiment, memecoin_reddit_sentiment)
        
        return combined_df, processed_sentiment, processed_reddit_sentiment
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)    
def process_sentiment_data(traditional_sentiment, memecoin_sentiment):
    """Process sentiment data for interactive visualization"""
    
    # Define coin mention columns for traditional
    traditional_coins = {
        'BTC': 'mentions_bitcoin',
        'ETH': 'mentions_ethereum', 
        'LTC': 'mentions_litecoin',
        'ADA': 'mentions_cardano',
        'XRP': 'mentions_xrp',
        'DOT': 'mentions_dot'
    }
    
    # Define coin mention columns for memecoins
    memecoin_coins = {
        'DOGE': 'mentions_dogecoin',
        'SHIBA': 'mentions_shiba inu',
        'PEPE': 'mentions_pepe',
        'FLOKI': 'mentions_floki',
        'BONK': 'mentions_bonk',
        'WOJAK': 'mentions_wojak'
    }
    
    processed_data = []
    
    # Process traditional coins
    for coin, mention_col in traditional_coins.items():
        if mention_col in traditional_sentiment.columns:
            coin_data = traditional_sentiment[traditional_sentiment[mention_col] == True].copy()
            if len(coin_data) > 0:
                daily_sentiment = coin_data.set_index('date').resample('D')[
                    ['compound_score', 'positive_score', 'neutral_score', 'negative_score']
                ].mean().reset_index()
                daily_sentiment['coin'] = coin
                daily_sentiment['type_group'] = 'traditional'
                processed_data.append(daily_sentiment)
    
    # Process memecoins
    for coin, mention_col in memecoin_coins.items():
        if mention_col in memecoin_sentiment.columns:
            coin_data = memecoin_sentiment[memecoin_sentiment[mention_col] == True].copy()
            if len(coin_data) > 0:
                daily_sentiment = coin_data.set_index('date').resample('D')[
                    ['compound_score', 'positive_score', 'neutral_score', 'negative_score']
                ].mean().reset_index()
                daily_sentiment['coin'] = coin
                daily_sentiment['type_group'] = 'memecoin'
                processed_data.append(daily_sentiment)
    
    # Process aggregated data by type
    traditional_daily = traditional_sentiment.set_index('date').resample('D')[
        ['compound_score', 'positive_score', 'neutral_score', 'negative_score']
    ].mean().reset_index()
    traditional_daily['coin'] = 'ALL_TRADITIONAL'
    traditional_daily['type_group'] = 'traditional'
    processed_data.append(traditional_daily)
    
    memecoin_daily = memecoin_sentiment.set_index('date').resample('D')[
        ['compound_score', 'positive_score', 'neutral_score', 'negative_score']
    ].mean().reset_index()
    memecoin_daily['coin'] = 'ALL_MEMECOIN'
    memecoin_daily['type_group'] = 'memecoin'
    processed_data.append(memecoin_daily)
    
    if processed_data:
        return pd.concat(processed_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_sentiment_comparison_chart(sentiment_df, selected_coins=None, sentiment_type="compound_score"):
    """Create interactive sentiment comparison chart"""
    
    if selected_coins:
        df_filtered = sentiment_df[sentiment_df['coin'].isin(selected_coins)]
    else:
        df_filtered = sentiment_df
    
    fig = go.Figure()
    
    # Color mapping for different coin types
    color_map = {
        'traditional': px.colors.qualitative.Set1,
        'memecoin': px.colors.qualitative.Set2
    }
    
    coin_colors = {}
    traditional_count = 0
    memecoin_count = 0
    
    for coin in df_filtered['coin'].unique():
        coin_data = df_filtered[df_filtered['coin'] == coin]
        coin_type = coin_data['type_group'].iloc[0]
        
        if coin_type == 'traditional':
            coin_colors[coin] = color_map['traditional'][traditional_count % len(color_map['traditional'])]
            traditional_count += 1
        else:
            coin_colors[coin] = color_map['memecoin'][memecoin_count % len(color_map['memecoin'])]
            memecoin_count += 1
    
    for coin in df_filtered['coin'].unique():
        coin_data = df_filtered[df_filtered['coin'] == coin]
        coin_data_clean = coin_data.dropna(subset=[sentiment_type])
        
        if len(coin_data_clean) > 0:
            # Determine line style based on type
            line_style = dict(width=2) if coin_data['type_group'].iloc[0] == 'traditional' else dict(width=2, dash='dash')
            
            fig.add_trace(
                go.Scatter(
                    x=coin_data_clean['date'],
                    y=coin_data_clean[sentiment_type],
                    mode='lines',
                    name=f"{coin} ({'Traditional' if coin_data['type_group'].iloc[0] == 'traditional' else 'Memecoin'})",
                    line=line_style,
                    line_color=coin_colors[coin],
                    hovertemplate="<b>%{fullData.name}</b><br>" +
                                "Date: %{x}<br>" +
                                f"{sentiment_type.replace('_', ' ').title()}: %{{y:.4f}}<extra></extra>"
                )
            )
    
    # Add neutral line for compound score
    if sentiment_type == 'compound_score':
        fig.add_hline(y=0, line_dash="dot", line_color="black", 
                     annotation_text="Neutral Line", annotation_position="bottom right")
    
    fig.update_layout(
        title=f"Twitter Sentiment Analysis: {sentiment_type.replace('_', ' ').title()}",
        xaxis_title="Date",
        yaxis_title=sentiment_type.replace('_', ' ').title(),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    return fig

def create_sentiment_distribution_chart(sentiment_df, selected_coins=None, sentiment_type="compound_score"):
    """Create sentiment distribution comparison chart"""
    if selected_coins:
        df_filtered = sentiment_df[sentiment_df['coin'].isin(selected_coins)]
    else:
        df_filtered = sentiment_df
    
    fig = go.Figure()
    
    for coin in df_filtered['coin'].unique():
        coin_data = df_filtered[df_filtered['coin'] == coin]
        sentiment_data = coin_data[sentiment_type].dropna()
        
        if len(sentiment_data) > 0:
            fig.add_trace(
                go.Box(
                    y=sentiment_data,
                    name=f"{coin} ({'Traditional' if coin_data['type_group'].iloc[0] == 'traditional' else 'Memecoin'})",
                    boxpoints='outliers'
                )
            )
    
    fig.update_layout(
        title=f"Sentiment Distribution: {sentiment_type.replace('_', ' ').title()}",
        yaxis_title=sentiment_type.replace('_', ' ').title(),
        template='plotly_white',
        height=500
    )
    
    return fig

@st.cache_data(ttl=3600)
def calculate_sentiment_statistics(sentiment_df, selected_coins=None, sentiment_type="compound_score"):
    """Calculate sentiment statistics"""
    if selected_coins:
        df_filtered = sentiment_df[sentiment_df['coin'].isin(selected_coins)]
    else:
        df_filtered = sentiment_df
    
    # Group by type
    memecoin_sentiment = df_filtered[df_filtered['type_group'] == 'memecoin'][sentiment_type].dropna()
    traditional_sentiment = df_filtered[df_filtered['type_group'] == 'traditional'][sentiment_type].dropna()
    
    results = {}
    
    if len(memecoin_sentiment) > 0:
        results['memecoin'] = {
            'mean': memecoin_sentiment.mean(),
            'std': memecoin_sentiment.std(),
            'median': memecoin_sentiment.median(),
            'count': len(memecoin_sentiment)
        }
    
    if len(traditional_sentiment) > 0:
        results['traditional'] = {
            'mean': traditional_sentiment.mean(),
            'std': traditional_sentiment.std(),
            'median': traditional_sentiment.median(),
            'count': len(traditional_sentiment)
        }
    
    # T-test if both groups have data
    if len(memecoin_sentiment) > 0 and len(traditional_sentiment) > 0:
        t_stat, p_val = ttest_ind(memecoin_sentiment, traditional_sentiment, equal_var=False)
        results['t_test'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    return results

def create_volatility_comparison_chart(df, selected_coins=None, chart_type="both"):
    """Create interactive Plotly chart for volatility comparison"""
    
    if selected_coins:
        df_filtered = df[df['coin'].isin(selected_coins)]
    else:
        df_filtered = df
    
    # Create subplot figure
    if chart_type == "both":
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('7-Day Rolling Volatility', '7-Day Rolling volatility (Log Scale)'),
            vertical_spacing=0.08
        )
        
        # Regular scale plot
        for coin in df_filtered['coin'].unique():
            coin_data = df_filtered[df_filtered['coin'] == coin]
            coin_data_clean = coin_data.dropna(subset=['volatility_7d'])
            
            if len(coin_data_clean) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=coin_data_clean['date'],
                        y=coin_data_clean['volatility_7d'],
                        mode='lines',
                        name=f"{coin} ({'Memecoin' if coin_data['type_group'].iloc[0] == 'memecoin' else 'Traditional'})",
                        line=dict(width=2),
                        hovertemplate="<b>%{fullData.name}</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Volatility: %{y:.4f}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # Log scale plot
        for coin in df_filtered['coin'].unique():
            coin_data = df_filtered[df_filtered['coin'] == coin]
            coin_data_clean = coin_data[(coin_data['volatility_7d'] > 0) & (~coin_data['volatility_7d'].isna())]
            
            if len(coin_data_clean) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=coin_data_clean['date'],
                        y=coin_data_clean['volatility_7d'],
                        mode='lines',
                        name=f"{coin} ({'Memecoin' if coin_data['type_group'].iloc[0] == 'memecoin' else 'Traditional'})",
                        line=dict(width=2),
                        showlegend=False,
                        hovertemplate="<b>%{fullData.name}</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Volatility: %{y:.4f}<extra></extra>"
                    ),
                    row=2, col=1
                )
        
        fig.update_yaxes(type="log", row=2, col=1)
        fig.update_layout(height=800)
        
    else:
        # Single chart
        fig = go.Figure()
        
        for coin in df_filtered['coin'].unique():
            coin_data = df_filtered[df_filtered['coin'] == coin]
            coin_data_clean = coin_data.dropna(subset=['volatility_7d']) if chart_type == "linear" else coin_data[(coin_data['volatility_7d'] > 0) & (~coin_data['volatility_7d'].isna())]
            
            if len(coin_data_clean) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=coin_data_clean['date'],
                        y=coin_data_clean['volatility_7d'],
                        mode='lines',
                        name=f"{coin} ({'Memecoin' if coin_data['type_group'].iloc[0] == 'memecoin' else 'Traditional'})",
                        line=dict(width=2),
                        hovertemplate="<b>%{fullData.name}</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Volatility: %{y:.4f}<extra></extra>"
                    )
                )
        
        if chart_type == "log":
            fig.update_yaxes(type="log")
            
        fig.update_layout(height=600)
    
    # Update layout
    fig.update_layout(
        title="Cryptocurrency Volatility Analysis",
        xaxis_title="Date",
        yaxis_title="Volatility (Standard Deviation)",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    return fig

def create_volatility_distribution_chart(df, selected_coins=None):
    """Create distribution comparison chart"""
    if selected_coins:
        df_filtered = df[df['coin'].isin(selected_coins)]
    else:
        df_filtered = df
    
    fig = go.Figure()
    
    for coin in df_filtered['coin'].unique():
        coin_data = df_filtered[df_filtered['coin'] == coin]
        vol_data = coin_data['volatility_7d'].dropna()
        
        if len(vol_data) > 0:
            fig.add_trace(
                go.Box(
                    y=vol_data,
                    name=f"{coin} ({'Memecoin' if coin_data['type_group'].iloc[0] == 'memecoin' else 'Traditional'})",
                    boxpoints='outliers'
                )
            )
    
    fig.update_layout(
        title="Volatility Distribution by Cryptocurrency",
        yaxis_title="Volatility (Standard Deviation)",
        template='plotly_white',
        height=500
    )
    
    return fig

@st.cache_data(ttl=3600)
def calculate_statistics(df, selected_coins=None):
    """Calculate and display statistical analysis"""
    if selected_coins:
        df_filtered = df[df['coin'].isin(selected_coins)]
    else:
        df_filtered = df
    
    # Group by type
    memecoin_vol = df_filtered[df_filtered['type_group'] == 'memecoin']['volatility_7d'].dropna()
    traditional_vol = df_filtered[df_filtered['type_group'] == 'traditional']['volatility_7d'].dropna()
    
    results = {}
    
    if len(memecoin_vol) > 0:
        results['memecoin'] = {
            'mean': memecoin_vol.mean(),
            'std': memecoin_vol.std(),
            'median': memecoin_vol.median(),
            'count': len(memecoin_vol)
        }
    
    if len(traditional_vol) > 0:
        results['traditional'] = {
            'mean': traditional_vol.mean(),
            'std': traditional_vol.std(),
            'median': traditional_vol.median(),
            'count': len(traditional_vol)
        }
    
    # T-test if both groups have data
    if len(memecoin_vol) > 0 and len(traditional_vol) > 0:
        t_stat, p_val = ttest_ind(memecoin_vol, traditional_vol, equal_var=False)
        results['t_test'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    return results

@st.cache_data
def precompute_all_statistics(df, sentiment_df, reddit_sentiment_df):
    """Pre-compute all statistical combinations to avoid recalculation"""
    stats_cache = {}
    
    # Get all unique coins for each dataset
    all_coins = df['coin'].unique() if not df.empty else []
    sentiment_coins = sentiment_df['coin'].unique() if not sentiment_df.empty else []
    reddit_coins = reddit_sentiment_df['coin'].unique() if not reddit_sentiment_df.empty else []
    
    # Pre-compute volatility stats for common combinations
    coin_combinations = [
        None,  # All coins
        list(df[df['type_group'] == 'memecoin']['coin'].unique()) if not df.empty else [],
        list(df[df['type_group'] == 'traditional']['coin'].unique()) if not df.empty else []
    ]
    
    for coins in coin_combinations:
        key = f"vol_{str(coins)}"
        try:
            stats_cache[key] = calculate_statistics(df, coins)
        except:
            stats_cache[key] = {}
    
    # Pre-compute sentiment stats for each sentiment type
    sentiment_types = ['compound_score', 'positive_score', 'negative_score', 'neutral_score']
    
    for sentiment_type in sentiment_types:
        for coins in coin_combinations:
            # Twitter sentiment
            key = f"twitter_{sentiment_type}_{str(coins)}"
            try:
                stats_cache[key] = calculate_sentiment_statistics(sentiment_df, coins, sentiment_type)
            except:
                stats_cache[key] = {}
            
            # Reddit sentiment
            key = f"reddit_{sentiment_type}_{str(coins)}"
            try:
                stats_cache[key] = calculate_sentiment_statistics(reddit_sentiment_df, coins, sentiment_type)
            except:
                stats_cache[key] = {}
    
    return stats_cache

# Main Streamlit App
# Replace the main() function with this updated version:

def main():
    st.title("📈 Cryptocurrency Analysis Dashboard")
    st.markdown("Interactive analysis comparing volatility and sentiment between memecoins and traditional cryptocurrencies")
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Load data with progress bar
    progress_bar = st.progress(0)
    with st.spinner("Loading cryptocurrency data..."):
        progress_bar.progress(0.25)
        df, sentiment_df, reddit_sentiment_df = load_and_process_data()
        progress_bar.progress(0.75)
        
        # Pre-compute statistics
        if not df.empty:
            stats_cache = precompute_all_statistics(df, sentiment_df, reddit_sentiment_df)
            st.session_state.stats_cache = stats_cache
        progress_bar.progress(1.0)
    
    # Clear progress bar
    progress_bar.empty()
    
    if df.empty:
        st.error("No data available. Please check your CSV files.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Volatility Analysis", "🐦 Twitter Sentiment", "Reddit Sentiment"])
    
    with tab1:
        # Move all existing volatility analysis code here
        volatility_analysis_tab(df)
    
    with tab2:
        # Add sentiment analysis tab
        sentiment_analysis_tab(sentiment_df)

    with tab3:
        # Add Reddit sentiment analysis tab
        reddit_sentiment_analysis_tab(reddit_sentiment_df)

def get_cached_stats(cache_key, fallback_func, *args):
    """Get stats from cache or compute if not available"""
    if 'stats_cache' in st.session_state and cache_key in st.session_state.stats_cache:
        return st.session_state.stats_cache[cache_key]
    else:
        return fallback_func(*args)

def volatility_analysis_tab(df):
    """Volatility analysis tab content"""
    st.header("🌊 Volatility Analysis")
    
    # Sidebar filters for volatility
    st.sidebar.header("🔍 Volatility Filters")
    
    # Get available coins
    available_coins = sorted(df['coin'].unique())
    memecoin_list = sorted(df[df['type_group'] == 'memecoin']['coin'].unique())
    traditional_list = sorted(df[df['type_group'] == 'traditional']['coin'].unique())
    
    # Filter options
    filter_type = st.sidebar.selectbox(
        "Filter Type",
        ["All Coins", "Memecoins Only", "Traditional Only", "Custom Selection"],
        key="vol_filter"
    )
    
    selected_coins = None
    if filter_type == "Memecoins Only":
        selected_coins = memecoin_list
    elif filter_type == "Traditional Only":
        selected_coins = traditional_list
    elif filter_type == "Custom Selection":
        selected_coins = st.sidebar.multiselect(
            "Select Specific Coins",
            available_coins,
            default=available_coins[:5] if len(available_coins) > 5 else available_coins,
            key="vol_coins"
        )
    
    # Chart type selection
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["both", "linear", "log"],
        format_func=lambda x: {"both": "Both Linear & Log", "linear": "Linear Scale Only", "log": "Log Scale Only"}[x],
        key="vol_chart_type"
    )
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        key="vol_date_range"
    )
    
    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    else:
        df_filtered = df
    
    # Main content
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric(
            "Total Data Points",
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):,}" if len(df_filtered) != len(df) else None
        )
    
    with col2:
        memecoin_count = len(df_filtered[df_filtered['type_group'] == 'memecoin'])
        st.metric("Memecoin Data Points", f"{memecoin_count:,}")
    
    with col3:
        traditional_count = len(df_filtered[df_filtered['type_group'] == 'traditional'])
        st.metric("Traditional Data Points", f"{traditional_count:,}")
    
    # Main volatility chart
    st.subheader("🌊 Volatility Time Series")
    volatility_chart = create_volatility_comparison_chart(df_filtered, selected_coins, chart_type)
    st.plotly_chart(volatility_chart, use_container_width=True)
    
    # Distribution chart
    st.subheader("📊 Volatility Distribution")
    distribution_chart = create_volatility_distribution_chart(df_filtered, selected_coins)
    st.plotly_chart(distribution_chart, use_container_width=True)
    
    # Statistical Analysis
    st.subheader("📈 Statistical Analysis")
    #stats = calculate_statistics(df_filtered, selected_coins)

    cache_key = f"vol_{str(selected_coins)}"
    stats = get_cached_stats(cache_key, calculate_statistics, df_filtered, selected_coins)
    
    if stats:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'memecoin' in stats:
                st.subheader("🎭 Memecoins")
                st.metric("Mean Volatility", f"{stats['memecoin']['mean']:.6f}")
                st.metric("Median Volatility", f"{stats['memecoin']['median']:.6f}")
                st.metric("Std Deviation", f"{stats['memecoin']['std']:.6f}")
                st.metric("Data Points", f"{stats['memecoin']['count']:,}")
        
        with col2:
            if 'traditional' in stats:
                st.subheader("🏛️ Traditional Crypto")
                st.metric("Mean Volatility", f"{stats['traditional']['mean']:.6f}")
                st.metric("Median Volatility", f"{stats['traditional']['median']:.6f}")
                st.metric("Std Deviation", f"{stats['traditional']['std']:.6f}")
                st.metric("Data Points", f"{stats['traditional']['count']:,}")
        
        # T-test results
        if 't_test' in stats:
            st.subheader("🔬 Statistical Significance Test")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("T-Statistic", f"{stats['t_test']['t_statistic']:.4f}")
            
            with col2:
                st.metric("P-Value", f"{stats['t_test']['p_value']:.4f}")
            
            with col3:
                significance = "✅ Significant" if stats['t_test']['significant'] else "❌ Not Significant"
                st.metric("Result (p < 0.05)", significance)
            
            # Interpretation
            if 'memecoin' in stats and 'traditional' in stats:
                higher_vol = "Memecoins" if stats['memecoin']['mean'] > stats['traditional']['mean'] else "Traditional Crypto"
                st.info(f"**Conclusion:** {higher_vol} show higher average volatility. " + 
                       ("The difference is statistically significant." if stats['t_test']['significant'] 
                        else "The difference is not statistically significant."))

def sentiment_analysis_tab(sentiment_df):
    """Sentiment analysis tab content"""
    st.header("🐦 Twitter Sentiment")
    
    if sentiment_df.empty:
        st.error("No sentiment data available. Please check your sentiment CSV files.")
        return
    
    # Sidebar filters for sentiment
    st.sidebar.header("🔍 Sentiment Filters")
    
    # Get available coins for sentiment
    available_sentiment_coins = sorted(sentiment_df['coin'].unique())
    memecoin_sentiment_list = sorted(sentiment_df[sentiment_df['type_group'] == 'memecoin']['coin'].unique())
    traditional_sentiment_list = sorted(sentiment_df[sentiment_df['type_group'] == 'traditional']['coin'].unique())
    
    # Filter options for sentiment
    sentiment_filter_type = st.sidebar.selectbox(
        "Sentiment Filter Type",
        ["All Coins", "Memecoins Only", "Traditional Only", "Custom Selection"],
        key="sentiment_filter"
    )
    
    selected_sentiment_coins = None
    if sentiment_filter_type == "Memecoins Only":
        selected_sentiment_coins = memecoin_sentiment_list
    elif sentiment_filter_type == "Traditional Only":
        selected_sentiment_coins = traditional_sentiment_list
    elif sentiment_filter_type == "Custom Selection":
        selected_sentiment_coins = st.sidebar.multiselect(
            "Select Specific Coins",
            available_sentiment_coins,
            default=available_sentiment_coins[:6] if len(available_sentiment_coins) > 6 else available_sentiment_coins,
            key="sentiment_coins"
        )
    
    # Sentiment type selection
    sentiment_type = st.sidebar.selectbox(
        "Sentiment Metric",
        ["compound_score", "positive_score", "negative_score", "neutral_score"],
        format_func=lambda x: x.replace('_', ' ').title(),
        key="sentiment_type"
    )
    
    # Date range filter for sentiment
    min_sentiment_date = sentiment_df['date'].min().date()
    max_sentiment_date = sentiment_df['date'].max().date()
    
    sentiment_date_range = st.sidebar.date_input(
        "Sentiment Date Range",
        value=[min_sentiment_date, max_sentiment_date],
        min_value=min_sentiment_date,
        max_value=max_sentiment_date,
        key="sentiment_date_range"
    )
    
    # Filter sentiment data by date range
    if len(sentiment_date_range) == 2:
        start_date, end_date = sentiment_date_range
        sentiment_df_filtered = sentiment_df[(sentiment_df['date'].dt.date >= start_date) & (sentiment_df['date'].dt.date <= end_date)]
    else:
        sentiment_df_filtered = sentiment_df
    
    # Sentiment metrics
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric(
            "Total Sentiment Data Points",
            f"{len(sentiment_df_filtered):,}",
            delta=f"{len(sentiment_df_filtered) - len(sentiment_df):,}" if len(sentiment_df_filtered) != len(sentiment_df) else None
        )
    
    with col2:
        memecoin_sentiment_count = len(sentiment_df_filtered[sentiment_df_filtered['type_group'] == 'memecoin'])
        st.metric("Memecoin Sentiment Data", f"{memecoin_sentiment_count:,}")
    
    with col3:
        traditional_sentiment_count = len(sentiment_df_filtered[sentiment_df_filtered['type_group'] == 'traditional'])
        st.metric("Traditional Sentiment Data", f"{traditional_sentiment_count:,}")
    
    # Main sentiment chart
    st.subheader("📈 Sentiment Time Series")
    sentiment_chart = create_sentiment_comparison_chart(sentiment_df_filtered, selected_sentiment_coins, sentiment_type)
    st.plotly_chart(sentiment_chart, use_container_width=True)
    
    # Sentiment distribution chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_distribution_chart = create_sentiment_distribution_chart(sentiment_df_filtered, selected_sentiment_coins, sentiment_type)
    st.plotly_chart(sentiment_distribution_chart, use_container_width=True)
    
    # Sentiment Statistical Analysis
    st.subheader("📈 Sentiment Statistical Analysis")
    #sentiment_stats = calculate_sentiment_statistics(sentiment_df_filtered, selected_sentiment_coins, sentiment_type)
    
    sentiment_cache_key = f"vol_{str(selected_sentiment_coins)}"
    sentiment_stats = get_cached_stats(sentiment_cache_key, calculate_sentiment_statistics, sentiment_df_filtered, selected_sentiment_coins)
    if sentiment_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'memecoin' in sentiment_stats:
                st.subheader("🎭 Memecoin Sentiment")
                st.metric("Mean Sentiment", f"{sentiment_stats['memecoin']['mean']:.6f}")
                st.metric("Median Sentiment", f"{sentiment_stats['memecoin']['median']:.6f}")
                st.metric("Std Deviation", f"{sentiment_stats['memecoin']['std']:.6f}")
                st.metric("Data Points", f"{sentiment_stats['memecoin']['count']:,}")
        
        with col2:
            if 'traditional' in sentiment_stats:
                st.subheader("🏛️ Traditional Sentiment")
                st.metric("Mean Sentiment", f"{sentiment_stats['traditional']['mean']:.6f}")
                st.metric("Median Sentiment", f"{sentiment_stats['traditional']['median']:.6f}")
                st.metric("Std Deviation", f"{sentiment_stats['traditional']['std']:.6f}")
                st.metric("Data Points", f"{sentiment_stats['traditional']['count']:,}")
        
        # T-test results for sentiment
        if 't_test' in sentiment_stats:
            st.subheader("🔬 Sentiment Statistical Significance Test")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("T-Statistic", f"{sentiment_stats['t_test']['t_statistic']:.4f}")
            
            with col2:
                st.metric("P-Value", f"{sentiment_stats['t_test']['p_value']:.4f}")
            
            with col3:
                significance = "✅ Significant" if sentiment_stats['t_test']['significant'] else "❌ Not Significant"
                st.metric("Result (p < 0.05)", significance)
            
            # Interpretation for sentiment
            if 'memecoin' in sentiment_stats and 'traditional' in sentiment_stats:
                higher_sentiment = "Memecoins" if sentiment_stats['memecoin']['mean'] > sentiment_stats['traditional']['mean'] else "Traditional Crypto"
                st.info(f"**Conclusion:** {higher_sentiment} show higher average {sentiment_type.replace('_', ' ')}. " + 
                       ("The difference is statistically significant." if sentiment_stats['t_test']['significant'] 
                        else "The difference is not statistically significant."))
    
    # Sentiment data summary
    with st.expander("📋 Sentiment Data Summary"):
        st.write(f"**Sentiment Analysis Period:** {sentiment_df_filtered['date'].min().date()} to {sentiment_df_filtered['date'].max().date()}")
        st.write(f"**Available Cryptocurrencies for Sentiment:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Memecoins:**")
            for coin in memecoin_sentiment_list:
                st.write(f"- {coin}")
        
        with col2:
            st.write("**Traditional:**")
            for coin in traditional_sentiment_list:
                st.write(f"- {coin}")

def reddit_sentiment_analysis_tab(reddit_sentiment_df):
    """Reddit sentiment analysis tab content"""
    st.header("🔴 Reddit Sentiment")
    
    if reddit_sentiment_df.empty:
        st.error("No Reddit sentiment data available. Please check your Reddit sentiment CSV files.")
        return
    
    # Sidebar filters for Reddit sentiment
    st.sidebar.header("🔍 Reddit Sentiment Filters")
    
    # Get available coins for Reddit sentiment
    available_reddit_sentiment_coins = sorted(reddit_sentiment_df['coin'].unique())
    memecoin_reddit_sentiment_list = sorted(reddit_sentiment_df[reddit_sentiment_df['type_group'] == 'memecoin']['coin'].unique())
    traditional_reddit_sentiment_list = sorted(reddit_sentiment_df[reddit_sentiment_df['type_group'] == 'traditional']['coin'].unique())
    
    # Filter options for Reddit sentiment
    reddit_sentiment_filter_type = st.sidebar.selectbox(
        "Reddit Sentiment Filter Type",
        ["All Coins", "Memecoins Only", "Traditional Only", "Custom Selection"],
        key="reddit_sentiment_filter"
    )
    
    selected_reddit_sentiment_coins = None
    if reddit_sentiment_filter_type == "Memecoins Only":
        selected_reddit_sentiment_coins = memecoin_reddit_sentiment_list
    elif reddit_sentiment_filter_type == "Traditional Only":
        selected_reddit_sentiment_coins = traditional_reddit_sentiment_list
    elif reddit_sentiment_filter_type == "Custom Selection":
        selected_reddit_sentiment_coins = st.sidebar.multiselect(
            "Select Specific Coins",
            available_reddit_sentiment_coins,
            default=available_reddit_sentiment_coins[:6] if len(available_reddit_sentiment_coins) > 6 else available_reddit_sentiment_coins,
            key="reddit_sentiment_coins"
        )
    
    # Reddit sentiment type selection
    reddit_sentiment_type = st.sidebar.selectbox(
        "Reddit Sentiment Metric",
        ["compound_score", "positive_score", "negative_score", "neutral_score"],
        format_func=lambda x: x.replace('_', ' ').title(),
        key="reddit_sentiment_type"
    )
    
    # Date range filter for Reddit sentiment
    min_reddit_sentiment_date = reddit_sentiment_df['date'].min().date()
    max_reddit_sentiment_date = reddit_sentiment_df['date'].max().date()
    
    reddit_sentiment_date_range = st.sidebar.date_input(
        "Reddit Sentiment Date Range",
        value=[min_reddit_sentiment_date, max_reddit_sentiment_date],
        min_value=min_reddit_sentiment_date,
        max_value=max_reddit_sentiment_date,
        key="reddit_sentiment_date_range"
    )
    
    # Filter Reddit sentiment data by date range
    if len(reddit_sentiment_date_range) == 2:
        start_date, end_date = reddit_sentiment_date_range
        reddit_sentiment_df_filtered = reddit_sentiment_df[(reddit_sentiment_df['date'].dt.date >= start_date) & (reddit_sentiment_df['date'].dt.date <= end_date)]
    else:
        reddit_sentiment_df_filtered = reddit_sentiment_df
    
    # Reddit sentiment metrics
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric(
            "Total Reddit Sentiment Data Points",
            f"{len(reddit_sentiment_df_filtered):,}",
            delta=f"{len(reddit_sentiment_df_filtered) - len(reddit_sentiment_df):,}" if len(reddit_sentiment_df_filtered) != len(reddit_sentiment_df) else None
        )
    
    with col2:
        memecoin_reddit_sentiment_count = len(reddit_sentiment_df_filtered[reddit_sentiment_df_filtered['type_group'] == 'memecoin'])
        st.metric("Memecoin Reddit Sentiment Data", f"{memecoin_reddit_sentiment_count:,}")
    
    with col3:
        traditional_reddit_sentiment_count = len(reddit_sentiment_df_filtered[reddit_sentiment_df_filtered['type_group'] == 'traditional'])
        st.metric("Traditional Reddit Sentiment Data", f"{traditional_reddit_sentiment_count:,}")
    
    # Main Reddit sentiment chart
    st.subheader("📈 Reddit Sentiment Time Series")
    reddit_sentiment_chart = create_sentiment_comparison_chart(reddit_sentiment_df_filtered, selected_reddit_sentiment_coins, reddit_sentiment_type)
    st.plotly_chart(reddit_sentiment_chart, use_container_width=True)
    
    # Reddit sentiment distribution chart
    st.subheader("📊 Reddit Sentiment Distribution")
    reddit_sentiment_distribution_chart = create_sentiment_distribution_chart(reddit_sentiment_df_filtered, selected_reddit_sentiment_coins, reddit_sentiment_type)
    st.plotly_chart(reddit_sentiment_distribution_chart, use_container_width=True)
    
    # Reddit Sentiment Statistical Analysis
    st.subheader("📈 Reddit Sentiment Statistical Analysis")
    #reddit_sentiment_stats = calculate_sentiment_statistics(reddit_sentiment_df_filtered, selected_reddit_sentiment_coins, reddit_sentiment_type)

    reddit_sentiment_cache_key = f"vol_{str(selected_reddit_sentiment_coins)}"
    reddit_sentiment_stats = get_cached_stats(reddit_sentiment_cache_key, calculate_sentiment_statistics, reddit_sentiment_df_filtered, selected_reddit_sentiment_coins)
    
    if reddit_sentiment_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'memecoin' in reddit_sentiment_stats:
                st.subheader("🎭 Memecoin Reddit Sentiment")
                st.metric("Mean Reddit Sentiment", f"{reddit_sentiment_stats['memecoin']['mean']:.6f}")
                st.metric("Median Reddit Sentiment", f"{reddit_sentiment_stats['memecoin']['median']:.6f}")
                st.metric("Std Deviation", f"{reddit_sentiment_stats['memecoin']['std']:.6f}")
                st.metric("Data Points", f"{reddit_sentiment_stats['memecoin']['count']:,}")
        
        with col2:
            if 'traditional' in reddit_sentiment_stats:
                st.subheader("🏛️ Traditional Reddit Sentiment")
                st.metric("Mean Reddit Sentiment", f"{reddit_sentiment_stats['traditional']['mean']:.6f}")
                st.metric("Median Reddit Sentiment", f"{reddit_sentiment_stats['traditional']['median']:.6f}")
                st.metric("Std Deviation", f"{reddit_sentiment_stats['traditional']['std']:.6f}")
                st.metric("Data Points", f"{reddit_sentiment_stats['traditional']['count']:,}")
        
        # T-test results for Reddit sentiment
        if 't_test' in reddit_sentiment_stats:
            st.subheader("🔬 Reddit Sentiment Statistical Significance Test")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("T-Statistic", f"{reddit_sentiment_stats['t_test']['t_statistic']:.4f}")
            
            with col2:
                st.metric("P-Value", f"{reddit_sentiment_stats['t_test']['p_value']:.4f}")
            
            with col3:
                significance = "✅ Significant" if reddit_sentiment_stats['t_test']['significant'] else "❌ Not Significant"
                st.metric("Result (p < 0.05)", significance)
            
            # Interpretation for Reddit sentiment
            if 'memecoin' in reddit_sentiment_stats and 'traditional' in reddit_sentiment_stats:
                higher_reddit_sentiment = "Memecoins" if reddit_sentiment_stats['memecoin']['mean'] > reddit_sentiment_stats['traditional']['mean'] else "Traditional Crypto"
                st.info(f"**Conclusion:** {higher_reddit_sentiment} show higher average {reddit_sentiment_type.replace('_', ' ')} on Reddit. " + 
                       ("The difference is statistically significant." if reddit_sentiment_stats['t_test']['significant'] 
                        else "The difference is not statistically significant."))
    
    # Reddit sentiment data summary
    with st.expander("📋 Reddit Sentiment Data Summary"):
        st.write(f"**Reddit Sentiment Analysis Period:** {reddit_sentiment_df_filtered['date'].min().date()} to {reddit_sentiment_df_filtered['date'].max().date()}")
        st.write(f"**Available Cryptocurrencies for Reddit Sentiment:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Memecoins:**")
            for coin in memecoin_reddit_sentiment_list:
                st.write(f"- {coin}")
        
        with col2:
            st.write("**Traditional:**")
            for coin in traditional_reddit_sentiment_list:
                st.write(f"- {coin}")

if __name__ == "__main__":
    main()