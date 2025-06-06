import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Data Overview - Memecoin Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    
    .filter-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading function
@st.cache_data
def load_price_data():
    """Load and preprocess price data"""
    try:
        # Load the price data
        df = pd.read_csv('data/merged_cleaned_price_data.csv')
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    except Exception as e:
        st.error(f"Error loading price data: {e}")
        # Return sample data for demo
        return create_sample_price_data()

def create_sample_price_data():
    """Create sample data for demonstration"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    types = ['DOGE', 'PEPE', 'SHIB', 'FLOKI', 'BONK', 'WOJAK']
    
    data = []
    for coin_type in types:
        base_price = np.random.uniform(0.00001, 0.1)
        for date in dates:
            price = base_price * (1 + np.random.normal(0, 0.05))
            data.append({
                'date': date,
                'volume': np.random.uniform(1000000, 50000000),
                'open': price * (1 + np.random.uniform(-0.02, 0.02)),
                'high': price * (1 + np.random.uniform(0, 0.05)),
                'low': price * (1 + np.random.uniform(-0.05, 0)),
                'type': coin_type
            })
            base_price = price  # Slight price continuity
    
    return pd.DataFrame(data)

def format_number(num):
    """Format numbers to handle scientific notation properly"""
    if pd.isna(num):
        return "N/A"
    if abs(num) < 0.001:
        return f"{num:.2e}"
    elif abs(num) < 1:
        return f"{num:.6f}"
    elif abs(num) < 1000:
        return f"{num:.2f}"
    elif abs(num) < 1000000:
        return f"{num/1000:.1f}K"
    else:
        return f"{num/1000000:.1f}M"

def calculate_statistics(df, selected_types):
    """Calculate key statistics for selected memecoin types"""
    if selected_types and 'All' not in selected_types:
        df_filtered = df[df['type'].isin(selected_types)]
    else:
        df_filtered = df
    
    stats = {}
    
    if not df_filtered.empty:
        stats['total_records'] = len(df_filtered)
        stats['avg_volume'] = df_filtered['volume'].mean()
        stats['avg_price'] = df_filtered[['open', 'high', 'low']].mean().mean()
        stats['max_high'] = df_filtered['high'].max()
        stats['min_low'] = df_filtered['low'].min()
        stats['date_range'] = f"{df_filtered['date'].min().strftime('%Y-%m-%d')} to {df_filtered['date'].max().strftime('%Y-%m-%d')}"
        
        # Calculate volatility (using high-low spread)
        df_filtered['volatility'] = (df_filtered['high'] - df_filtered['low']) / df_filtered['open']
        stats['avg_volatility'] = df_filtered['volatility'].mean() * 100
        
    return stats

def create_price_chart(df, selected_types):
    """Create interactive price chart"""
    if selected_types and 'All' not in selected_types:
        df_filtered = df[df['type'].isin(selected_types)]
    else:
        df_filtered = df
    
    if df_filtered.empty:
        return None
    
    # Create subplots for price and volume
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Trends', 'Volume Trends'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, coin_type in enumerate(df_filtered['type'].unique()):
        coin_data = df_filtered[df_filtered['type'] == coin_type]
        color = colors[i % len(colors)]
        
        # Price chart (candlestick-style line)
        fig.add_trace(
            go.Scatter(
                x=coin_data['date'],
                y=coin_data['high'],
                mode='lines',
                name=f'{coin_type} Price',
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{coin_type}</b><br>' +
                             'Date: %{x}<br>' +
                             'Price: %{y:.8f}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Scatter(
                x=coin_data['date'],
                y=coin_data['volume'],
                mode='lines',
                name=f'{coin_type} Volume',
                line=dict(color=color, width=1),
                opacity=0.7,
                hovertemplate=f'<b>{coin_type}</b><br>' +
                             'Date: %{x}<br>' +
                             'Volume: %{y:,.0f}<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="Price and Volume Analysis",
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_volatility_chart(df, selected_types):
    """Create volatility comparison chart"""
    if selected_types and 'All' not in selected_types:
        df_filtered = df[df['type'].isin(selected_types)]
    else:
        df_filtered = df
    
    if df_filtered.empty:
        return None
    
    # Calculate daily volatility
    df_filtered = df_filtered.copy()
    df_filtered['volatility'] = ((df_filtered['high'] - df_filtered['low']) / df_filtered['open']) * 100
    
    # Create box plot for volatility comparison
    fig = px.box(
        df_filtered,
        x='type',
        y='volatility',
        title='Volatility Comparison Across Memecoins',
        labels={'volatility': 'Daily Volatility (%)', 'type': 'Memecoin Type'}
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white'
    )
    
    return fig

# Main page content
st.markdown('<h1 class="main-header">📊 Data Overview</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive analysis of memecoin price data, trading volumes, and market statistics**")

# Load data
with st.spinner("Loading price data..."):
    df = load_price_data()

if df is not None and not df.empty:
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🎯 Data Filters")
        
        # Memecoin type filter
        available_types = ['All'] + sorted(df['type'].unique().tolist())
        selected_types = st.multiselect(
            "Select Memecoin Types:",
            available_types,
            default=['All'],
            help="Choose specific memecoins to analyze or select 'All' for comprehensive view"
        )
        
        # Date range filter
        st.markdown("### 📅 Date Range")
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.date_input(
            "Select Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    
    # Statistics section
    st.markdown("## 📈 Key Statistics")
    
    stats = calculate_statistics(df, selected_types)
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FF6B6B; margin: 0;">{stats['total_records']:,}</h3>
                <p style="margin: 0;">Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #4ECDC4; margin: 0;">{format_number(stats['avg_volume'])}</h3>
                <p style="margin: 0;">Avg Volume</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #45B7D1; margin: 0;">{format_number(stats['avg_price'])}</h3>
                <p style="margin: 0;">Avg Price</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #96CEB4; margin: 0;">{stats['avg_volatility']:.2f}%</h3>
                <p style="margin: 0;">Avg Volatility</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="data-card">
                <h4>📊 Price Range</h4>
                <p><strong>Highest:</strong> {format_number(stats['max_high'])}</p>
                <p><strong>Lowest:</strong> {format_number(stats['min_low'])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="data-card">
                <h4>📅 Data Coverage</h4>
                <p>{stats['date_range']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Interactive charts
    st.markdown("## 📈 Interactive Visualizations")
    
    # Price and volume chart
    price_chart = create_price_chart(df, selected_types)
    if price_chart:
        st.plotly_chart(price_chart, use_container_width=True)
    
    # Volatility comparison
    volatility_chart = create_volatility_chart(df, selected_types)
    if volatility_chart:
        st.plotly_chart(volatility_chart, use_container_width=True)
    
    # Data table section
    st.markdown("## 📋 Raw Data Table")
    
    # Filter data for display
    if selected_types and 'All' not in selected_types:
        display_df = df[df['type'].isin(selected_types)]
    else:
        display_df = df
    
    # Format the dataframe for display
    display_df_formatted = display_df.copy()
    for col in ['volume', 'open', 'high', 'low']:
        if col in display_df_formatted.columns:
            display_df_formatted[col] = display_df_formatted[col].apply(format_number)
    
    # Display options
    col1, col2 = st.columns([3, 1])
    with col1:
        show_all = st.checkbox("Show all data", value=False)
    with col2:
        rows_to_show = st.selectbox("Rows per page:", [10, 25, 50, 100], index=1)
    
    if not show_all:
        display_df_formatted = display_df_formatted.head(rows_to_show)
    
    st.dataframe(
        display_df_formatted,
        use_container_width=True,
        height=400
    )
    
    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"memecoin_price_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.error("❌ Unable to load price data. Please check if the data file exists in the correct location.")
    st.info("💡 Expected file: `data/merged_cleaned_price_data.csv`")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📊 Data Overview | Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)