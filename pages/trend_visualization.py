import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Trend Analysis - Memecoin Mania",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Load Data -----

@st.cache_data
def load_price_data():
    df = pd.read_csv('data/merged_cleaned_price_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['avg_price'] = (df['high'] + df['low']) / 2
    return df

@st.cache_data
def load_sentiment_data():
    twitter = pd.read_csv('data/memecoin_sentiment_results_twitter.csv')
    reddit = pd.read_csv('data/memecoins_sentiments_results_reddit.xls')
    twitter['date'] = pd.to_datetime(twitter['date'])
    reddit['date'] = pd.to_datetime(reddit['date'])
    twitter.columns = [col.replace(' ', '_') for col in twitter.columns]
    reddit.columns = [col.replace(' ', '_') for col in reddit.columns]
    return twitter, reddit


def process_sentiment(df, platform):
    sentiments = {}
    for coin in ['dogecoin', 'shiba_inu', 'pepe', 'floki']:
        col = f'mentions_{coin}'
        if col in df.columns:
            coin_df = df[df[col] > 0].copy()
            daily = coin_df.groupby('date').agg({
                'compound_score': 'mean',
                'positive_score': 'mean',
                'neutral_score': 'mean',
                'negative_score': 'mean'
            }).reset_index()
            daily['coin'] = coin.capitalize()
            daily['platform'] = platform
            sentiments[coin] = daily
    return sentiments

# ----- Sidebar -----
col_names = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'avg_price': 'Average'
}

st.sidebar.title("🛠️ Customize View")
st.sidebar.markdown("### Price Trend Over Time")
selected_col = st.sidebar.selectbox("Select Price Type", list(col_names.keys()), format_func=lambda x: col_names[x])

st.sidebar.markdown("### Sentiment Over Time")
sentiment_source = st.sidebar.radio("Select Sentiment Source:", ["Twitter", "Reddit", "Both"])

st.sidebar.markdown("### Social Media Activity")
mention_coin = st.sidebar.selectbox(
    "Select Memecoin", 
    ['dogecoin', 'shiba_inu', 'pepe', 'floki', 'bonk', 'wojak'], 
    format_func=lambda x: x.title()
)


price_df = load_price_data()
twitter_df, reddit_df = load_sentiment_data()

twitter_sentiments = process_sentiment(twitter_df, 'Twitter')
reddit_sentiments = process_sentiment(reddit_df, 'Reddit')

def get_combined_sentiment(source):
    if source == "Twitter":
        return twitter_sentiments
    elif source == "Reddit":
        return reddit_sentiments
    else:
        combined = {}
        for coin in ['dogecoin', 'shiba_inu', 'pepe', 'floki', 'bonk', 'wojak']:
            t_df = twitter_sentiments.get(coin)
            r_df = reddit_sentiments.get(coin)
            if t_df is not None and r_df is not None:
                merged = pd.concat([t_df, r_df], ignore_index=True)
                merged = merged.groupby(['date', 'coin']).mean(numeric_only=True).reset_index()
                merged['platform'] = 'Both'
                combined[coin] = merged
        return combined

selected_sentiments = get_combined_sentiment(sentiment_source)

colors = {
    'dogecoin': 'blue',
    'shiba_inu': 'green',
    'pepe': 'orange',
    'floki': 'magenta',
    'bonk': 'cyan',
    'wojak': 'purple'
}

# ----- Main Dashboard -----

st.title("📊 Trend Analysis and Visualization")

# ----- Price Trend -----
st.markdown("## 📈 Price Trend Over Time")

price_fig = go.Figure()
for coin in price_df['type'].unique():
    coin_data = price_df[price_df['type'] == coin]
    price_fig.add_trace(go.Scatter(
        x=coin_data['date'], y=coin_data[selected_col],
        mode='lines', name=coin.upper()
    ))

price_fig.update_layout(
    title=f"{col_names[selected_col]} Price Over Time for Each Memecoin",
    xaxis_title="Date",
    yaxis_title=f"{col_names[selected_col]} (USD)",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(price_fig, use_container_width=True)

# ----- Combined Price Trend -----
combined_df = price_df.groupby('date').mean(numeric_only=True).reset_index()

fig_all_prices = go.Figure()
fig_all_prices.add_trace(go.Scatter(x=combined_df['date'], y=combined_df['open'], name='Open', line=dict(color='blue')))
fig_all_prices.add_trace(go.Scatter(x=combined_df['date'], y=combined_df['high'], name='High', line=dict(color='green')))
fig_all_prices.add_trace(go.Scatter(x=combined_df['date'], y=combined_df['low'], name='Low', line=dict(color='red')))
fig_all_prices.add_trace(go.Scatter(x=combined_df['date'], y=combined_df['avg_price'], name='Average', line=dict(color='orange', dash='dash')))

fig_all_prices.update_layout(
    title="Combined Price Trends (All Memecoins)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(fig_all_prices, use_container_width=True)

# ----- Sentiment Score Over Time -----
st.markdown("## 😊 Sentiment Score Over Time")

fig_sentiment = go.Figure()
for coin, sentiment_df in selected_sentiments.items():
    fig_sentiment.add_trace(go.Scatter(
        x=sentiment_df['date'],
        y=sentiment_df['compound_score'],
        mode='lines',
        name=coin.capitalize(),
        line=dict(color=colors.get(coin, 'black'))
    ))

fig_sentiment.update_layout(
    title=f"Daily Average Sentiment Score Over Time ({sentiment_source})",
    xaxis_title="Date",
    yaxis_title="Compound Sentiment Score",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(fig_sentiment, use_container_width=True)

# ----- Social Media Activity -----
st.markdown("## 📢 Social Media Activity Over Time")

mention_col = f'mentions_{mention_coin}'
twitter_mentions = twitter_df.groupby('date')[mention_col].sum().reset_index()
reddit_mentions = reddit_df.groupby('date')[mention_col].sum().reset_index()

combined_mentions = pd.merge(twitter_mentions, reddit_mentions, on='date', suffixes=('_twitter', '_reddit'))

fig_mentions = go.Figure()
fig_mentions.add_trace(go.Scatter(x=combined_mentions['date'], y=combined_mentions[f'{mention_col}_twitter'],
                                  mode='lines', name='Twitter Mentions', line=dict(color='red')))
fig_mentions.add_trace(go.Scatter(x=combined_mentions['date'], y=combined_mentions[f'{mention_col}_reddit'],
                                  mode='lines', name='Reddit Mentions', line=dict(color='blue', dash='dash')))

fig_mentions.update_layout(
    title=f"Daily Mentions of {mention_coin.title()} on Social Media",
    xaxis_title="Date",
    yaxis_title="Number of Mentions",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(fig_mentions, use_container_width=True)
