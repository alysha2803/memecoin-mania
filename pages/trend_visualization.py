import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA


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

    twitter.columns = twitter.columns.str.strip().str.lower().str.replace(' ', '_')
    reddit.columns = reddit.columns.str.strip().str.lower().str.replace(' ', '_')

    mention_cols = [col for col in twitter.columns if col.startswith('mentions_')]
    for col in mention_cols:
        twitter[col] = twitter[col].astype(int)
    mention_cols = [col for col in reddit.columns if col.startswith('mentions_')]
    for col in mention_cols:
        reddit[col] = reddit[col].astype(int)

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
            daily['coin'] = coin.lower()
            daily['platform'] = platform
            sentiments[coin.lower()] = daily
    return sentiments

def forecast_arima_series(series, steps=30):
    try:
        series = series.ffill().bfill().astype(float)
        model = ARIMA(series, order=(1, 0, 3))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast, model_fit
    except Exception as e:
        st.error(f"ARIMA error: {e}")
        return None, None



# ----- Sidebar -----
col_names = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'avg_price': 'Average'
}

ticker_to_name = {
    'DOGE': 'dogecoin',
    'SHIB': 'shiba_inu',
    'PEPE': 'pepe',
    'FLOKI': 'floki'
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

st.sidebar.markdown("### Correlation Analysis")
ticker = st.sidebar.selectbox("Select Coin to Correlate", ticker_to_name.keys(), key='correlation_coin')


# ----- Load Data -----
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

ticker_to_name = {
    'DOGE': 'dogecoin',
    'SHIB': 'shiba_inu',
    'PEPE': 'pepe',
    'FLOKI': 'floki'
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

st.markdown("## 🔄 Correlation: Price vs Sentiment (with Forecast)")

coin_name = ticker_to_name[ticker]
sentiment_df = selected_sentiments.get(coin_name)

price_coin_df = price_df[price_df['type'].str.upper() == ticker]

if sentiment_df is not None and not price_coin_df.empty:
    # Prepare price series
    price_df_daily = price_coin_df.groupby('date')['avg_price'].mean().reset_index()
    price_df_daily.set_index('date', inplace=True)
    price_series = price_df_daily['avg_price']

    # Prepare sentiment series
    sentiment_df = sentiment_df.copy()
    sentiment_df.set_index('date', inplace=True)
    sentiment_series = sentiment_df['compound_score']

    # Forecast both if enough data
    price_forecast, _ = forecast_arima_series(price_series, steps=90) if len(price_series) >= 50 else (None, None)
    sentiment_forecast, _ = forecast_arima_series(sentiment_series, steps=90) if len(sentiment_series) >= 50 else (None, None)

    fig_corr_forecast = go.Figure()

    # Historical price
    fig_corr_forecast.add_trace(go.Scatter(
        x=price_series.index, y=price_series,
        name='Avg Price (USD)', yaxis='y1', line=dict(color='blue')
    ))

    # Historical sentiment
    fig_corr_forecast.add_trace(go.Scatter(
        x=sentiment_series.index, y=sentiment_series,
        name='Sentiment Score', yaxis='y2', line=dict(color='orange')
    ))

    # Forecasted price
    if price_forecast is not None:
        future_price_dates = pd.date_range(start=price_series.index.max() + pd.Timedelta(days=1), periods=len(price_forecast), freq='D')
        fig_corr_forecast.add_trace(go.Scatter(
            x=future_price_dates, y=price_forecast,
            name='Price Forecast (ARIMA)', yaxis='y1',
            line=dict(color='blue', dash='dot')
        ))

    # Forecasted sentiment
    if sentiment_forecast is not None:
        future_sentiment_dates = pd.date_range(start=sentiment_series.index.max() + pd.Timedelta(days=1), periods=len(sentiment_forecast), freq='D')
        fig_corr_forecast.add_trace(go.Scatter(
            x=future_sentiment_dates, y=sentiment_forecast,
            name='Sentiment Forecast (ARIMA)', yaxis='y2',
            line=dict(color='orange', dash='dot')
        ))

    fig_corr_forecast.update_layout(
        title=f"📉 Price vs Sentiment Over Time for {coin_name.title()} (with Forecasts)",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price (USD)', side='left'),
        yaxis2=dict(title='Sentiment Score', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        hovermode='x unified'
    )

    st.plotly_chart(fig_corr_forecast, use_container_width=True)

else:
    st.warning("Sentiment or price data not available for selected coin.")

