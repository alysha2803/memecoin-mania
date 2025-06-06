import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("💬 Sentiment Analysis")

@st.cache_data
def load_twitter():
    df = pd.read_csv("data/memecoin_sentiment_results_twitter.csv", parse_dates=['date'])
    df['platform'] = 'Twitter'
    df['type'] = df.apply(lambda row: "DOGE" if row["mentions_dogecoin"] else 
                                       "SHIBA" if row["mentions_shiba inu"] else
                                       "PEPE" if row["mentions_pepe"] else
                                       "FLOKI" if row["mentions_floki"] else
                                       "BONK" if row["mentions_bonk"] else
                                       "WOJAK" if row["mentions_wojak"] else "OTHER", axis=1)
    return df

@st.cache_data
def load_reddit():
    df = pd.read_csv("data/sentiment_price_reddit.csv", parse_dates=['parsed_date'])
    df['platform'] = 'Reddit'
    df['type'] = df.apply(lambda row: "DOGE" if row["mentions_dogecoin"] else 
                                       "SHIBA" if row["mentions_shiba_inu"] else
                                       "PEPE" if row["mentions_pepe"] else
                                       "FLOKI" if row["mentions_floki"] else
                                       "BONK" if row["mentions_bonk"] else
                                       "WOJAK" if row["mentions_wojak"] else "OTHER", axis=1)
    df.rename(columns={'parsed_date': 'date'}, inplace=True)
    return df

df_twitter = load_twitter()
df_reddit = load_reddit()
combined_df = pd.concat([df_twitter, df_reddit])

# Filter
coin_options = ['All'] + sorted(combined_df['type'].unique())
selected_coin = st.selectbox("Select Memecoin", coin_options)

if selected_coin != 'All':
    filtered_df = combined_df[combined_df['type'] == selected_coin]
else:
    filtered_df = combined_df.copy()

# Tabbed interface
tab1, tab2 = st.tabs(["📊 Sentiment Distribution", "🔀 Platform Comparison"])

with tab1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    chart = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Overall Sentiment")
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Trend Over Time")
    trend = filtered_df.groupby(['date', 'sentiment_category']).size().reset_index(name='count')
    trend_chart = px.line(trend, x='date', y='count', color='sentiment_category', title="Sentiment Over Time")
    st.plotly_chart(trend_chart, use_container_width=True)

with tab2:
    st.subheader("Sentiment by Platform")
    platform_sentiment = filtered_df.groupby(['platform', 'sentiment_category']).size().reset_index(name='count')
    platform_chart = px.bar(platform_sentiment, x='platform', y='count', color='sentiment_category',
                            barmode='group', title="Sentiment Comparison: Reddit vs Twitter")
    st.plotly_chart(platform_chart, use_container_width=True)

# Add more insights
st.markdown("#### 🧠 Additional Stats")
st.write(filtered_df[['positive_score', 'negative_score', 'neutral_score', 'compound_score']].describe())
