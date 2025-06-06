import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Predictive Modeling - Memecoin Mania",
    page_icon="🤖",
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
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E4057;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .model-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
    
    .comparison-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for model training and evaluation
@st.cache_data
def load_memecoin_data():
    """Load and prepare memecoin data"""
    try:
        # Load datasets (adjust paths as needed)
        datasets = {
            'rdoge': pd.read_csv('data/R Doge SentimentPrice.csv'),
            'rpepe': pd.read_csv('data/R Pepe SentimentPrice.csv'),
            'rshiba': pd.read_csv('data/R Shiba SentimentPrice.csv'),
            'rfloki': pd.read_csv('data/R Floki SentimentPrice.csv'),
            'tdoge': pd.read_csv('data/Sorted Sent+Price - Doge.csv'),
            'tpepe': pd.read_csv('data/Sorted Sent+Price - Pepe.csv'),
            'tshiba': pd.read_csv('data/Sorted Sent+Price - Shiba.csv'),
            'tfloki': pd.read_csv('data/Sorted Sent+Price - Floki.csv'),
        }

        coin_names = ['doge', 'pepe', 'shiba', 'floki']
        
        for coin in coin_names:
            # This is a placeholder - replace with your actual data loading
            # For demonstration, I'll create sample data structure
            dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
            n_samples = len(dates)
            
            # Generate realistic-looking sample data
            np.random.seed(42 + hash(coin) % 100)
            base_price = np.random.uniform(0.0001, 1.0)
            
            datasets[coin] = pd.DataFrame({
                'Date': dates,
                'compound_score': np.random.uniform(-0.5, 0.5, n_samples),
                'positive_score': np.random.uniform(0.1, 0.9, n_samples),
                'neutral_score': np.random.uniform(0.1, 0.8, n_samples),
                'negative_score': np.random.uniform(0.05, 0.3, n_samples),
                'Open': base_price * (1 + np.random.normal(0, 0.1, n_samples)).cumsum() / 100,
                'High': base_price * (1 + np.random.normal(0.01, 0.1, n_samples)).cumsum() / 100,
                'Low': base_price * (1 + np.random.normal(-0.01, 0.1, n_samples)).cumsum() / 100,
                'Close': base_price * (1 + np.random.normal(0, 0.1, n_samples)).cumsum() / 100,
            })
            
            # Ensure price relationships are logical
            datasets[coin]['High'] = np.maximum(datasets[coin]['High'], 
                                              np.maximum(datasets[coin]['Open'], datasets[coin]['Close']))
            datasets[coin]['Low'] = np.minimum(datasets[coin]['Low'], 
                                             np.minimum(datasets[coin]['Open'], datasets[coin]['Close']))
        
        return datasets
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}

def train_random_forest_model(df, target):
    """Train Random Forest model with GridSearchCV"""
    X = df[['compound_score']].values
    y = df[target].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Train/test split (non-shuffled for time series)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Predictions
    y_pred = grid_search.predict(X_test)
    y_full_pred = grid_search.predict(X_scaled)

    # Future forecast
    future_scores = np.linspace(-1, 1, 10).reshape(-1, 1)
    future_scaled = scaler_X.transform(future_scores)
    future_pred = grid_search.predict(future_scaled)

    return {
        "model": grid_search.best_estimator_,
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "predictions": y_pred,
        "actuals": y_test,
        "future_forecast": future_pred,
        "best_params": grid_search.best_params_,
        "full_predictions": y_full_pred,
        "full_actuals": y,
        "dates": df['Date'].values,
        "test_dates": df['Date'].iloc[split_idx:].values
    }

def train_svr_model(df, target):
    """Train SVR model with GridSearchCV"""
    X = df[['compound_score']].values
    y = df[target].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Train/test split
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    param_grid = {
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 0.2],
        "kernel": ["rbf"]
    }
    
    grid_search = GridSearchCV(SVR(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Predictions
    y_pred_scaled = grid_search.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Full predictions
    y_full_pred_scaled = grid_search.predict(X_scaled)
    y_full_pred = scaler_y.inverse_transform(y_full_pred_scaled.reshape(-1, 1)).ravel()

    # Future forecast
    future_scores = np.linspace(-1, 1, 10).reshape(-1, 1)
    future_scaled = scaler_X.transform(future_scores)
    future_pred_scaled = grid_search.predict(future_scaled)
    future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1)).ravel()

    return {
        "model": grid_search.best_estimator_,
        "mse": mean_squared_error(y_test_inv, y_pred),
        "r2": r2_score(y_test_inv, y_pred),
        "mae": mean_absolute_error(y_test_inv, y_pred),
        "predictions": y_pred,
        "actuals": y_test_inv,
        "future_forecast": future_pred,
        "best_params": grid_search.best_params_,
        "full_predictions": y_full_pred,
        "full_actuals": y,
        "dates": df['Date'].values,
        "test_dates": df['Date'].iloc[split_idx:].values
    }

def create_interactive_prediction_plot(results_rf, results_svr, coin, target):
    """Create interactive Plotly prediction plot"""
    # Prepare data
    dates_full = pd.to_datetime(results_rf['dates'])
    test_dates = pd.to_datetime(results_rf['test_dates'])
    
    # Future dates
    last_date = dates_full.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{coin.upper()} - {target} Price Prediction (Full Dataset)', 
                       f'{coin.upper()} - {target} Test Set Predictions'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Full dataset plot
    fig.add_trace(
        go.Scatter(x=dates_full, y=results_rf['full_actuals'], 
                  name='Actual Price', line=dict(color='black', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=dates_full, y=results_rf['full_predictions'], 
                  name='Random Forest', line=dict(color='#FF6B6B', width=2, dash='dot')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=dates_full, y=results_svr['full_predictions'], 
                  name='SVR', line=dict(color='#4ECDC4', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Future forecast
    fig.add_trace(
        go.Scatter(x=future_dates, y=results_rf['future_forecast'], 
                  name='RF Forecast', mode='markers+lines', 
                  marker=dict(color='#FF6B6B', size=8), line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=future_dates, y=results_svr['future_forecast'], 
                  name='SVR Forecast', mode='markers+lines', 
                  marker=dict(color='#4ECDC4', size=8), line=dict(color='#4ECDC4')),
        row=1, col=1
    )
    
    # Test set detailed view
    fig.add_trace(
        go.Scatter(x=test_dates, y=results_rf['actuals'], 
                  name='Test Actual', line=dict(color='black', width=3),
                  showlegend=False),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_dates, y=results_rf['predictions'], 
                  name='RF Test Pred', line=dict(color='#FF6B6B', width=2),
                  showlegend=False),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_dates, y=results_svr['predictions'], 
                  name='SVR Test Pred', line=dict(color='#4ECDC4', width=2),
                  showlegend=False),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=f"{coin.upper()} - {target} Price Prediction Analysis",
        title_x=0.5,
        template="plotly_white",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    
    return fig

def create_model_comparison_chart(results_rf, results_svr, coins, target):
    """Create model comparison chart"""
    metrics_data = []
    
    for coin in coins:
        if coin in results_rf and coin in results_svr:
            rf_res = results_rf[coin][target]
            svr_res = results_svr[coin][target]
            
            metrics_data.append({
                'Coin': coin.upper(),
                'Model': 'Random Forest',
                'R²': rf_res['r2'],
                'MSE': rf_res['mse'],
                'MAE': rf_res['mae']
            })
            
            metrics_data.append({
                'Coin': coin.upper(),
                'Model': 'SVR',
                'R²': svr_res['r2'],
                'MSE': svr_res['mse'],
                'MAE': svr_res['mae']
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('R² Score', 'Mean Squared Error', 'Mean Absolute Error'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = {'Random Forest': '#FF6B6B', 'SVR': '#4ECDC4'}
    
    for i, metric in enumerate(['R²', 'MSE', 'MAE'], 1):
        for model in ['Random Forest', 'SVR']:
            model_data = df_metrics[df_metrics['Model'] == model]
            fig.add_trace(
                go.Bar(x=model_data['Coin'], y=model_data[metric], 
                      name=model, marker_color=colors[model],
                      showlegend=(i == 1)),
                row=1, col=i
            )
    
    fig.update_layout(
        height=400,
        title=f"Model Performance Comparison - {target} Prediction",
        title_x=0.5,
        template="plotly_white",
        barmode='group'
    )
    
    return fig

def create_sentiment_price_correlation(data, coin):
    """Create sentiment-price correlation visualization"""
    fig = go.Figure()
    
    # Price trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        name='Close Price',
        line=dict(color='#2E86AB'),
        yaxis='y'
    ))
    
    # Sentiment trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['compound_score'],
        name='Sentiment Score',
        line=dict(color='#A23B72'),
        yaxis='y2'
    ))
    
    # Update layout for dual y-axis
    fig.update_layout(
        title=f"{coin.upper()} - Price vs Sentiment Analysis",
        xaxis_title="Date",
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="Sentiment Score", side="right", overlaying="y"),
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    return fig

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Predictive Modeling Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Machine Learning Models for Memecoin Price Prediction</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading memecoin data..."):
        datasets = load_memecoin_data()
    
    if not datasets:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar for controls
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Coin selection
    available_coins = list(datasets.keys())
    selected_coins = st.sidebar.multiselect(
        "Select Memecoins",
        options=available_coins,
        default=available_coins,
        help="Choose which memecoins to analyze"
    )
    
    # Target selection
    target_price = st.sidebar.selectbox(
        "Select Price Target",
        options=['Open', 'High', 'Low', 'Close'],
        index=3,
        help="Choose which price metric to predict"
    )
    
    # Model selection
    model_options = st.sidebar.multiselect(
        "Select Models",
        options=['Random Forest', 'SVR'],
        default=['Random Forest', 'SVR'],
        help="Choose which models to display"
    )
    
    if not selected_coins:
        st.warning("Please select at least one memecoin to analyze.")
        return
    
    # Train models
    st.markdown('<div class="sub-header">🏃‍♂️ Training Models...</div>', unsafe_allow_html=True)
    
    results_rf = {}
    results_svr = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_operations = len(selected_coins) * 2  # RF and SVR for each coin
    current_operation = 0
    
    for coin in selected_coins:
        df = datasets[coin].copy()
        df = df.sort_values('Date').dropna()
        
        # Train Random Forest
        if 'Random Forest' in model_options:
            status_text.text(f"Training Random Forest for {coin.upper()}...")
            results_rf[coin] = {target_price: train_random_forest_model(df, target_price)}
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
        
        # Train SVR
        if 'SVR' in model_options:
            status_text.text(f"Training SVR for {coin.upper()}...")
            results_svr[coin] = {target_price: train_svr_model(df, target_price)}
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
    
    progress_bar.empty()
    status_text.empty()
    
    # Display Results
    st.markdown('<div class="sub-header">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    # Metrics cards
    cols = st.columns(len(selected_coins))
    for i, coin in enumerate(selected_coins):
        with cols[i]:
            st.markdown(f"### {coin.upper()}")
            
            if 'Random Forest' in model_options and coin in results_rf:
                rf_r2 = results_rf[coin][target_price]['r2']
                rf_mse = results_rf[coin][target_price]['mse']
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🌲 Random Forest</h4>
                    <p>R² Score: {rf_r2:.4f}</p>
                    <p>MSE: {rf_mse:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if 'SVR' in model_options and coin in results_svr:
                svr_r2 = results_svr[coin][target_price]['r2']
                svr_mse = results_svr[coin][target_price]['mse']
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h4>🎯 SVR</h4>
                    <p>R² Score: {svr_r2:.4f}</p>
                    <p>MSE: {svr_mse:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Model Comparison Chart
    if len(model_options) == 2:
        st.markdown('<div class="sub-header">📈 Model Performance Comparison</div>', unsafe_allow_html=True)
        comparison_fig = create_model_comparison_chart(results_rf, results_svr, selected_coins, target_price)
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Individual Coin Analysis
    st.markdown('<div class="sub-header">🔍 Detailed Prediction Analysis</div>', unsafe_allow_html=True)
    
    for coin in selected_coins:
        st.markdown(f'<div class="model-section">', unsafe_allow_html=True)
        st.markdown(f"#### 🪙 {coin.upper()} Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🔗 Sentiment Correlation", "📋 Model Details"])
        
        with tab1:
            if coin in results_rf and coin in results_svr and len(model_options) == 2:
                pred_fig = create_interactive_prediction_plot(
                    results_rf[coin][target_price], 
                    results_svr[coin][target_price], 
                    coin, 
                    target_price
                )
                st.plotly_chart(pred_fig, use_container_width=True)
            else:
                st.info("Select both Random Forest and SVR models to see comparison plots.")
        
        with tab2:
            corr_fig = create_sentiment_price_correlation(datasets[coin], coin)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Correlation coefficient
            corr_coef = datasets[coin]['compound_score'].corr(datasets[coin]['Close'])
            st.metric("Sentiment-Price Correlation", f"{corr_coef:.4f}")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            if 'Random Forest' in model_options and coin in results_rf:
                with col1:
                    st.markdown("**🌲 Random Forest Parameters**")
                    rf_params = results_rf[coin][target_price]['best_params']
                    for param, value in rf_params.items():
                        st.write(f"• {param}: {value}")
            
            if 'SVR' in model_options and coin in results_svr:
                with col2:
                    st.markdown("**🎯 SVR Parameters**")
                    svr_params = results_svr[coin][target_price]['best_params']
                    for param, value in svr_params.items():
                        st.write(f"• {param}: {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary Section
    st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🎯 Key Insights & Recommendations</div>', unsafe_allow_html=True)
    
    if len(model_options) == 2 and results_rf and results_svr:
        # Calculate average performance
        avg_rf_r2 = np.mean([results_rf[coin][target_price]['r2'] for coin in selected_coins if coin in results_rf])
        avg_svr_r2 = np.mean([results_svr[coin][target_price]['r2'] for coin in selected_coins if coin in results_svr])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average RF R²", f"{avg_rf_r2:.4f}")
        with col2:
            st.metric("Average SVR R²", f"{avg_svr_r2:.4f}")
        with col3:
            better_model = "Random Forest" if avg_rf_r2 > avg_svr_r2 else "SVR"
            st.metric("Better Model", better_model)
        
        # Insights
        st.markdown("#### 💡 Model Performance Insights:")
        if avg_rf_r2 > avg_svr_r2:
            st.success("🌲 Random Forest shows superior performance on average, indicating better generalization for memecoin price prediction.")
        else:
            st.success("🎯 SVR demonstrates superior performance, suggesting better capture of non-linear sentiment-price relationships.")
        
        st.info("📊 Consider ensemble methods combining both models for optimal performance.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()