import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")

# Title and description
st.title("Stock Portfolio Optimization System")
st.markdown("""
This application uses LSTM (Long Short-Term Memory) neural networks for stock prediction and
a Genetic Algorithm for portfolio optimization. The system provides:
- Historical stock data visualization
- Price predictions for the next 30 days
- Buy/Hold/Sell recommendations
- Optimized portfolio allocation
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # Stock selection
    stock_input = st.text_input("Enter stock symbols (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
    stock_list = [s.strip() for s in stock_input.split(',')]
    
    # Date range selection
    today = datetime.now()
    start_date = st.date_input("Start Date", today - timedelta(days=365*2))
    end_date = st.date_input("End Date", today)
    
    # LSTM parameters
    st.subheader("LSTM Model Parameters")
    lookback = st.slider("Lookback Period (days)", 7, 60, 30)
    epochs = st.slider("Training Epochs", 10, 100, 50)
    batch_size = st.slider("Batch Size", 8, 64, 32)
    
    # Genetic Algorithm parameters
    st.subheader("Genetic Algorithm Parameters")
    pop_size = st.slider("Population Size", 50, 200, 100)
    generations = st.slider("Generations", 10, 100, 30)
    
    # Risk preference
    risk_preference = st.slider("Risk Preference (Higher = More Risk)", 0.0, 1.0, 0.5)
    
    # Button to run analysis
    run_button = st.button("Run Analysis")

# Function to fetch stock data
@st.cache_data(ttl=3600)
def fetch_data(tickers, start, end):
    data = {}
    info = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(start=start, end=end)
            info[ticker] = {
                'name': stock.info.get('shortName', ticker),
                'sector': stock.info.get('sector', 'N/A'),
                'industry': stock.info.get('industry', 'N/A'),
                'marketCap': stock.info.get('marketCap', 'N/A'),
                'peRatio': stock.info.get('trailingPE', 'N/A'),
                'divYield': stock.info.get('dividendYield', 'N/A'),
                'beta': stock.info.get('beta', 'N/A'),
                '52weekHigh': stock.info.get('fiftyTwoWeekHigh', 'N/A'),
                '52weekLow': stock.info.get('fiftyTwoWeekLow', 'N/A'),
                'avgVolume': stock.info.get('averageVolume', 'N/A')
            }
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue
    return data, info

# Create LSTM model
def create_lstm_model(lookback):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Prepare data for LSTM
def prepare_data(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# Train LSTM and make predictions
def train_predict_lstm(stock_data, lookback, epochs, batch_size, forecast_days=30):
    # Get closing prices and normalize
    data = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Prepare training data
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    
    X_train, y_train = prepare_data(train_data, lookback)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Create and train model
    model = create_lstm_model(lookback)
    with st.spinner('Training LSTM model...'):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Make predictions on test data
    test_data = scaled_data[train_size - lookback:]
    X_test, y_test = prepare_data(test_data, lookback)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Calculate RMSE
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(np.mean(np.square(predictions - y_test_unscaled)))
    
    # Future predictions
    future_predictions = []
    curr_batch = scaled_data[-lookback:].reshape(1, lookback, 1)
    
    for _ in range(forecast_days):
        future_pred = model.predict(curr_batch)[0]
        future_predictions.append(future_pred)
        # Update batch for next prediction
        curr_batch = np.append(curr_batch[:, 1:, :], [[future_pred]], axis=1)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    # Create date range for future predictions
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    return predictions, future_predictions, future_dates, rmse

# Genetic Algorithm for portfolio optimization
def optimize_portfolio(stock_data, risk_preference, pop_size, generations):
    # Calculate daily returns
    returns = {}
    for ticker, data in stock_data.items():
        returns[ticker] = data['Close'].pct_change().dropna().values
    
    # Align return data lengths
    min_length = min(len(ret) for ret in returns.values())
    aligned_returns = np.array([returns[ticker][-min_length:] for ticker in stock_data.keys()])
    
    # Mean returns and covariance
    mean_returns = np.mean(aligned_returns, axis=1)
    cov_matrix = np.cov(aligned_returns)
    
    # Define genetic algorithm components
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # Attribute generator
    def random_weights():
        weights = [random.random() for _ in range(len(stock_data))]
        weights = [w/sum(weights) for w in weights]  # Normalize
        return weights
    
    # Structure initializers
    toolbox.register("weights", random_weights)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation function
    def evaluate(individual):
        portfolio_return = np.sum(mean_returns * individual)
        portfolio_risk = np.sqrt(np.dot(individual, np.dot(cov_matrix, individual)))
        # Balance return and risk according to risk preference
        return portfolio_return, (1 - risk_preference) * portfolio_risk
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    
    def checkBounds(min_val=0.0, max_val=1.0):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max_val:
                            child[i] = max_val
                        elif child[i] < min_val:
                            child[i] = min_val
                    # Normalize
                    total = sum(child)
                    child[:] = [i/total for i in child]
                return offspring
            return wrapper
        return decorator
    
    toolbox.register("select", tools.selNSGA2)
    toolbox.decorate("mate", checkBounds(0.0, 1.0))
    toolbox.decorate("mutate", checkBounds(0.0, 1.0))
    
    # Run the algorithm
    with st.spinner('Running genetic algorithm...'):
        population = toolbox.population(n=pop_size)
        hof = tools.ParetoFront()
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, 
                          ngen=generations, stats=stats, halloffame=hof, verbose=False)
    
    # Select best individual based on risk preference
    best_idx = 0
    if len(hof) > 1:
        returns = [ind.fitness.values[0] for ind in hof]
        risks = [ind.fitness.values[1] for ind in hof]
        # Normalize returns and risks
        max_return, min_return = max(returns), min(returns)
        max_risk, min_risk = max(risks), min(risks)
        
        norm_returns = [(r - min_return) / (max_return - min_return) if max_return > min_return else 0.5 for r in returns]
        norm_risks = [(r - min_risk) / (max_risk - min_risk) if max_risk > min_risk else 0.5 for r in risks]
        
        # Calculate utility based on risk preference
        utilities = [risk_preference * ret - (1 - risk_preference) * risk for ret, risk in zip(norm_returns, norm_risks)]
        best_idx = utilities.index(max(utilities))
    
    best_portfolio = list(hof[best_idx])
    portfolio_return = np.sum(mean_returns * best_portfolio)
    portfolio_risk = np.sqrt(np.dot(best_portfolio, np.dot(cov_matrix, best_portfolio)))
    
    # Annualize return and risk
    annual_return = (1 + portfolio_return) ** 252 - 1
    annual_risk = portfolio_risk * np.sqrt(252)
    
    return best_portfolio, annual_return, annual_risk

# Generate buy/hold/sell recommendation
def generate_recommendation(current_price, predicted_prices, historical_volatility):
    avg_pred_price = np.mean(predicted_prices)
    price_change_pct = (avg_pred_price - current_price) / current_price
    
    # Calculate confidence interval based on historical volatility
    daily_vol = historical_volatility / np.sqrt(252)
    price_range_low = current_price * (1 + price_change_pct - 1.96 * daily_vol * np.sqrt(30))
    price_range_high = current_price * (1 + price_change_pct + 1.96 * daily_vol * np.sqrt(30))
    
    # Decision thresholds (adjustable)
    buy_threshold = 0.05  # 5% expected increase
    sell_threshold = -0.03  # 3% expected decrease
    
    if price_change_pct > buy_threshold:
        action = "BUY"
        explanation = f"The stock is predicted to increase by {price_change_pct:.2%} over the next 30 days."
    elif price_change_pct < sell_threshold:
        action = "SELL"
        explanation = f"The stock is predicted to decrease by {price_change_pct:.2%} over the next 30 days."
    else:
        action = "HOLD"
        explanation = f"The stock is predicted to have minimal movement ({price_change_pct:.2%}) over the next 30 days."
    
    return {
        "action": action,
        "explanation": explanation,
        "avg_predicted_price": avg_pred_price,
        "price_change_pct": price_change_pct,
        "price_range_low": price_range_low,
        "price_range_high": price_range_high
    }

# Main analysis function
def run_analysis(stock_list, start_date, end_date, lookback, epochs, batch_size, 
                pop_size, generations, risk_preference):
    # Fetch data
    with st.spinner('Fetching stock data...'):
        stock_data, stock_info = fetch_data(stock_list, start_date, end_date)
    
    if not stock_data:
        st.error("No data available for the selected stocks.")
        return
    
    # Create tabs
    tabs = st.tabs(["Overview", "Stock Analysis", "Portfolio Optimization", "Recommendations"])
    
    # Overview tab
    with tabs[0]:
        st.header("Portfolio Overview")
        
        # Display stock information
        for ticker in stock_data:
            st.subheader(f"{stock_info[ticker]['name']} ({ticker})")
            
            # Create two columns for info display
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Sector:** {stock_info[ticker]['sector']}")
                st.write(f"**Industry:** {stock_info[ticker]['industry']}")
                st.write(f"**Market Cap:** {stock_info[ticker]['marketCap']:,}" if isinstance(stock_info[ticker]['marketCap'], (int, float)) else f"**Market Cap:** {stock_info[ticker]['marketCap']}")
                st.write(f"**P/E Ratio:** {stock_info[ticker]['peRatio']:.2f}" if isinstance(stock_info[ticker]['peRatio'], (int, float)) else f"**P/E Ratio:** {stock_info[ticker]['peRatio']}")
                st.write(f"**Dividend Yield:** {stock_info[ticker]['divYield']*100:.2f}%" if isinstance(stock_info[ticker]['divYield'], (int, float)) else f"**Dividend Yield:** {stock_info[ticker]['divYield']}")
            
            with col2:
                st.write(f"**Beta:** {stock_info[ticker]['beta']:.2f}" if isinstance(stock_info[ticker]['beta'], (int, float)) else f"**Beta:** {stock_info[ticker]['beta']}")
                st.write(f"**52-Week High:** ${stock_info[ticker]['52weekHigh']:.2f}" if isinstance(stock_info[ticker]['52weekHigh'], (int, float)) else f"**52-Week High:** {stock_info[ticker]['52weekHigh']}")
                st.write(f"**52-Week Low:** ${stock_info[ticker]['52weekLow']:.2f}" if isinstance(stock_info[ticker]['52weekLow'], (int, float)) else f"**52-Week Low:** {stock_info[ticker]['52weekLow']}")
                st.write(f"**Average Volume:** {stock_info[ticker]['avgVolume']:,}" if isinstance(stock_info[ticker]['avgVolume'], (int, float)) else f"**Average Volume:** {stock_info[ticker]['avgVolume']}")
                st.write(f"**Current Price:** ${stock_data[ticker]['Close'].iloc[-1]:.2f}")
            
            # Display performance metrics
            returns = stock_data[ticker]['Close'].pct_change().dropna()
            st.write(f"**Daily Returns (Mean):** {returns.mean()*100:.2f}%")
            st.write(f"**Daily Volatility:** {returns.std()*100:.2f}%")
            st.write(f"**Annual Volatility:** {returns.std()*np.sqrt(252)*100:.2f}%")
            
            # Historical price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data[ticker].index,
                y=stock_data[ticker]['Close'],
                mode='lines',
                name=f'{ticker} Close Price',
                line={'color': 'blue'}
            ))
            fig.update_layout(
                title=f'{ticker} Historical Price',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    # Stock Analysis tab
    with tabs[1]:
        st.header("Stock Analysis and Predictions")
        
        # Analyze each stock with LSTM
        predictions_data = {}
        
        for ticker in stock_data:
            st.subheader(f"{ticker} Price Prediction Analysis")
            
            # Train LSTM and get predictions
            with st.spinner(f'Training LSTM model for {ticker}...'):
                predictions, future_predictions, future_dates, rmse = train_predict_lstm(
                    stock_data[ticker], lookback, epochs, batch_size)
            
            # Store predictions for recommendations
            predictions_data[ticker] = {
                'current_price': stock_data[ticker]['Close'].iloc[-1],
                'future_predictions': future_predictions.flatten(),
                'historical_volatility': stock_data[ticker]['Close'].pct_change().std()
            }
            
            # Visualization of actual vs predicted prices
            test_start_idx = int(len(stock_data[ticker]) * 0.8)
            test_data = stock_data[ticker].iloc[test_start_idx:]
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Past data and predictions
            fig.add_trace(
                go.Scatter(
                    x=test_data.index,
                    y=test_data['Close'],
                    mode='lines',
                    name='Actual Price',
                    line={'color': 'blue'}
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=test_data.index[-len(predictions):],
                    y=predictions.flatten(),
                    mode='lines',
                    name='Predicted Price',
                    line={'color': 'red', 'dash': 'dash'}
                )
            )
            
            # Future predictions
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=future_predictions.flatten(),
                    mode='lines',
                    name='Future Prediction',
                    line={'color': 'green'}
                )
            )
            
            # Add confidence intervals for future predictions
            daily_vol = predictions_data[ticker]['historical_volatility'] / np.sqrt(252)
            upper_bound = [future_predictions[i][0] * (1 + 1.96 * daily_vol * np.sqrt(i+1)) for i in range(len(future_predictions))]
            lower_bound = [future_predictions[i][0] * (1 - 1.96 * daily_vol * np.sqrt(i+1)) for i in range(len(future_predictions))]
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    name='Upper Bound (95% CI)',
                    line={'color': 'rgba(0,128,0,0.2)'},
                    fill=None
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    name='Lower Bound (95% CI)',
                    line={'color': 'rgba(0,128,0,0.2)'},
                    fill='tonexty'
                )
            )
            
            fig.update_layout(
                title=f'{ticker} Price Prediction (RMSE: ${rmse:.2f})',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance metrics
            st.write(f"**RMSE (Root Mean Square Error):** ${rmse:.2f}")
            
            # Calculate MAPE
            actual = test_data['Close'].iloc[-len(predictions):].values
            mape = np.mean(np.abs((actual - predictions.flatten()) / actual)) * 100
            st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape:.2f}%")
            
            # Future price statistics
            st.write("### 30-Day Price Forecast")
            st.write(f"**Starting Price:** ${predictions_data[ticker]['current_price']:.2f}")
            st.write(f"**Average Predicted Price:** ${np.mean(future_predictions):.2f}")
            st.write(f"**Predicted Price Range:** ${np.min(lower_bound):.2f} - ${np.max(upper_bound):.2f}")
            st.write(f"**Predicted Change:** {((np.mean(future_predictions) - predictions_data[ticker]['current_price']) / predictions_data[ticker]['current_price'] * 100):.2f}%")
            
            st.markdown("---")
    
    # Portfolio Optimization tab
    with tabs[2]:
        st.header("Portfolio Optimization")
        
        # Run genetic algorithm for portfolio optimization
        with st.spinner('Optimizing portfolio allocation...'):
            optimal_weights, expected_return, expected_risk = optimize_portfolio(
                stock_data, risk_preference, pop_size, generations)
        
        # Display optimization results
        st.subheader("Optimal Portfolio Allocation")
        
        # Prepare data for visualization
        allocation_data = {
            'Stock': list(stock_data.keys()),
            'Allocation': [f"{w*100:.2f}%" for w in optimal_weights],
            'Weight': optimal_weights
        }
        
        # Show allocation table
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df[['Stock', 'Allocation']])
        
        # Pie chart of allocations
        fig = go.Figure(data=[go.Pie(
            labels=allocation_df['Stock'],
            values=allocation_df['Weight'],
            hoverinfo='label+percent',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='Portfolio Allocation',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio performance metrics
        st.subheader("Expected Portfolio Performance")
        st.write(f"**Expected Annual Return:** {expected_return*100:.2f}%")
        st.write(f"**Expected Annual Risk (Volatility):** {expected_risk*100:.2f}%")
        st.write(f"**Sharpe Ratio (Rf=0%):** {(expected_return/expected_risk):.2f}")
        
        # Efficient frontier simulation
        st.subheader("Efficient Frontier Analysis")
        
        # Simulate different portfolios
        num_portfolios = 1000
        returns = []
        volatilities = []
        
        # Calculate returns and volatilities for random portfolios
        for _ in range(num_portfolios):
            weights = np.random.random(len(stock_data))
            weights /= np.sum(weights)
            
            # Calculate returns and volatility
            returns_data = np.array([stock_data[ticker]['Close'].pct_change().dropna().values 
                             for ticker in stock_data.keys()])
            min_length = min(len(ret) for ret in returns_data)
            aligned_returns = np.array([ret[-min_length:] for ret in returns_data])
            
            mean_returns = np.mean(aligned_returns, axis=1)
            cov_matrix = np.cov(aligned_returns)
            
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Annualize
            returns.append((1 + portfolio_return) ** 252 - 1)
            volatilities.append(portfolio_volatility * np.sqrt(252))
        
        # Plot efficient frontier
        fig = go.Figure()
        
        # Add random portfolios
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(100, 100, 100, 0.5)'
            ),
            name='Random Portfolios'
        ))
        
        # Add optimal portfolio
        fig.add_trace(go.Scatter(
            x=[expected_risk],
            y=[expected_return],
            mode='markers',
            marker=dict(
                size=15,
                color='red'
            ),
            name='Optimal Portfolio'
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Annualized Risk (Volatility)',
            yaxis_title='Annualized Return',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations tab
    with tabs[3]:
        st.header("Investment Recommendations")
        
        # Generate recommendations for each stock
        recommendations = {}
        for ticker in predictions_data:
            recommendations[ticker] = generate_recommendation(
                predictions_data[ticker]['current_price'],
                predictions_data[ticker]['future_predictions'],
                predictions_data[ticker]['historical_volatility']
            )
        
        # Display recommendations
        for ticker in recommendations:
            rec = recommendations[ticker]
            
            # Set color based on recommendation
            if rec['action'] == 'BUY':
                color = 'green'
            elif rec['action'] == 'SELL':
                color = 'red'
            else:
                color = 'orange'
            
            st.markdown(f"### {ticker}: <span style='color:{color};font-weight:bold'>{rec['action']}</span>", unsafe_allow_html=True)
            
            # Current price and prediction
            st.write(f"**Current Price:** ${predictions_data[ticker]['current_price']:.2f}")
            st.write(f"**Average Predicted Price (30 days):** ${rec['avg_predicted_price']:.2f}")
            st.write(f"**Expected Change:** {rec['price_change_pct']*100:.2f}%")
            
            # Price range
            st.write(f"**Predicted Price Range (95% confidence):**")
            st.write(f"${rec['price_range_low']:.2f} to ${rec['price_range_high']:.2f}")
            
            # Explanation
            st.write(f"**Analysis:** {rec['explanation']}")
            
            # Visualize prediction
            fig = go.Figure()
            
            # Current price line
            fig.add_hline(
                y=predictions_data[ticker]['current_price'],
                line_width=2,
                line_dash="dash",
                line_color="blue",
                annotation_text="Current Price",
                annotation_position="bottom right"
            )
            
            # Prediction range
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=rec['avg_predicted_price'],
                title={'text': f"{ticker} 30-Day Price Forecast"},
                gauge={
                    'axis': {'range': [rec['price_range_low']*0.9, rec['price_range_high']*1.1]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [rec['price_range_low'], rec['price_range_high']], 'color': 'lightgray'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': predictions_data[ticker]['current_price']
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
        
        # Overall portfolio recommendation
        st.subheader("Portfolio Strategy Recommendation")
        
        # Count recommendations
        buy_count = sum(1 for rec in recommendations.values() if rec['action'] == 'BUY')
        sell_count = sum(1 for rec in recommendations.values() if rec['action'] == 'SELL')
        hold_count = sum(1 for rec in recommendations.values() if rec['action'] == 'HOLD')
        
        # Generate overall strategy
        if buy_count > sell_count and buy_count > hold_count:
            strategy = "Increase investment in the portfolio"
            explanation = "The majority of stocks in your portfolio are expected to rise. Consider adding to your positions."
        elif sell_count > buy_count and sell_count > hold_count:
            strategy = "Reduce exposure to the portfolio"
            explanation = "The majority of stocks in your portfolio are expected to decline. Consider taking profits or cutting losses."
        else:
            strategy = "Maintain current positions"
            explanation = "The majority of stocks in your portfolio are expected to remain stable. Hold your current positions."
        
        st.markdown(f"### Overall Strategy: **{strategy}**")
        st.write(explanation)
        
        # Show recommendation distribution
        rec_data = {
            'Recommendation': ['BUY', 'HOLD', 'SELL'],
            'Count': [buy_count, hold_count, sell_count]
        }
        
        # Create recommendation distribution chart
        fig = go.Figure(data=[go.Bar(
            x=rec_data['Recommendation'],
            y=rec_data['Count'],
            marker_color=['green', 'orange', 'red']
        )])
        
        fig.update_layout(
            title='Recommendation Distribution',
            xaxis_title='Recommendation',
            yaxis_title='Number of Stocks',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Investment plan based on optimization
        st.subheader("Investment Plan")
        st.write("""
        Based on our portfolio optimization and stock predictions, we recommend the following investment plan:
        """)
        
        # Create allocation recommendation
        allocation_rec = []
        for ticker, weight in zip(stock_data.keys(), optimal_weights):
            current_rec = recommendations[ticker]['action']
            
            if current_rec == 'BUY':
                suggested_change = "Increase allocation"
                action = f"Buy more {ticker}"
            elif current_rec == 'SELL':
                suggested_change = "Decrease allocation"
                action = f"Reduce {ticker} position"
            else:
                suggested_change = "Maintain allocation"
                action = f"Hold {ticker}"
            
            allocation_rec.append({
                'Stock': ticker,
                'Optimal Weight': f"{weight*100:.2f}%",
                'Recommendation': current_rec,
                'Suggested Change': suggested_change,
                'Action': action
            })
        
        # Display investment plan table
        st.table(pd.DataFrame(allocation_rec))
        
        # Risk assessment
        st.subheader("Risk Assessment")
        
        # Calculate portfolio beta
        portfolio_beta = 0
        for ticker, weight in zip(stock_data.keys(), optimal_weights):
            stock_beta = stock_info[ticker]['beta']
            if isinstance(stock_beta, (int, float)):
                portfolio_beta += weight * stock_beta
            
        if isinstance(portfolio_beta, (int, float)):
            st.write(f"**Portfolio Beta:** {portfolio_beta:.2f}")
            
            if portfolio_beta > 1.2:
                st.write("Your portfolio has **high market sensitivity**. It's likely to amplify market movements in both directions.")
            elif portfolio_beta < 0.8:
                st.write("Your portfolio has **low market sensitivity**. It's likely to be more stable during market fluctuations.")
            else:
                st.write("Your portfolio has **moderate market sensitivity**, closely tracking overall market movements.")
        
        st.write(f"**Portfolio Volatility:** {expected_risk*100:.2f}%")
        
        # Market correlation analysis
        st.subheader("Market Correlation Analysis")
        
        try:
            # Fetch S&P 500 data for comparison
            spy_data = yf.download('^GSPC', start=start_date, end=end_date)['Close']
            
            # Calculate correlations with market
            correlations = []
            for ticker in stock_data:
                stock_returns = stock_data[ticker]['Close'].pct_change().dropna()
                spy_returns = spy_data.pct_change().dropna()
                
                # Align dates
                aligned_data = pd.concat([stock_returns, spy_returns], axis=1).dropna()
                if len(aligned_data) > 0:
                    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    correlations.append({
                        'Stock': ticker,
                        'Market Correlation': correlation
                    })
            
            # Display correlation table
            if correlations:
                st.table(pd.DataFrame(correlations))
                
                # Visualize correlations
                fig = go.Figure(data=[go.Bar(
                    x=[c['Stock'] for c in correlations],
                    y=[c['Market Correlation'] for c in correlations],
                    marker_color='blue'
                )])
                
                fig.update_layout(
                    title='Market Correlation by Stock',
                    xaxis_title='Stock',
                    yaxis_title='Correlation with S&P 500',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Unable to perform market correlation analysis.")
            st.write(f"Error: {e}")

# Run the main app
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

if run_button or st.session_state.run_clicked:
    st.session_state.run_clicked = True
    run_analysis(stock_list, start_date, end_date, lookback, epochs, batch_size, 
                pop_size, generations, risk_preference)
else:
    st.write("Configure your parameters and click 'Run Analysis' to start.")
    
    # Display example visualization
    st.subheader("Example Visualization")
    
    # Create demo data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), periods=365)
    prices = np.random.normal(loc=100, scale=2, size=365).cumsum() + 100
    
    # Generate sample plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Sample Stock Price',
        line={'color': 'blue'}
    ))
    
    # Add some prediction data
    future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30)
    last_price = prices[-1]
    future_prices = np.random.normal(loc=0, scale=1, size=30).cumsum() + last_price
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines',
        name='Sample Prediction',
        line={'color': 'green'}
    ))
    
    fig.update_layout(
        title='Example Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature explanation
    st.markdown("""
    ## System Features
    
    This portfolio optimization system provides:
    
    1. **Stock Analysis**
       - Historical performance visualization
       - Company information and key metrics
       - Technical indicators
    
    2. **LSTM Prediction Model**
       - 30-day price forecasts
       - Price range predictions with confidence intervals
       - Model performance metrics (RMSE, MAPE)
    
    3. **Genetic Algorithm Optimization**
       - Optimal portfolio allocation
       - Risk-return analysis
       - Efficient frontier visualization
    
    4. **Investment Recommendations**
       - Buy/Hold/Sell recommendations for each stock
       - Price targets and expected returns
       - Overall portfolio strategy
    
    Configure the parameters in the sidebar and click "Run Analysis" to get started.
    """)

# Add footer
st.markdown("""
---
### How to Use This System

1. **Enter Stock Symbols**: Input the stock symbols you're interested in analyzing, separated by commas.
2. **Select Date Range**: Choose the historical data period for analysis.
3. **Adjust Model Parameters**: Fine-tune the LSTM and Genetic Algorithm parameters for better results.
4. **Set Risk Preference**: Adjust your risk tolerance to influence portfolio optimization.
5. **Run Analysis**: Click the button to start the analysis process.
6. **Review Results**: Explore the different tabs to see predictions, optimal allocations, and recommendations.

**Note**: The predictions are based on historical patterns and may not account for unexpected market events. Always combine these insights with fundamental analysis and market research before making investment decisions.
""")
