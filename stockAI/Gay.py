Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
# 🚀 COMPLETE STOCK PREDICTION SYSTEM FOR BEGINNERS
# Just run this code and it will predict stock prices!

# STEP 1: Install required packages (run these in your terminal first)
# pip install yfinance pandas scikit-learn matplotlib numpy

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🎉 WELCOME TO YOUR STOCK PREDICTION SYSTEM!")
print("=" * 60)

# STEP 2: Choose your stock (change this to any stock symbol you want)
STOCK_SYMBOL = "AAPL"  # Apple - you can change to "MSFT", "GOOGL", "TSLA", etc.

def download_stock_data(symbol):
    """Download stock data from Yahoo Finance"""
    print(f"📊 Getting data for {symbol}...")
    try:
        stock = yf.Ticker(symbol)
        # Get 2 years of data
        data = stock.history(period="2y")
        print(f"✅ Successfully downloaded {len(data)} days of data!")
        return data
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def create_prediction_features(data):
    """Create features to help predict stock prices"""
    print("🔧 Creating prediction features...")
    
    df = data.copy()
    
    # Basic price features
    df['Price_Change'] = df['Close'].pct_change()  # Daily price change %
    df['High_Low_Diff'] = df['High'] - df['Low']   # Daily price range
    df['Volume_Change'] = df['Volume'].pct_change()  # Volume change %
    
    # Moving averages (trend indicators)
    df['MA_5'] = df['Close'].rolling(5).mean()    # 5-day average
    df['MA_10'] = df['Close'].rolling(10).mean()  # 10-day average
    df['MA_20'] = df['Close'].rolling(20).mean()  # 20-day average
    
    # Volatility (how much price moves)
    df['Volatility'] = df['Close'].rolling(10).std()
    
    # RSI (momentum indicator - 0 to 100)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # What we want to predict: tomorrow's closing price
    df['Tomorrow_Price'] = df['Close'].shift(-1)
    
    # Remove rows with missing data
    df = df.dropna()
    
    print(f"✅ Created features for {len(df)} days!")
    return df

def train_prediction_model(df):
    """Train AI model to predict stock prices"""
    print("🧠 Training AI model...")
    
    # Features we'll use to predict
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Price_Change', 'High_Low_Diff', 'Volume_Change',
        'MA_5', 'MA_10', 'MA_20', 'Volatility', 'RSI'
    ]
    
    X = df[features]  # Input features
    y = df['Tomorrow_Price']  # What we want to predict
    
    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Create and train the AI model
    model = RandomForestRegressor(
        n_estimators=100,  # Number of decision trees
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"📈 MODEL PERFORMANCE:")
    print(f"   Accuracy Score: {r2*100:.1f}%")
    print(f"   Mean Error: ${np.sqrt(mse):.2f}")
    
    return model, X_test, y_test, predictions, features

def predict_tomorrow(model, df, features):
    """Predict tomorrow's stock price"""
    print("🔮 Making tomorrow's prediction...")
    
    # Get the latest data
    latest_data = df[features].iloc[-1:].values
    
    # Make prediction
    tomorrow_prediction = model.predict(latest_data)[0]
    today_price = df['Close'].iloc[-1]
    
    # Calculate expected change
    price_change = tomorrow_prediction - today_price
    percent_change = (price_change / today_price) * 100
    
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   Today's Price: ${today_price:.2f}")
    print(f"   Tomorrow's Predicted Price: ${tomorrow_prediction:.2f}")
    print(f"   Expected Change: ${price_change:.2f} ({percent_change:+.1f}%)")
    
    if percent_change > 0:
        print(f"   📈 PREDICTION: STOCK WILL GO UP!")
    else:
        print(f"   📉 PREDICTION: STOCK WILL GO DOWN!")
    
    return tomorrow_prediction

def show_feature_importance(model, features):
    """Show which features are most important for predictions"""
    print(f"\n🔍 MOST IMPORTANT PREDICTION FACTORS:")
    
    importance = model.feature_importances_
    feature_importance = list(zip(features, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:5]):
        print(f"   {i+1}. {feature}: {importance:.1%}")

def create_visualization(df, predictions, y_test):
    """Create a chart showing actual vs predicted prices"""
    print("📊 Creating visualization...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted prices
    plt.subplot(1, 2, 1)
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot recent price trend
    plt.subplot(1, 2, 2)
    recent_data = df.tail(30)
    plt.plot(recent_data.index, recent_data['Close'], label='Stock Price', color='green')
    plt.plot(recent_data.index, recent_data['MA_10'], label='10-Day Average', color='orange')
    plt.title('Recent Price Trend (Last 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def investment_advice(percent_change, volatility):
    """Give simple investment advice based on prediction"""
    print(f"\n💡 INVESTMENT ADVICE:")
    
    if abs(percent_change) < 1:
        print("   🔄 Small change expected - might be good for day trading")
    elif percent_change > 2:
        print("   🚀 Strong upward prediction - consider buying")
    elif percent_change < -2:
        print("   ⚠️ Strong downward prediction - consider selling/shorting")
    
    if volatility > df['Volatility'].mean():
        print("   ⚡ High volatility - higher risk but potentially higher reward")
    else:
        print("   😌 Low volatility - more stable, lower risk")
    
    print("\n⚠️ IMPORTANT: This is for educational purposes only!")
    print("   Always do your own research before investing real money!")

# MAIN PROGRAM - THIS IS WHERE EVERYTHING RUNS
def main():
    print(f"🎯 ANALYZING {STOCK_SYMBOL} STOCK...")
    
    # Step 1: Download data
    data = download_stock_data(STOCK_SYMBOL)
    if data is None:
        return
    
    # Step 2: Create features
    df = create_prediction_features(data)
    
    # Step 3: Train model
    model, X_test, y_test, predictions, features = train_prediction_model(df)
    
    # Step 4: Make prediction
    tomorrow_price = predict_tomorrow(model, df, features)
    
    # Step 5: Show important features
    show_feature_importance(model, features)
    
    # Step 6: Give investment advice
    today_price = df['Close'].iloc[-1]
    percent_change = ((tomorrow_price - today_price) / today_price) * 100
    current_volatility = df['Volatility'].iloc[-1]
    investment_advice(percent_change, current_volatility)
    
    # Step 7: Create visualization
    create_visualization(df, predictions, y_test)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Want to analyze a different stock? Change STOCK_SYMBOL at the top!")

# RUN THE PROGRAM
if __name__ == "__main__":
    main()
