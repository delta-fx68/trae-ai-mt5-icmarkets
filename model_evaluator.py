# Remove the yfinance import since we won't need it
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yfinance as yf  # Add this import

# Constants (matching your gold_predictor.py)
MODEL_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\Models\gold_predictor.onnx'
MARKET_DATA_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\current_market_data.csv'
MT5_DATA_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\mt5_gold_data.csv'  # Path to MT5 data
HISTORICAL_DATA_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\historical_market_data.csv'
RESULTS_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\evaluation_results.csv'
STATS_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\model_stats.txt'
INPUT_WINDOW = 60
FORECAST_HORIZON = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10
ACCURACY_THRESHOLD = 0.6  # Minimum accuracy to keep the model

class GoldPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=FORECAST_HORIZON):
        super(GoldPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

def download_historical_data(years=2):
    """Download historical gold price data"""
    print("Downloading historical gold price data...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(HISTORICAL_DATA_PATH), exist_ok=True)
    
    try:
        # Download gold futures data (GC=F is the ticker for gold futures)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        gold_data = yf.download('GC=F', start=start_date, end=end_date, interval='1h')
        
        if len(gold_data) > 100:  # Ensure we got enough data
            # Format the data
            gold_data.reset_index(inplace=True)
            gold_data.rename(columns={
                'Date': 'datetime',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Save to CSV
            gold_data.to_csv(HISTORICAL_DATA_PATH, index=False)
            print(f"Downloaded {len(gold_data)} historical data points")
            return True
        else:
            print("Not enough data downloaded, trying alternative method...")
    except Exception as e:
        print(f"Error downloading data: {e}")
    
    # If yfinance fails, create synthetic data for testing
    print("Creating synthetic data for testing...")
    
    # Generate synthetic data
    dates = pd.date_range(end=datetime.now(), periods=5000, freq='H')
    
    # Start with a base price and add random walks
    base_price = 1900.0
    np.random.seed(42)  # For reproducibility
    
    # Generate random price movements
    price_changes = np.random.normal(0, 1, len(dates)) * 2.0  # 2.0 is volatility
    
    # Calculate cumulative price changes
    cumulative_changes = np.cumsum(price_changes)
    
    # Calculate prices
    close_prices = base_price + cumulative_changes
    
    # Add some volatility for high/low
    high_prices = close_prices + np.abs(np.random.normal(0, 1, len(dates))) * 3.0
    low_prices = close_prices - np.abs(np.random.normal(0, 1, len(dates))) * 3.0
    
    # Ensure low prices don't go below 0
    low_prices = np.maximum(low_prices, 0.1)
    
    # Create open prices (previous close with some noise)
    open_prices = np.roll(close_prices, 1) + np.random.normal(0, 0.5, len(dates))
    open_prices[0] = base_price  # First open price
    
    # Create volume (random)
    volume = np.random.randint(1000, 10000, len(dates))
    
    # Create DataFrame
    synthetic_df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Save to CSV
    synthetic_df.to_csv(HISTORICAL_DATA_PATH, index=False)
    print(f"Created synthetic dataset with {len(synthetic_df)} data points")
    return True

def load_mt5_data():
    """Load historical data from MT5 CSV file"""
    print("Loading MT5 data...")
    
    # Check if MT5 data exists
    if not os.path.exists(MT5_DATA_PATH):
        print(f"MT5 data file not found: {MT5_DATA_PATH}")
        # Look for any CSV files in the Data directory
        data_dir = os.path.dirname(MT5_DATA_PATH)
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if csv_files:
            print(f"Found alternative CSV files: {csv_files}")
            # Use the first CSV file found
            mt5_data_path = os.path.join(data_dir, csv_files[0])
            print(f"Using {mt5_data_path} instead")
        else:
            print("No CSV files found in Data directory")
            return None
    else:
        mt5_data_path = MT5_DATA_PATH
    
    try:
        # Load the MT5 data
        df = pd.read_csv(mt5_data_path)
        
        # Check if required columns exist
        required_cols = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"MT5 data missing required columns. Found: {df.columns.tolist()}")
            
            # Try to map columns if they have different names
            column_mapping = {}
            for col in df.columns:
                if col.lower() in ['time', 'date']:
                    column_mapping[col] = 'datetime'
                elif col.lower() in ['o', 'open']:
                    column_mapping[col] = 'open'
                elif col.lower() in ['h', 'high']:
                    column_mapping[col] = 'high'
                elif col.lower() in ['l', 'low']:
                    column_mapping[col] = 'low'
                elif col.lower() in ['c', 'close']:
                    column_mapping[col] = 'close'
                elif col.lower() in ['v', 'vol', 'volume']:
                    column_mapping[col] = 'volume'
            
            # Rename columns if mapping found
            if column_mapping:
                df = df.rename(columns=column_mapping)
                print(f"Renamed columns: {column_mapping}")
        
        # Add volume column if missing
        if 'volume' not in df.columns:
            print("Adding synthetic volume column")
            df['volume'] = np.random.randint(1000, 10000, len(df))
        
        # Convert datetime column
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Sort by datetime
            df.sort_index(inplace=True)
        
        print(f"Loaded MT5 data with {len(df)} rows")
        
        # Save as historical data for consistency
        df.reset_index().to_csv(HISTORICAL_DATA_PATH, index=False)
        
        return df
    except Exception as e:
        print(f"Error loading MT5 data: {e}")
        return None

def load_historical_data():
    """Load historical market data for testing"""
    # First try to load MT5 data
    df = load_mt5_data()
    if df is not None:
        return df
    
    # If MT5 data not available, check for historical data
    if not os.path.exists(HISTORICAL_DATA_PATH):
        print(f"Historical data file not found: {HISTORICAL_DATA_PATH}")
        return create_synthetic_data()
    
    try:
        df = pd.read_csv(HISTORICAL_DATA_PATH)
        
        # Convert datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Sort by datetime
        df.sort_index(inplace=True)
        
        print(f"Loaded historical data with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic data for testing when no real data is available"""
    print("Creating synthetic data for testing...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(HISTORICAL_DATA_PATH), exist_ok=True)
    
    # Generate synthetic data
    dates = pd.date_range(end=datetime.now(), periods=5000, freq='H')
    
    # Start with a base price and add random walks
    base_price = 1900.0
    np.random.seed(42)  # For reproducibility
    
    # Generate random price movements
    price_changes = np.random.normal(0, 1, len(dates)) * 2.0  # 2.0 is volatility
    
    # Calculate cumulative price changes
    cumulative_changes = np.cumsum(price_changes)
    
    # Calculate prices
    close_prices = base_price + cumulative_changes
    
    # Add some volatility for high/low
    high_prices = close_prices + np.abs(np.random.normal(0, 1, len(dates))) * 3.0
    low_prices = close_prices - np.abs(np.random.normal(0, 1, len(dates))) * 3.0
    
    # Ensure low prices don't go below 0
    low_prices = np.maximum(low_prices, 0.1)
    
    # Create open prices (previous close with some noise)
    open_prices = np.roll(close_prices, 1) + np.random.normal(0, 0.5, len(dates))
    open_prices[0] = base_price  # First open price
    
    # Create volume (random)
    volume = np.random.randint(1000, 10000, len(dates))
    
    # Create DataFrame
    synthetic_df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Save to CSV
    synthetic_df.to_csv(HISTORICAL_DATA_PATH, index=False)
    print(f"Created synthetic dataset with {len(synthetic_df)} data points")
    
    # Set datetime as index
    synthetic_df.set_index('datetime', inplace=True)
    
    return synthetic_df

def prepare_test_data(df, test_size=0.3):
    """Prepare data for testing the model"""
    # Prepare features and targets
    X = []
    y = []
    actual_prices = []
    timestamps = []
    
    for i in range(len(df) - INPUT_WINDOW - FORECAST_HORIZON):
        # Input window
        window = df.iloc[i:i+INPUT_WINDOW][['open', 'high', 'low', 'close', 'volume']].values
        
        # Target (future prices)
        future_prices = df.iloc[i+INPUT_WINDOW:i+INPUT_WINDOW+FORECAST_HORIZON]['close'].values
        last_price = window[-1, 3]  # Last close price in window
        
        # Calculate price changes relative to last price
        price_changes = (future_prices - last_price) / last_price
        
        X.append(window)
        y.append(price_changes)
        actual_prices.append(future_prices)
        timestamps.append(df.index[i+INPUT_WINDOW])
    
    X = np.array(X)
    y = np.array(y)
    actual_prices = np.array(actual_prices)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    actual_train, actual_test = actual_prices[:split_idx], actual_prices[split_idx:]
    timestamps_test = timestamps[split_idx:]
    
    return X_train, X_test, y_train, y_test, actual_test, timestamps_test

def load_onnx_model():
    """Load the ONNX model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return None
    
    try:
        session = ort.InferenceSession(MODEL_PATH)
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_inference_on_test_data(session, X_test):
    """Run inference on test data"""
    predictions = []
    
    for i in range(len(X_test)):
        # Prepare input
        price_input = X_test[i:i+1].astype(np.float32)
        
        # Get input names
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        # Create inputs dictionary
        inputs = {}
        if len(input_names) >= 1:
            inputs[input_names[0]] = price_input
        
        # Run inference
        outputs = session.run(output_names, inputs)
        predictions.append(outputs[0][0])
    
    return np.array(predictions)

def calculate_metrics(predictions, y_test, actual_test, last_prices):
    """Calculate accuracy metrics"""
    # Direction accuracy
    pred_directions = np.sign(predictions)
    true_directions = np.sign(y_test)
    
    direction_accuracy = np.mean(pred_directions == true_directions)
    
    # Calculate predicted prices
    predicted_prices = np.zeros_like(predictions)
    for i in range(len(predictions)):
        for j in range(FORECAST_HORIZON):
            predicted_prices[i, j] = last_prices[i] * (1 + predictions[i, j])
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(actual_test.flatten(), predicted_prices.flatten())
    rmse = np.sqrt(mean_squared_error(actual_test.flatten(), predicted_prices.flatten()))
    
    # Calculate percentage error
    mape = np.mean(np.abs((actual_test.flatten() - predicted_prices.flatten()) / actual_test.flatten())) * 100
    
    return {
        'direction_accuracy': direction_accuracy,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

def train_adjusted_model(X_train, y_train, metrics):
    """Train an adjusted model based on performance metrics"""
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Create model
    model = GoldPredictor()
    
    # Adjust learning rate based on metrics
    adjusted_lr = LEARNING_RATE
    if metrics['mape'] > 5.0:  # If error is high, increase learning rate
        adjusted_lr *= 1.5
    
    # Adjust epochs based on metrics
    adjusted_epochs = EPOCHS
    if metrics['direction_accuracy'] < 0.55:  # If accuracy is low, train longer
        adjusted_epochs = int(EPOCHS * 1.5)
    
    print(f"Training with adjusted parameters: LR={adjusted_lr}, Epochs={adjusted_epochs}")
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=adjusted_lr)
    criterion = nn.MSELoss()
    
    for epoch in range(adjusted_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{adjusted_epochs}, Loss: {loss.item():.4f}")
    
    # Save model
    dummy_input = torch.randn(1, INPUT_WINDOW, 5)
    torch.onnx.export(model, dummy_input, MODEL_PATH)
    
    return model

# Complete the truncated plot function
def plot_results(timestamps, actual_prices, predicted_prices, horizon_idx=0):
    """Plot actual vs predicted prices for a specific horizon"""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, actual_prices[:, horizon_idx], label='Actual')
    plt.plot(timestamps, predicted_prices[:, horizon_idx], label=f'Predicted (t+{horizon_idx+1} days)')
    plt.title(f'Gold Price Prediction (t+{horizon_idx+1} days)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Save plot
    plot_path = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\prediction_plot.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    
    # Create additional plots for different forecast horizons
    if FORECAST_HORIZON > 1:
        # Plot for 5-day forecast if available
        mid_idx = min(4, FORECAST_HORIZON-1)
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, actual_prices[:, mid_idx], label='Actual')
        plt.plot(timestamps, predicted_prices[:, mid_idx], label=f'Predicted (t+{mid_idx+1} days)')
        plt.title(f'Gold Price Prediction (t+{mid_idx+1} days)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        mid_plot_path = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\prediction_plot_mid.png'
        plt.savefig(mid_plot_path)
        plt.close()
        
        # Plot for longest forecast if available
        last_idx = FORECAST_HORIZON-1
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, actual_prices[:, last_idx], label='Actual')
        plt.plot(timestamps, predicted_prices[:, last_idx], label=f'Predicted (t+{last_idx+1} days)')
        plt.title(f'Gold Price Prediction (t+{last_idx+1} days)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        long_plot_path = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\prediction_plot_long.png'
        plt.savefig(long_plot_path)
        plt.close()

# Add the scan_data_directory function
def scan_data_directory():
    """Scan the data directory for available CSV files, prioritizing D1 data"""
    data_dir = os.path.dirname(HISTORICAL_DATA_PATH)
    print(f"Scanning directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data directory: {data_dir}")
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files:")
    for i, file in enumerate(csv_files):
        file_path = os.path.join(data_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"  {i+1}. {file} ({file_size:.1f} KB)")
    
    # Look for D1 data files first (xauusd_d1.csv or similar)
    d1_files = [f for f in csv_files if 'd1' in f.lower() or 'daily' in f.lower()]
    xauusd_files = [f for f in csv_files if 'xauusd' in f.lower() or 'gold' in f.lower()]
    
    # Prioritize files that match both criteria
    priority_files = [f for f in d1_files if f in xauusd_files]
    
    if priority_files:
        print(f"Found XAUUSD D1 data files: {priority_files}")
        return priority_files
    elif d1_files:
        print(f"Found D1 data files: {d1_files}")
        return d1_files
    elif xauusd_files:
        print(f"Found XAUUSD data files: {xauusd_files}")
        return xauusd_files
    else:
        print(f"No specific gold/XAUUSD D1 files found. Using available files.")
        return csv_files

# Add the update_current_market_data function
def update_current_market_data():
    """Update current market data from the latest available data"""
    print("Updating current market data...")
    
    # Scan for available data files
    csv_files = scan_data_directory()
    
    if not csv_files:
        print("No CSV files found in data directory")
        return False
    
    # Use the first file in the prioritized list
    data_file = os.path.join(os.path.dirname(HISTORICAL_DATA_PATH), csv_files[0])
    print(f"Using data file: {data_file}")
    
    try:
        # Load the data
        df = pd.read_csv(data_file)
        print(f"Data file columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        required_cols = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"Data file missing required columns.")
            
            # Try to map columns if they have different names
            column_mapping = {}
            for col in df.columns:
                if col.lower() in ['time', 'date']:
                    column_mapping[col] = 'datetime'
                elif col.lower() in ['o', 'open']:
                    column_mapping[col] = 'open'
                elif col.lower() in ['h', 'high']:
                    column_mapping[col] = 'high'
                elif col.lower() in ['l', 'low']:
                    column_mapping[col] = 'low'
                elif col.lower() in ['c', 'close']:
                    column_mapping[col] = 'close'
                elif col.lower() in ['v', 'vol', 'volume']:
                    column_mapping[col] = 'volume'
            
            # Rename columns if mapping found
            if column_mapping:
                df = df.rename(columns=column_mapping)
                print(f"Renamed columns: {column_mapping}")
        
        # Add volume column if missing
        if 'volume' not in df.columns:
            print("Adding synthetic volume column")
            df['volume'] = np.random.randint(1000, 10000, len(df))
        
        # Convert datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Get the most recent data point
        latest_data = df.iloc[-1:].copy()
        
        # Save to current market data file
        os.makedirs(os.path.dirname(MARKET_DATA_PATH), exist_ok=True)
        latest_data.to_csv(MARKET_DATA_PATH, index=False)
        
        print(f"Updated current market data with latest data from {latest_data['datetime'].iloc[0]}")
        return True
    except Exception as e:
        print(f"Error updating current market data: {e}")
        return False

# Update the main function to call update_current_market_data and add more logging
def main():
    """Main function to evaluate and adjust the model"""
    print("=" * 50)
    print("Starting Gold Price Predictor Evaluation")
    print("=" * 50)
    
    try:
        # Update current market data first
        print("\n1. Updating current market data...")
        update_current_market_data()
        
        # Load historical data
        print("\n2. Loading historical data...")
        hist_df = load_historical_data()
        if hist_df is None:
            print("Failed to load historical data")
            return
        
        print(f"Data shape: {hist_df.shape}")
        print(f"Date range: {hist_df.index.min()} to {hist_df.index.max()}")
        
        # Prepare test data
        print("\n3. Preparing test data...")
        result = prepare_test_data(hist_df)
        if result is None:
            print("Failed to prepare test data")
            return
            
        X_train, X_test, y_train, y_test, actual_test, timestamps_test = result
        print(f"Created {len(X_train)} training samples and {len(X_test)} test samples")
        
        # Get last prices for each test window
        last_prices = X_test[:, -1, 3]  # Last close price in each window
        
        # Load model or create a new one if it doesn't exist
        print("\n4. Loading model...")
        session = load_onnx_model()
        if session is None:
            print("Model not found, creating a new one...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Train a new model
            model = GoldPredictor()
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            
            # Train model
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.MSELoss()
            
            print(f"Training new model with {len(X_train)} samples...")
            for epoch in range(EPOCHS):
                model.train()
                optimizer.zero_grad()
                
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                
                loss.backward()
                optimizer.step()
                
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
            
            # Save model
            dummy_input = torch.randn(1, INPUT_WINDOW, 5)
            torch.onnx.export(model, dummy_input, MODEL_PATH)
            print(f"New model saved to {MODEL_PATH}")
            
            # Reload the model
            session = load_onnx_model()
            if session is None:
                print("Failed to create and load model")
                return
        
        # Run inference on test data
        print("\n5. Running inference on test data...")
        predictions = run_inference_on_test_data(session, X_test)
        
        # Calculate metrics
        print("\n6. Calculating performance metrics...")
        metrics = calculate_metrics(predictions, y_test, actual_test, last_prices)
        
        print("\n===== Model Performance =====")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        print(f"Mean Absolute Error: {metrics['mae']:.2f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.2f}")
        print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        
        # Calculate predicted prices for plotting
        predicted_prices = np.zeros_like(predictions)
        for i in range(len(predictions)):
            for j in range(FORECAST_HORIZON):
                predicted_prices[i, j] = last_prices[i] * (1 + predictions[i, j])
        
        # Plot results
        print("\n7. Creating plots...")
        plot_results(timestamps_test, actual_test, predicted_prices)
        
        # Save evaluation results
        print("\n8. Saving evaluation results...")
        results_df = pd.DataFrame({
            'datetime': timestamps_test,
            'actual_h1': actual_test[:, 0],
            'predicted_h1': predicted_prices[:, 0],
            'actual_h5': actual_test[:, 4] if FORECAST_HORIZON > 4 else None,
            'predicted_h5': predicted_prices[:, 4] if FORECAST_HORIZON > 4 else None,
            'actual_h10': actual_test[:, 9] if FORECAST_HORIZON > 9 else None,
            'predicted_h10': predicted_prices[:, 9] if FORECAST_HORIZON > 9 else None,
        })
        
        results_df.to_csv(RESULTS_PATH, index=False)
        print(f"Evaluation results saved to {RESULTS_PATH}")
        
        # Auto-adjust model if accuracy is below threshold
        if metrics['direction_accuracy'] < ACCURACY_THRESHOLD:
            print(f"\n9. Direction accuracy ({metrics['direction_accuracy']:.2%}) below threshold ({ACCURACY_THRESHOLD:.2%})")
            print("Auto-adjusting model...")
            
            # Train adjusted model
            adjusted_model = train_adjusted_model(X_train, y_train, metrics)
            
            print("Model adjusted and saved")
        else:
            print(f"\n9. Model performing well (accuracy: {metrics['direction_accuracy']:.2%})")
        
        print("\n" + "=" * 50)
        print("Evaluation completed successfully")
        print("=" * 50)
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()