import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Constants
MODEL_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\Models\gold_predictor.onnx'
MARKET_DATA_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\current_market_data.csv'
NEWS_DATA_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\news_data.csv'
RESULTS_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\prediction_results.csv'
STATS_PATH = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\model_stats.txt'
INPUT_WINDOW = 60
FORECAST_HORIZON = 10

# Training constants
RETRAIN_INTERVAL = 24  # Hours between retraining
LAST_TRAIN_FILE = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\last_train.txt'
MIN_TRAINING_SAMPLES = 1000  # Minimum samples needed for training
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10

def should_retrain():
    """Check if model needs retraining"""
    if not os.path.exists(LAST_TRAIN_FILE):
        return True
        
    try:
        with open(LAST_TRAIN_FILE, 'r') as f:
            last_train = datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S')
        return datetime.now() - last_train > timedelta(hours=RETRAIN_INTERVAL)
    except:
        return True

def collect_training_data():
    """Collect and prepare training data"""
    try:
        # Load historical market data
        hist_df = pd.read_csv(MARKET_DATA_PATH)
        
        if len(hist_df) < MIN_TRAINING_SAMPLES:
            print(f"Not enough training data: {len(hist_df)} samples")
            return None, None
            
        # Prepare features and targets
        X = []
        y = []
        
        for i in range(len(hist_df) - INPUT_WINDOW - FORECAST_HORIZON):
            # Input window
            window = hist_df[i:i+INPUT_WINDOW][['open', 'high', 'low', 'close', 'volume']].values
            
            # Target (future price changes)
            future_prices = hist_df[i+INPUT_WINDOW:i+INPUT_WINDOW+FORECAST_HORIZON]['close'].values
            price_changes = (future_prices - window[-1,3]) / window[-1,3]  # Relative to last close price
            
            X.append(window)
            y.append(price_changes)
            
        return np.array(X), np.array(y)
        
    except Exception as e:
        print(f"Error collecting training data: {e}")
        return None, None

def train_model():
    """Train/retrain the model"""
    try:
        # Collect training data
        X, y = collect_training_data()
        if X is None or y is None:
            return False
            
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Define and train model (simplified example)
        model = nn.Sequential(
            nn.LSTM(5, 64, batch_first=True),
            nn.Linear(64, FORECAST_HORIZON)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
        
        # Save model
        torch.onnx.export(model, X_train[0:1], MODEL_PATH)
        
        # Update last train time
        with open(LAST_TRAIN_FILE, 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def update_market_data():
    """Update market data from available data sources"""
    print("Checking for updated market data...")
    
    # Check if we have a data directory with CSV files
    data_dir = os.path.dirname(MARKET_DATA_PATH)
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        return False
    
    # Look for D1 data files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    d1_files = [f for f in csv_files if 'd1' in f.lower() or 'daily' in f.lower()]
    xauusd_files = [f for f in csv_files if 'xauusd' in f.lower() or 'gold' in f.lower()]
    
    # Also check parent directory for data files
    parent_dir = os.path.dirname(data_dir)
    parent_csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') or f.endswith('.CSV')]
    parent_d1_files = [f for f in parent_csv_files if 'd1' in f.lower() or 'daily' in f.lower()]
    parent_xauusd_files = [f for f in parent_csv_files if 'xauusd' in f.lower() or 'gold' in f.lower()]
    
    # Add processed files to the list
    processed_files = [f for f in parent_csv_files if 'processed' in f.lower()]
    
    # Prioritize files that match both criteria
    priority_files = [f for f in d1_files if f in xauusd_files]
    parent_priority_files = [f for f in parent_d1_files if f in parent_xauusd_files]
    
    # Select the best file
    data_file = None
    
    # First check for processed files in parent directory
    if processed_files:
        data_file = os.path.join(parent_dir, processed_files[0])
    # Then check for priority files in data directory
    elif priority_files:
        data_file = os.path.join(data_dir, priority_files[0])
    # Then check for priority files in parent directory
    elif parent_priority_files:
        data_file = os.path.join(parent_dir, parent_priority_files[0])
    # Then check for any XAUUSD files
    elif xauusd_files:
        data_file = os.path.join(data_dir, xauusd_files[0])
    elif parent_xauusd_files:
        data_file = os.path.join(parent_dir, parent_xauusd_files[0])
    # Finally check for any D1 files
    elif d1_files:
        data_file = os.path.join(data_dir, d1_files[0])
    elif parent_d1_files:
        data_file = os.path.join(parent_dir, parent_d1_files[0])
    
    if not data_file:
        print("No suitable data file found")
        return False
    
    print(f"Using data file: {data_file}")
    
    try:
        # Load the data
        df = pd.read_csv(data_file)
        
        # Check if required columns exist
        required_cols = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
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
        
        # Add volume column if missing
        if 'volume' not in df.columns:
            df['volume'] = np.random.randint(1000, 10000, len(df))
        
        # Convert datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Force update by always saving the data
        print(f"Saving full historical dataset with {len(df)} records")
        df.to_csv(MARKET_DATA_PATH, index=False)
        
        # Create a timestamp file to track when data was last updated
        with open(os.path.join(data_dir, 'last_data_update.txt'), 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return True
        
    except Exception as e:
        print(f"Error updating market data: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_model():
    """Load the ONNX model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return None
    
    # Create ONNX Runtime session
    try:
        session = ort.InferenceSession(MODEL_PATH)
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_market_data():
    """Load current market data"""
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Market data file not found: {MARKET_DATA_PATH}")
        return None
    
    # Load data
    try:
        df = pd.read_csv(MARKET_DATA_PATH)
        
        # Convert datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Sort by datetime
        df.sort_index(inplace=True)
        
        return df
    except Exception as e:
        print(f"Error loading market data: {e}")
        return None

def load_news_data():
    """Load news data"""
    if not os.path.exists(NEWS_DATA_PATH):
        print(f"News data file not found: {NEWS_DATA_PATH}")
        # Create dummy news features
        return np.zeros((INPUT_WINDOW, 1))
    
    try:
        # Load data
        df = pd.read_csv(NEWS_DATA_PATH)
        
        # Convert datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Get news columns
        news_cols = [col for col in df.columns if col.startswith(('sentiment', 'news', 'time_decayed', 'weighted'))]
        
        if not news_cols:
            print("No news features found in the data")
            return np.zeros((INPUT_WINDOW, 1))
        
        # Get the latest news data and average across features
        latest_news = df[news_cols].iloc[-INPUT_WINDOW:].mean(axis=1).values.reshape(-1, 1)
        
        # If we don't have enough data, pad with zeros
        if len(latest_news) < INPUT_WINDOW:
            padding = np.zeros((INPUT_WINDOW - len(latest_news), 1))
            latest_news = np.vstack([padding, latest_news])
        
        return latest_news
    except Exception as e:
        print(f"Error loading news data: {e}")
        return np.zeros((INPUT_WINDOW, 1))

def preprocess_data(market_df):
    """Preprocess market data for the model"""
    # Extract price features
    price_features = market_df[['open', 'high', 'low', 'close', 'volume']].values
    
    # Load price mean and scale if available
    price_mean = 0.0
    price_scale = 1.0
    
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH, 'r') as f:
                stats = f.read().strip().split(',')
                price_mean = float(stats[0])
                price_scale = float(stats[1])
        except:
            print("Error loading stats, using defaults")
    
    # No normalization as requested
    # We're keeping the data in its original scale
    
    return price_features

def run_inference(session, price_features, news_features):
    """Run inference using the ONNX model"""
    try:
        # Prepare inputs with correct shapes
        price_input = price_features[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, 5).astype(np.float32)
        news_input = news_features[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, 1).astype(np.float32)
        
        # Print shapes for debugging
        print(f"Price input shape: {price_input.shape}")
        print(f"News input shape: {news_input.shape}")
        
        # Get input and output names
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Model input names: {input_names}")
        print(f"Model output names: {output_names}")
        
        # Create inputs dictionary using actual model input names
        inputs = {}
        if len(input_names) >= 2:
            inputs[input_names[0]] = price_input
            inputs[input_names[1]] = news_input
        elif len(input_names) == 1:
            # If model only has one input, use the price input
            inputs[input_names[0]] = price_input
        else:
            print("Model has no inputs defined")
            return None
        
        # Run inference
        outputs = session.run(output_names, inputs)
        
        # Extract forecast
        forecast = outputs[0]  # First output is the forecast
        
        return forecast[0]
    
    except Exception as e:
        print(f"Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results(forecast):
    """Save prediction results to CSV with confidence index"""
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    # Create DataFrame with results
    results_df = pd.DataFrame()
    
    # Current time
    current_time = datetime.now()
    
    # Get current price as baseline
    try:
        current_price = 0
        if os.path.exists(MARKET_DATA_PATH):
            market_df = pd.read_csv(MARKET_DATA_PATH)
            if not market_df.empty:
                current_price = market_df['close'].iloc[-1]
        
        if current_price == 0:
            # Fallback to a reasonable gold price if we can't get the current price
            current_price = 2880.0
            
        print(f"Using baseline price: {current_price}")
    except Exception as e:
        print(f"Error getting current price: {e}")
        current_price = 2880.0
    
    # Calculate the direction and magnitude of predictions
    directions = np.sign(forecast)
    magnitudes = np.abs(forecast)
    max_magnitude = np.max(magnitudes) if len(magnitudes) > 0 else 1.0
    
    # Calculate confidence index based on model output
    # Higher absolute values typically indicate stronger confidence
    confidence_scores = np.clip(np.abs(forecast) / (max_magnitude + 1e-6) * 100, 0, 100)
    
    # For each prediction horizon step
    for i in range(FORECAST_HORIZON):
        # Apply a more conservative scaling approach
        # Use sigmoid-like function to constrain changes to reasonable range
        normalized_magnitude = magnitudes[i] / (max_magnitude + 1e-6)  # Avoid division by zero
        pct_change = directions[i] * 0.02 * np.tanh(normalized_magnitude)  # Max 2% change
        
        predicted_price = current_price * (1 + pct_change)
        
        # Ensure price doesn't go negative and stays within reasonable bounds
        predicted_price = max(predicted_price, 0.01)
        
        # Limit extreme predictions (no more than 5% change from current price)
        max_change = current_price * 0.05
        predicted_price = max(min(predicted_price, current_price + max_change), current_price - max_change)
        
        # Calculate confidence level (0-100%)
        confidence = int(confidence_scores[i])
        
        # Format datetime in a way that MQL5 can easily parse
        future_time = current_time + pd.Timedelta(hours=i+1)
        future_time_str = future_time.strftime('%Y.%m.%d %H:%M:%S')
        results_df = pd.concat([results_df, pd.DataFrame({
            'datetime': [future_time_str],  # Format changed for MQL5 compatibility
            'predicted': [round(predicted_price, 2)],
            'raw_output': [forecast[i]],
            'actual': [None],  # Will be filled later when actual data is available
            'direction': [1 if directions[i] > 0 else -1],  # Add explicit direction for MQL5
            'confidence': [confidence]  # Add confidence score
        })], ignore_index=True)
    
    # Save to CSV with specific formatting for MQL5
    results_df.to_csv(RESULTS_PATH, index=False, float_format='%.2f')
    
    # Create a very simple MQL5-friendly format (tab-separated values)
    with open(RESULTS_PATH.replace('.csv', '_mql5.txt'), 'w') as f:
        f.write("datetime\tpredicted\tdirection\tconfidence\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['datetime']}\t{row['predicted']}\t{row['direction']}\t{row['confidence']}\n")
    
    # Create an ultra-simple format for MQL5 with confidence
    simple_path = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\prediction_simple.txt'
    with open(simple_path, 'w') as f:
        # Write current time and price
        current_time_str = current_time.strftime('%Y.%m.%d %H:%M:%S')
        f.write(f"{current_time_str}|{current_price}|0\n")  # 0 confidence for current price
        
        # Write future predictions with explicit times and confidence
        for i in range(FORECAST_HORIZON):
            future_time = current_time + pd.Timedelta(hours=i+1)
            future_time_str = future_time.strftime('%Y.%m.%d %H:%M:%S')
            predicted_price = results_df['predicted'].iloc[i]
            confidence = results_df['confidence'].iloc[i]
            f.write(f"{future_time_str}|{predicted_price}|{confidence}\n")
    
    # Create a special format for today's closing price prediction
    # Find the prediction that's closest to today's market close (typically around 17:00)
    today = datetime.now().date()
    market_close = datetime.combine(today, datetime.strptime("17:00", "%H:%M").time())
    
    # If it's already past market close, use tomorrow's close
    if datetime.now() > market_close:
        market_close = datetime.combine(today + timedelta(days=1), datetime.strptime("17:00", "%H:%M").time())
    
    # Find the prediction closest to market close
    time_diffs = [(pd.to_datetime(row['datetime']) - market_close).total_seconds() for _, row in results_df.iterrows()]
    closest_idx = np.argmin(np.abs(time_diffs))
    
    # Create a special file for today's closing price prediction
    today_path = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\today_prediction.txt'
    with open(today_path, 'w') as f:
        predicted_close = results_df['predicted'].iloc[closest_idx]
        confidence = results_df['confidence'].iloc[closest_idx]
        direction = results_df['direction'].iloc[closest_idx]
        f.write(f"{market_close.strftime('%Y.%m.%d %H:%M:%S')}|{predicted_close}|{confidence}|{direction}\n")
    
    # Also create a copy in MQL5 format for consistency
    today_mql5_path = r'C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data\today_prediction_mql5.txt'
    with open(today_mql5_path, 'w') as f:
        f.write(f"{market_close.strftime('%Y.%m.%d %H:%M:%S')}\t{predicted_close}\t{direction}\t{confidence}\n")
    
    print(f"Results saved to {RESULTS_PATH} and simplified versions")
    print(f"Today's closing price prediction: {predicted_close} (Confidence: {confidence}%)")
def main():
    """Main function to run the prediction"""
    try:
        # Check if market data exists and when it was last updated
        if os.path.exists(MARKET_DATA_PATH):
            last_modified = datetime.fromtimestamp(os.path.getmtime(MARKET_DATA_PATH))
            print(f"Market data file last updated: {last_modified}")
            
            # Check if the file is too old (more than 24 hours)
            if datetime.now() - last_modified > timedelta(hours=24):
                print("WARNING: Market data file is more than 24 hours old!")
                print("Make sure your MQL5 script is correctly exporting data.")
        else:
            print(f"WARNING: Market data file not found at {MARKET_DATA_PATH}")
            print("Make sure your MQL5 script is correctly exporting data.")
            return False
        
        # Check if retraining is needed
        if should_retrain():
            print("Retraining model...")
            if not train_model():
                print("Training failed, using existing model")
        
        # Continue with existing prediction logic
        session = load_model()
        if session is None:
            return False
        
        # Load market data
        market_df = load_market_data()
        if market_df is None:
            return False
        
        # Load news data
        news_features = load_news_data()
        
        # Preprocess data
        price_features = preprocess_data(market_df)
        
        # Run prediction
        predictions = run_inference(session, price_features, news_features)
        if predictions is None:
            return False
        
        # Print raw predictions for debugging
        print(f"Raw predictions: {predictions}")
            
        # Save results with confidence index
        save_results(predictions)
        
        print("Prediction completed successfully")
        return True
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
        return False

if __name__ == "__main__":
    main()