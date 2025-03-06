import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def load_mt5_csv(file_path):
    """Load MT5 CSV file and convert to pandas DataFrame with proper types"""
    print(f"Loading data from {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return None
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Datetime', inplace=True)
    
    # Convert price columns to float
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert volume to float
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Convert spread to int
    df['Spread'] = pd.to_numeric(df['Spread'], errors='coerce').astype('int32')
    
    print(f"Loaded {len(df)} records from {os.path.basename(file_path)}")
    return df

def add_technical_indicators(df):
    """Add basic technical indicators to the DataFrame"""
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_20'] - df['EMA_50']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def analyze_data(df, timeframe):
    """Perform basic analysis on the data"""
    print(f"\nAnalysis for {timeframe} data:")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total records: {len(df)}")
    
    # Price statistics
    print("\nPrice Statistics:")
    print(df[['Open', 'High', 'Low', 'Close']].describe())
    
    # Volume statistics
    print("\nVolume Statistics:")
    print(df['Volume'].describe())
    
    # Returns statistics
    print("\nReturns Statistics:")
    print(df['Returns'].describe())
    
    # Plot price chart
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'])
    plt.title(f'Gold Price ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.savefig(f'Gold_Price_{timeframe}.png')
    
    # Plot volume
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df['Volume'], color='gray', alpha=0.5)
    plt.title(f'Gold Trading Volume ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.savefig(f'Gold_Volume_{timeframe}.png')
    
    print(f"Charts saved as Gold_Price_{timeframe}.png and Gold_Volume_{timeframe}.png")

def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'Data')
    if not os.path.exists(data_dir):
        print("Data directory not found. Please run the MQL5 script first.")
        return
    
    # Process each timeframe
    timeframes = ['M1', 'M15', 'H1', 'D1']
    dfs = {}
    
    for tf in timeframes:
        file_path = os.path.join(data_dir, f'XAUUSD_{tf}.csv')
        if os.path.exists(file_path):
            df = load_mt5_csv(file_path)
            if df is not None:
                # Add technical indicators
                df = add_technical_indicators(df)
                dfs[tf] = df
                
                # Save processed data
                processed_file = os.path.join(data_dir, f'XAUUSD_{tf}_processed.csv')
                df.to_csv(processed_file)
                print(f"Processed data saved to {processed_file}")
                
                # Analyze data
                analyze_data(df, tf)
        else:
            print(f"File for {tf} timeframe not found.")
    
    print("\nData processing completed!")

if __name__ == "__main__":
    main()