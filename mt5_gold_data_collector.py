import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import pytz

def connect_to_mt5():
    """Connect to the MetaTrader 5 terminal"""
    if not mt5.initialize():
        print(f"MT5 initialization failed with error: {mt5.last_error()}")
        return False
    
    print(f"Connected to MetaTrader 5 build {mt5.version()}")
    return True

def collect_gold_data(symbol="XAUUSD", timeframes=None, start_date=None, save_dir="Data"):
    """Collect gold data for multiple timeframes"""
    if timeframes is None:
        timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "D1": mt5.TIMEFRAME_D1
        }
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Set default start date to 1 year ago if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    
    # Make sure the symbol is available
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol {symbol}")
        return None
    
    print(f"Collecting data for {symbol}...")
    
    # Dictionary to store dataframes for each timeframe
    all_data = {}
    
    # Timezone for MT5 data (UTC)
    timezone = pytz.timezone("UTC")
    
    # Current time in UTC
    end_date = datetime.now(tz=timezone)
    
    # Convert start_date to UTC timezone if it's not timezone-aware
    if start_date.tzinfo is None:
        start_date = timezone.localize(start_date)
    
    for tf_name, tf_value in timeframes.items():
        print(f"Collecting {tf_name} data...")
        
        # Get rates
        rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"No data received for {tf_name}")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Save to CSV
        file_path = os.path.join(save_dir, f"{symbol}_{tf_name}.csv")
        df.to_csv(file_path, index=False)
        
        print(f"Saved {len(df)} records to {file_path}")
        
        # Store in dictionary
        all_data[tf_name] = df
    
    return all_data

def process_data(all_data, save_dir="Data"):
    """Process the collected data and add technical indicators"""
    for tf_name, df in all_data.items():
        print(f"Processing {tf_name} data...")
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Save processed data
        processed_file = os.path.join(save_dir, f"XAUUSD_{tf_name}_processed.csv")
        
        # Try to save with error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Close any open file handles that might be causing issues
                plt.close('all')
                
                # Try to save the file
                df.to_csv(processed_file)
                print(f"Processed data saved to {processed_file}")
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"Permission error when saving {processed_file}. Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                else:
                    print(f"Failed to save {processed_file} after {max_retries} attempts: {e}")
                    print("Continuing with analysis without saving...")
        
        # Analyze data
        try:
            analyze_data(df, tf_name)
        except Exception as e:
            print(f"Error during analysis of {tf_name} data: {e}")
    
    return all_data

def add_technical_indicators(df):
    """Add basic technical indicators to the DataFrame"""
    # Calculate returns
    df['Returns'] = df['close'].pct_change()
    
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
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
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
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
    print(df[['open', 'high', 'low', 'close']].describe())
    
    # Volume statistics
    print("\nVolume Statistics:")
    print(df['tick_volume'].describe())
    
    # Returns statistics
    print("\nReturns Statistics:")
    print(df['Returns'].describe())
    
    # Plot price chart
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'])
    plt.title(f'Gold Price ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.savefig(f'Gold_Price_{timeframe}.png')
    
    # Plot volume
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df['tick_volume'], color='gray', alpha=0.5)
    plt.title(f'Gold Trading Volume ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.savefig(f'Gold_Volume_{timeframe}.png')
    
    print(f"Charts saved as Gold_Price_{timeframe}.png and Gold_Volume_{timeframe}.png")

def main():
    # Connect to MT5
    if not connect_to_mt5():
        return
    
    try:
        # Collect data from 2005 instead of just the last 2 years
        start_date = datetime(2005, 1, 1)
        
        # Define timeframes to collect
        # Process larger timeframes first to avoid memory issues
        timeframes = {
            "D1": mt5.TIMEFRAME_D1,
            "H1": mt5.TIMEFRAME_H1,
            "M15": mt5.TIMEFRAME_M15,
            "M1": mt5.TIMEFRAME_M1
        }
        
        # Process each timeframe separately to avoid memory issues
        for tf_name, tf_value in timeframes.items():
            print(f"\nProcessing {tf_name} timeframe...")
            
            # Collect data for this timeframe only
            current_timeframes = {tf_name: tf_value}
            
            # Collect data
            all_data = collect_gold_data(
                symbol="XAUUSD", 
                timeframes=current_timeframes,
                start_date=start_date,
                save_dir="Data"
            )
            
            if all_data:
                # Process data for this timeframe
                process_data(all_data, save_dir="Data")
                
                # Clear memory
                all_data = None
                import gc
                gc.collect()
            
            print(f"Completed processing {tf_name} timeframe")
            
        print("\nData collection and processing completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Shutdown MT5 connection
        mt5.shutdown()
        print("MT5 connection closed")

if __name__ == "__main__":
    main()