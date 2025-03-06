import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def load_merged_data(timeframe):
    """
    Load merged price and news data
    """
    file_path = os.path.join("Data", f"merged_data_{timeframe}.csv")
    if not os.path.exists(file_path):
        print(f"Merged data file not found: {file_path}")
        return None
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert datetime column to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    return df

def add_lagged_features(df, timeframe):
    """
    Add lagged price and volume features
    """
    # Define lag periods based on timeframe
    if timeframe == 'M15':
        lags = {
            '1h': 4,    # 4 * 15min = 1h
            '4h': 16,   # 16 * 15min = 4h
            '24h': 96   # 96 * 15min = 24h
        }
    elif timeframe == 'H1':
        lags = {
            '1h': 1,    # 1 * 1h = 1h
            '4h': 4,    # 4 * 1h = 4h
            '24h': 24   # 24 * 1h = 24h
        }
    elif timeframe == 'D1':
        lags = {
            '1h': 1,    # Not applicable for daily data, use 1 day
            '4h': 1,    # Not applicable for daily data, use 1 day
            '24h': 1    # 1 * 1d = 24h
        }
    else:
        print(f"Unknown timeframe: {timeframe}")
        return df
    
    # Add lagged features
    for period_name, lag in lags.items():
        lag = int(lag)  # Ensure lag is an integer
        
        # Price lags
        df[f'close_lag_{period_name}'] = df['close'].shift(lag)
        df[f'open_lag_{period_name}'] = df['open'].shift(lag)
        df[f'high_lag_{period_name}'] = df['high'].shift(lag)
        df[f'low_lag_{period_name}'] = df['low'].shift(lag)
        
        # Price changes
        df[f'price_change_{period_name}'] = df['close'] / df[f'close_lag_{period_name}'] - 1
        
        # Volume lags
        if 'volume' in df.columns:
            df[f'volume_lag_{period_name}'] = df['volume'].shift(lag)
            df[f'volume_change_{period_name}'] = df['volume'] / df[f'volume_lag_{period_name}'] - 1
    
    return df

def calculate_rsi(df, window=14):
    """
    Calculate RSI manually
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands manually
    """
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    middle_band = rolling_mean
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD manually
    """
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_atr(df, window=14):
    """
    Calculate Average True Range manually
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr

def add_technical_indicators(df):
    """
    Add technical indicators using manual implementations
    """
    # Ensure we have OHLC data
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        print("Missing required OHLC columns for technical indicators")
        return df
    
    # RSI (14 periods)
    df['rsi_14'] = calculate_rsi(df, window=14)
    
    # Bollinger Bands (20 periods, 2 standard deviations)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df, window=20, num_std=2)
    
    # BB width and position
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
    
    # ATR (Average True Range)
    df['atr_14'] = calculate_atr(df, window=14)
    
    return df

def add_time_features(df):
    """
    Add time-based features
    """
    # Extract time components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # Cyclical encoding for hour of day (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Cyclical encoding for month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def add_session_flags(df):
    """
    Add trading session flags based on timestamp
    """
    # Define session hours (UTC)
    # Asian session: 00:00-09:00 UTC
    # London session: 08:00-17:00 UTC
    # New York session: 13:00-22:00 UTC
    
    # Convert index to UTC if it's not already
    df['hour_utc'] = df.index.hour
    
    # Create session flags
    df['asian_session'] = ((df['hour_utc'] >= 0) & (df['hour_utc'] < 9)).astype(int)
    df['london_session'] = ((df['hour_utc'] >= 8) & (df['hour_utc'] < 17)).astype(int)
    df['ny_session'] = ((df['hour_utc'] >= 13) & (df['hour_utc'] < 22)).astype(int)
    
    # Create overlap flags
    df['asian_london_overlap'] = ((df['hour_utc'] >= 8) & (df['hour_utc'] < 9)).astype(int)
    df['london_ny_overlap'] = ((df['hour_utc'] >= 13) & (df['hour_utc'] < 17)).astype(int)
    
    return df

def calculate_news_impact_score(df):
    """
    Calculate news impact score
    """
    # News impact score is already calculated in the merged data
    # But we can create additional features based on it
    
    # Normalize the score
    if 'time_decayed_score' in df.columns:
        max_score = df['time_decayed_score'].abs().max()
        if max_score > 0:
            df['normalized_news_impact'] = df['time_decayed_score'] / max_score
        else:
            df['normalized_news_impact'] = 0
    
    # Create categorical impact levels
    if 'normalized_news_impact' in df.columns:
        conditions = [
            (df['normalized_news_impact'] > 0.5),
            (df['normalized_news_impact'] > 0.2) & (df['normalized_news_impact'] <= 0.5),
            (df['normalized_news_impact'] > -0.2) & (df['normalized_news_impact'] <= 0.2),
            (df['normalized_news_impact'] > -0.5) & (df['normalized_news_impact'] <= -0.2),
            (df['normalized_news_impact'] <= -0.5)
        ]
        
        choices = ['strong_positive', 'positive', 'neutral', 'negative', 'strong_negative']
        df['news_impact_category'] = np.select(conditions, choices, default='neutral')
    
    return df

def create_features(timeframe):
    """
    Create all features for a specific timeframe
    """
    print(f"Creating features for {timeframe} timeframe...")
    
    # Load merged data
    df = load_merged_data(timeframe)
    if df is None:
        return None
    
    # Add lagged features
    df = add_lagged_features(df, timeframe)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add time features
    df = add_time_features(df)
    
    # Add session flags
    df = add_session_flags(df)
    
    # Calculate news impact score
    df = calculate_news_impact_score(df)
    
    # Drop rows with NaN values (due to lagging)
    df_clean = df.dropna()
    print(f"Dropped {len(df) - len(df_clean)} rows with NaN values")
    
    # Save the feature-engineered data
    output_path = os.path.join("Data", f"features_{timeframe}.csv")
    df_clean.to_csv(output_path)
    print(f"Saved feature-engineered data to {output_path}")
    
    return df_clean

def main():
    # Process each timeframe
    timeframes = ['M15', 'H1', 'D1']
    
    for timeframe in timeframes:
        df = create_features(timeframe)
        
        if df is not None:
            # Create a correlation heatmap for the most important features
            try:
                import seaborn as sns
                
                # Select important features
                important_cols = [
                    'close', 'rsi_14', 'bb_position', 'price_change_24h',
                    'sentiment_6h', 'sentiment_24h', 'time_decayed_score',
                    'normalized_news_impact'
                ]
                
                # Filter columns that exist in the dataframe
                existing_cols = [col for col in important_cols if col in df.columns]
                
                if len(existing_cols) > 1:
                    # Create correlation matrix
                    corr_matrix = df[existing_cols].corr()
                    
                    # Plot heatmap
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                    plt.title(f'Feature Correlation Heatmap - {timeframe}')
                    
                    # Save the figure
                    viz_dir = os.path.join("Data", "Visualizations")
                    if not os.path.exists(viz_dir):
                        os.makedirs(viz_dir)
                    plt.savefig(os.path.join(viz_dir, f'feature_correlation_{timeframe}.png'))
                    plt.close()
                    
                    print(f"Created correlation heatmap in {viz_dir}")
            except Exception as e:
                print(f"Error creating correlation heatmap: {e}")
    
    print("\nFeature engineering completed!")

if __name__ == "__main__":
    main()