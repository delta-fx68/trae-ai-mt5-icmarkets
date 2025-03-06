import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_mt5_data(timeframe):
    """
    Load MT5 data for a specific timeframe
    """
    file_path = os.path.join("Data", f"XAUUSD_{timeframe}.csv")
    if not os.path.exists(file_path):
        print(f"MT5 data file not found: {file_path}")
        return None
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert time column to datetime
    df['datetime'] = pd.to_datetime(df['time'])
    df.set_index('datetime', inplace=True)
    
    return df

def load_news_data():
    """
    Load news data from CSV
    
    Returns:
        DataFrame: News data with datetime index
    """
    file_path = os.path.join("Data", "news_data.csv")
    if not os.path.exists(file_path):
        print(f"News data file not found: {file_path}")
        return None
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert datetime column to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df

def merge_data(mt5_df, news_df, timeframe):
    """
    Merge MT5 data with news data
    
    Args:
        mt5_df (DataFrame): MT5 data
        news_df (DataFrame): News data
        timeframe (str): Timeframe of MT5 data
    
    Returns:
        DataFrame: Merged data
    """
    if mt5_df is None or news_df is None:
        print("Cannot merge data: one or both dataframes are None")
        return None
    
    # Create a resampled news dataframe with the same frequency as MT5 data
    freq_map = {
        'M15': '15min',
        'H1': '1H',
        'D1': '1D'
    }
    
    # Prepare news data for merging
    # First, create a time-indexed dataframe with all news items
    news_time_indexed = news_df.copy()
    news_time_indexed.set_index('datetime', inplace=True)
    
    # Create an empty dataframe with the same time index as MT5 data
    merged_df = mt5_df.copy()
    
    # Add news columns to merged dataframe
    merged_df['news_count'] = 0
    merged_df['sentiment_score'] = 0
    merged_df['weighted_sentiment'] = 0
    merged_df['adjusted_weight_sum'] = 0
    merged_df['alt_adjusted_weight_sum'] = 0
    
    # For each news item, add its influence to the appropriate time periods
    for _, news in news_df.iterrows():
        news_time = pd.to_datetime(news['datetime'])
        
        # Find the closest time index in MT5 data that is not after the news time
        closest_idx = merged_df.index.asof(news_time)
        
        if closest_idx is not None:
            # Add news influence to this and subsequent time periods
            merged_df.loc[closest_idx:, 'news_count'] += 1
            
            # Add weighted sentiment (sentiment * adjusted_weight)
            sentiment = news['sentiment_score']
            weight = news['adjusted_weight']
            alt_weight = news['alt_adjusted_weight']
            
            merged_df.loc[closest_idx:, 'weighted_sentiment'] += sentiment * weight
            merged_df.loc[closest_idx:, 'adjusted_weight_sum'] += weight
            merged_df.loc[closest_idx:, 'alt_adjusted_weight_sum'] += alt_weight
    
    # Calculate average sentiment (avoid division by zero)
    merged_df['avg_sentiment'] = np.where(
        merged_df['adjusted_weight_sum'] > 0,
        merged_df['weighted_sentiment'] / merged_df['adjusted_weight_sum'],
        0
    )
    
    # Calculate alt average sentiment
    merged_df['alt_avg_sentiment'] = np.where(
        merged_df['alt_adjusted_weight_sum'] > 0,
        merged_df['weighted_sentiment'] / merged_df['alt_adjusted_weight_sum'],
        0
    )
    
    # Calculate rolling sentiment averages
    periods_6h = {
        'M15': 24,  # 6 hours / 15 min = 24 periods
        'H1': 6,    # 6 hours / 1 hour = 6 periods
        'D1': 1     # For daily data, use at least 1 period
    }
    
    periods_24h = {
        'M15': 96,  # 24 hours / 15 min = 96 periods
        'H1': 24,   # 24 hours / 1 hour = 24 periods
        'D1': 1     # For daily data, use at least 1 period
    }
    
    # Calculate rolling averages
    merged_df['sentiment_6h'] = merged_df['avg_sentiment'].rolling(
        window=periods_6h[timeframe], min_periods=1
    ).mean()
    
    merged_df['sentiment_24h'] = merged_df['avg_sentiment'].rolling(
        window=periods_24h[timeframe], min_periods=1
    ).mean()
    
    merged_df['alt_sentiment_6h'] = merged_df['alt_avg_sentiment'].rolling(
        window=periods_6h[timeframe], min_periods=1
    ).mean()
    
    merged_df['alt_sentiment_24h'] = merged_df['alt_avg_sentiment'].rolling(
        window=periods_24h[timeframe], min_periods=1
    ).mean()
    
    # Calculate time-decayed score (combining news weight, sentiment, and decay)
    # This is already handled by the adjusted_weight in the news data
    merged_df['time_decayed_score'] = merged_df['avg_sentiment'] * merged_df['adjusted_weight_sum']
    merged_df['alt_time_decayed_score'] = merged_df['alt_avg_sentiment'] * merged_df['alt_adjusted_weight_sum']
    
    return merged_df

def main():
    # Load news data
    print("Loading news data...")
    news_df = load_news_data()
    if news_df is None:
        print("Failed to load news data. Exiting.")
        return
    
    print(f"Loaded {len(news_df)} news items")
    
    # Process each timeframe
    timeframes = ['M15', 'H1', 'D1']  # Include all timeframes
    
    for timeframe in timeframes:
        print(f"\nProcessing {timeframe} timeframe...")
        
        # Load MT5 data
        mt5_df = load_mt5_data(timeframe)
        if mt5_df is None:
            print(f"Skipping {timeframe} timeframe due to missing data")
            continue
        
        print(f"Loaded {len(mt5_df)} {timeframe} candles")
        
        # Merge data
        merged_df = merge_data(mt5_df, news_df, timeframe)
        if merged_df is None:
            print(f"Failed to merge data for {timeframe} timeframe")
            continue
        
        # Save merged data
        output_path = os.path.join("Data", f"merged_data_{timeframe}.csv")
        merged_df.to_csv(output_path)
        print(f"Saved merged data to {output_path}")
        
        # Create a sample visualization
        try:
            import matplotlib.pyplot as plt
            
            # Plot price and sentiment
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Price chart
            ax1.plot(merged_df.index, merged_df['close'], label='Close Price')
            ax1.set_title(f'{timeframe} Price Chart')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # Sentiment chart
            ax2.plot(merged_df.index, merged_df['sentiment_6h'], label='6h Sentiment', color='blue')
            ax2.plot(merged_df.index, merged_df['sentiment_24h'], label='24h Sentiment', color='red')
            ax2.set_title('News Sentiment')
            ax2.set_ylabel('Sentiment Score')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True)
            
            # Save the figure
            viz_dir = os.path.join("Data", "Visualizations")
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            plt.savefig(os.path.join(viz_dir, f'price_sentiment_{timeframe}.png'))
            plt.close()
            
            print(f"Created visualization in {viz_dir}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    print("\nData merging completed!")

if __name__ == "__main__":
    main()