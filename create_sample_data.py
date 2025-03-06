import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample market data for XAUUSD (Gold)
data = []
current_time = datetime.now()
price = 2880.0  # Current XAUUSD price level

for i in range(60):
    # Go back in time
    time_point = current_time - timedelta(hours=59-i)
    
    # Generate random price movements
    open_price = price
    high_price = open_price + np.random.uniform(0, 5)
    low_price = max(open_price - np.random.uniform(0, 5), 0)  # Ensure no negative prices
    close_price = np.random.uniform(low_price, high_price)
    volume = int(np.random.uniform(1000, 5000))
    
    # Add to data - ensure all price values are numeric
    data.append({
        'datetime': time_point.strftime('%Y-%m-%d %H:%M'),
        'open': round(float(open_price), 2),
        'high': round(float(high_price), 2),
        'low': round(float(low_price), 2),
        'close': round(float(close_price), 2),
        'volume': int(volume)
    })
    
    # Update price for next iteration
    price = close_price

# Create DataFrame
df = pd.DataFrame(data)

# Verify no non-numeric values in price columns
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Replace any NaN values with appropriate defaults
    if df[col].isna().any():
        print(f"Warning: Found non-numeric values in {col} column, replacing with defaults")
        if col == 'volume':
            df[col].fillna(1000, inplace=True)
        else:
            df[col].fillna(price, inplace=True)

# Save to CSV - ensure no string values in numeric columns
df.to_csv("C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\current_market_data.csv", index=False)
print("Sample XAUUSD market data created successfully!")

# Create a simple model stats file
with open("C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\model_stats.txt", "w") as f:
    f.write("2880.0,50.0")  # Mean and scale for XAUUSD
print("Model stats file created!")