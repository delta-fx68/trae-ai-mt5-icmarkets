import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from sklearn.metrics import mean_squared_error

class AutoTrainer:
    def __init__(self):
        self.data_dir = r"C:\Users\andre\Desktop\trae ai mt5 icmarkets\Data"
        self.predictions_file = os.path.join(self.data_dir, "prediction_results.csv")
        self.market_data_file = os.path.join(self.data_dir, "current_market_data.csv")
        self.performance_log = os.path.join(self.data_dir, "model_performance.csv")
        
    def evaluate_predictions(self):
        """Compare predictions with actual values and calculate error"""
        try:
            # Load predictions
            predictions_df = pd.read_csv(self.predictions_file)
            predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
            
            # Load actual market data
            market_df = pd.read_csv(self.market_data_file)
            market_df['datetime'] = pd.to_datetime(market_df['datetime'])
            
            # Match predictions with actual values
            merged_df = predictions_df.merge(market_df, 
                                          on='datetime', 
                                          how='left', 
                                          suffixes=('_pred', '_actual'))
            
            # Calculate error
            valid_rows = merged_df.dropna(subset=['predicted', 'close'])
            if len(valid_rows) == 0:
                print("No matching data points found for evaluation")
                return None
                
            mse = mean_squared_error(valid_rows['predicted'], valid_rows['close'])
            
            # Log performance
            self.log_performance(mse)
            
            return mse
        except Exception as e:
            print(f"Error evaluating predictions: {e}")
            return None
    
    def log_performance(self, mse):
        """Log model performance"""
        log_entry = pd.DataFrame({
            'datetime': [datetime.now()],
            'mse': [mse],
        })
        
        if os.path.exists(self.performance_log):
            log_df = pd.read_csv(self.performance_log)
            log_df = pd.concat([log_df, log_entry], ignore_index=True)
        else:
            log_df = log_entry
            
        log_df.to_csv(self.performance_log, index=False)
    
    def should_retrain(self):
        """Determine if model should be retrained based on performance"""
        if not os.path.exists(self.performance_log):
            return False
            
        log_df = pd.read_csv(self.performance_log)
        if len(log_df) < 10:  # Need minimum history
            return False
            
        # Calculate trend in error
        recent_errors = log_df['mse'].tail(10)
        error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        # Retrain if error is trending up
        return error_trend > 0

def main():
    trainer = AutoTrainer()
    
    while True:
        # Wait for new predictions
        time.sleep(3600)  # Check every hour
        
        # Evaluate current predictions
        mse = trainer.evaluate_predictions()
        if mse is None:
            continue
            
        print(f"Current MSE: {mse}")
        
        # Check if retraining is needed
        if trainer.should_retrain():
            print("Performance degrading, retraining recommended")
            # Here you would implement the retraining logic
            # This could involve collecting new training data
            # and updating the model

if __name__ == "__main__":
    main()