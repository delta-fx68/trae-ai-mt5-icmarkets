import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hybrid_model import HybridModel, NewsWeightedLoss

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
TIMEFRAME = 'H1'  # Use H1 data for training
INPUT_WINDOW = 60  # 60 time steps of history
FORECAST_HORIZON = 10  # Predict 10 steps ahead
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15  # Early stopping patience
TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2022-12-31'
VAL_START_DATE = '2023-01-01'
VAL_END_DATE = '2023-06-30'
TEST_START_DATE = '2023-07-01'
TEST_END_DATE = '2025-02-28'  # Or current date
MODEL_SAVE_PATH = 'c:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models\\gold_predictor.pt'
ONNX_SAVE_PATH = 'c:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models\\gold_predictor.onnx'

# Keywords for increased loss weighting
WAR_INFLATION_KEYWORDS = [
    'war', 'conflict', 'inflation', 'cpi', 'price index', 'interest rate', 
    'fed', 'federal reserve', 'ecb', 'central bank', 'hike', 'recession',
    'sanctions', 'ukraine', 'russia', 'military', 'attack'
]

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with news features"""
    def __init__(self, features_df, start_date=None, end_date=None):
        # Filter by date range if provided
        if start_date and end_date:
            mask = (features_df.index >= start_date) & (features_df.index <= end_date)
            self.df = features_df[mask].copy()
        else:
            self.df = features_df.copy()
        
        # Print column names for debugging
        print(f"Available columns: {self.df.columns.tolist()}")
        
        # Check if 'close' column exists
        if 'close' not in self.df.columns:
            raise ValueError("Dataset must contain a 'close' column")
            
        # Ensure all columns are numeric (except datetime)
        for col in self.df.columns:
            if pd.api.types.is_object_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                    print(f"Converted column '{col}' to numeric")
                except ValueError:
                    # If conversion fails, drop the column
                    print(f"Dropping non-numeric column: {col}")
                    self.df = self.df.drop(columns=[col])
            
        # Separate price/technical features from news features
        self.price_cols = [col for col in self.df.columns if not col.startswith(('sentiment', 'news', 'time_decayed', 'weighted'))]
        self.news_cols = [col for col in self.df.columns if col.startswith(('sentiment', 'news', 'time_decayed', 'weighted'))]
        
        print(f"Price columns: {self.price_cols}")
        print(f"News columns: {self.news_cols}")
        
        # If no news columns found, create a dummy one
        if len(self.news_cols) == 0:
            print("No news features found. Creating a dummy news feature.")
            self.df['dummy_news'] = 0.0
            self.news_cols = ['dummy_news']
        
        # Store original price mean and scale for reference but don't normalize
        self.price_mean = self.df['close'].mean()
        self.price_scale = self.df['close'].std()
        print(f"Gold price mean: {self.price_mean:.2f}, scale: {self.price_scale:.2f}")
        
        # Scale news features if they exist
        if len(self.news_cols) > 0 and 'dummy_news' not in self.news_cols:
            self.news_scaler = StandardScaler()
            news_data = self.df[self.news_cols].values
            self.df[self.news_cols] = self.news_scaler.fit_transform(news_data)
        
        # Store indices for sequences instead of creating them all at once
        self.sequence_indices = []
        for i in range(len(self.df) - INPUT_WINDOW - FORECAST_HORIZON + 1):
            self.sequence_indices.append(i)
            
        print(f"Created {len(self.sequence_indices)} sequences")
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        # Get the starting index for this sequence
        start_idx = self.sequence_indices[idx]
        
        # Extract data for this sequence
        x_price = self.df[self.price_cols].values[start_idx:start_idx+INPUT_WINDOW]
        x_news = self.df[self.news_cols].values[start_idx:start_idx+INPUT_WINDOW]
        
        # Target sequence (future close prices)
        y = self.df['close'].values[start_idx+INPUT_WINDOW:start_idx+INPUT_WINDOW+FORECAST_HORIZON]
        
        # News impact score for weighting the loss
        # Check if any war/inflation keywords are in the news
        news_impact = 0.0
        for col in self.news_cols:
            if any(keyword in col.lower() for keyword in WAR_INFLATION_KEYWORDS):
                # If keyword found, use the maximum absolute value as impact
                impact = np.max(np.abs(x_news[:, self.news_cols.index(col)]))
                news_impact = max(news_impact, impact)
        
        # Convert to tensors
        x_price = torch.tensor(x_price, dtype=torch.float32)
        x_news = torch.tensor(x_news, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        news_impact = torch.tensor(news_impact, dtype=torch.float32)
        
        return x_price, x_news, y, news_impact

def load_data():
    """Load and prepare data for training"""
    # Load feature-engineered data
    features_path = os.path.join("Data", f"features_{TIMEFRAME}.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    # Load data
    df = pd.read_csv(features_path)
    
    # Convert datetime column to datetime with flexible format
    try:
        # First try with default format
        df['datetime'] = pd.to_datetime(df['datetime'])
    except ValueError:
        # If that fails, try with a more flexible approach
        print("Flexible datetime parsing...")
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df

def create_dataloaders(df):
    """Create train, validation, and test dataloaders"""
    # Create datasets
    train_dataset = TimeSeriesDataset(df, TRAIN_START_DATE, TRAIN_END_DATE)
    val_dataset = TimeSeriesDataset(df, VAL_START_DATE, VAL_END_DATE)
    test_dataset = TimeSeriesDataset(df, TEST_START_DATE, TEST_END_DATE)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device):
    """Train the model with early stopping"""
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    criterion = NewsWeightedLoss(news_weight=2.0)  # Changed from news_weight_factor to news_weight
    
    # Set scaling parameters for the model to output gold prices in original scale
    # Get the scaling parameters from the first dataset in the dataloader
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'price_mean') and hasattr(train_dataset, 'price_scale'):
        model.set_scaling_params(train_dataset.price_mean, train_dataset.price_scale)
        print(f"Set model scaling: Gold price mean = {train_dataset.price_mean:.2f}, scale = {train_dataset.price_scale:.2f}")
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x_price, x_news, y_true, news_impact = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            y_pred, _, _, _, _, _ = model(x_price, x_news)
            loss = criterion(y_pred, y_true, news_impact)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_price, x_news, y_true, news_impact = [b.to(device) for b in batch]
                y_pred, _, _, _, _, _ = model(x_price, x_news)
                loss = criterion(y_pred, y_true, news_impact)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            model.save(MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Create directory if it doesn't exist
    viz_dir = os.path.join("Data", "Visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    plt.savefig(os.path.join(viz_dir, 'training_loss.png'))
    plt.close()
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_price, x_news, y_true, _ = [b.to(device) for b in batch]
            y_pred, trend, seasonality, volatility, _, _ = model(x_price, x_news)
            loss = criterion(y_pred, y_true)
            test_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Plot predictions vs targets for a sample
    plt.figure(figsize=(12, 6))
    sample_idx = np.random.randint(0, len(all_preds))
    plt.plot(all_targets[sample_idx], label='Actual')
    plt.plot(all_preds[sample_idx], label='Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.title('Predicted vs Actual Prices')
    plt.legend()
    
    # Create directory if it doesn't exist
    viz_dir = os.path.join("Data", "Visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    plt.savefig(os.path.join(viz_dir, 'prediction_sample.png'))
    plt.close()
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame()
    
    # For each prediction horizon step
    for i in range(FORECAST_HORIZON):
        predictions_df[f'actual_t+{i+1}'] = all_targets[:, i]
        predictions_df[f'predicted_t+{i+1}'] = all_preds[:, i]
        predictions_df[f'error_t+{i+1}'] = all_targets[:, i] - all_preds[:, i]
        predictions_df[f'pct_error_t+{i+1}'] = (all_targets[:, i] - all_preds[:, i]) / all_targets[:, i] * 100
    
    # Save to CSV
    csv_path = os.path.join("Data", "predictions_results.csv")
    predictions_df.to_csv(csv_path)
    
    return predictions_df

def export_to_onnx(model, input_size, device):
    """Export the model to ONNX format for deployment"""
    # Create dummy inputs
    dummy_price = torch.randn(1, input_size, model.num_features, device=device)
    dummy_news = torch.randn(1, input_size, model.num_news_features, device=device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(ONNX_SAVE_PATH), exist_ok=True)
    
    # Export to ONNX
    torch.onnx.export(
        model,                       # model being run
        (dummy_price, dummy_news),   # model input (or a tuple for multiple inputs)
        ONNX_SAVE_PATH,              # where to save the model
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=14,            # Increased from 13 to 14 to support scaled_dot_product_attention
        do_constant_folding=True,    # whether to execute constant folding for optimization
        input_names=['price_input', 'news_input'],   # the model's input names
        output_names=['forecast', 'trend', 'seasonality', 'volatility', 'feature_weights', 'news_weights'],  # the model's output names
        dynamic_axes={
            'price_input': {0: 'batch_size', 1: 'sequence'},    # variable length axes
            'news_input': {0: 'batch_size', 1: 'sequence'},
            'forecast': {0: 'batch_size'},
            'trend': {0: 'batch_size'},
            'seasonality': {0: 'batch_size'},
            'volatility': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {ONNX_SAVE_PATH}")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(ONNX_SAVE_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
        # Print model metadata
        print(f"ONNX model IR version: {onnx_model.ir_version}")
        print(f"ONNX model opset version: {onnx_model.opset_import[0].version}")
        print(f"ONNX model producer: {onnx_model.producer_name}")
        
    except ImportError:
        print("ONNX package not installed. Skipping verification.")
        print("You can install it with: pip install onnx")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
        print("This doesn't necessarily mean the model is invalid, but you may want to check it.")
def main():
    """Main function to run the training pipeline"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check CUDA version if available
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        if torch.backends.cudnn.enabled:
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(df)
    
    # Get feature dimensions from the first batch
    sample_batch = next(iter(train_loader))
    num_features = sample_batch[0].shape[2]  # Price features
    num_news_features = sample_batch[1].shape[2]  # News features
    
    print(f"Feature dimensions: {num_features} price features, {num_news_features} news features")
    
    # Initialize model
    print("Initializing model...")
    model = HybridModel(
        num_features=num_features,
        num_news_features=num_news_features,
        input_size=INPUT_WINDOW,
        horizon=FORECAST_HORIZON,
        hidden_size=64
    ).to(device)
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, device)
    
    # Evaluate model
    print("Evaluating model...")
    predictions_df = evaluate_model(model, test_loader, device)
    
    # Export model to ONNX
    print("Exporting model to ONNX...")
    export_to_onnx(model, INPUT_WINDOW, device)
    
    print("Done!")
    return model, predictions_df

if __name__ == "__main__":
    main()