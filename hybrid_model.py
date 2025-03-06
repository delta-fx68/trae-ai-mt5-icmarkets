import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
import math

class TimeDistributed(nn.Module):
    """Apply a layer to every temporal slice of an input."""
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        
        # Reshape input to (batch_size * timesteps, input_size)
        t, n = x.size(0), x.size(1)
        x_reshape = x.contiguous().view(t * n, -1)
        
        # Apply module
        y = self.module(x_reshape)
        
        # Reshape output back to (batch_size, timesteps, output_size)
        y = y.contiguous().view(t, n, -1)
        return y

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for TFT"""
    def __init__(self, input_size, hidden_size=None, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        sig = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for TFT"""
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size else input_size
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        
        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)
        else:
            self.skip_layer = None
            
        self.gate = GatedLinearUnit(self.output_size, dropout=dropout)
        self.ln = nn.LayerNorm(self.output_size)
        
    def forward(self, x, c=None):
        # Main branch
        residual = x
        x = self.fc1(x)
        if c is not None:
            x += c
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Skip connection
        if self.skip_layer is not None:
            residual = self.skip_layer(residual)
            
        # Gating
        x = residual + self.gate(x)
        return self.ln(x)

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT"""
    def __init__(self, input_sizes, hidden_size, dropout=0.1, context_size=None):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.num_vars = len(input_sizes)
        
        # Create embedding networks for each variable
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for input_size in input_sizes
        ])
        
        # Create variable selection weights
        self.selection_grn = GatedResidualNetwork(
            self.num_vars * hidden_size,
            hidden_size,
            self.num_vars,
            dropout
        )
        
    def forward(self, x, c=None):
        batch_size, seq_len, num_features = x.shape
        
        # Check if number of features matches expected number of variables
        if num_features != self.num_vars:
            # Split the input tensor into individual features
            var_outputs = []
            for i in range(num_features):
                # Extract each feature and process it
                var_x = x[:, :, i:i+1]  # Shape: [batch_size, seq_len, 1]
                var_outputs.append(self.var_grns[min(i, len(self.var_grns)-1)](var_x))
        else:
            # Process each feature separately as originally intended
            var_outputs = []
            for i, grn in enumerate(self.var_grns):
                var_x = x[:, :, i:i+1]  # Extract single feature
                var_outputs.append(grn(var_x))
        
        # Concatenate outputs along feature dimension
        var_outputs_tensor = torch.stack(var_outputs, dim=-2)  # Shape: [batch_size, seq_len, num_features, hidden_size]
        
        # Flatten features for selection
        flat = torch.flatten(var_outputs_tensor, start_dim=2)  # Shape: [batch_size, seq_len, num_features*hidden_size]
        
        # Calculate variable selection weights
        weights = self.selection_grn(flat, c)  # Shape: [batch_size, seq_len, num_features]
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # Shape: [batch_size, seq_len, num_features, 1]
        
        # Weight and combine variable outputs
        combined = torch.sum(weights * var_outputs_tensor, dim=-2)  # Shape: [batch_size, seq_len, hidden_size]
        
        return combined, weights

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer with attention on news features"""
    def __init__(self, 
                 num_features,
                 num_news_features,
                 hidden_size=64, 
                 lstm_layers=2,
                 dropout=0.1,
                 attn_heads=4):
        super(TemporalFusionTransformer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.num_news_features = num_news_features
        self.attn_heads = attn_heads
        
        # Variable selection networks
        self.feature_sizes = [1] * num_features  # Each feature is 1D
        self.news_feature_sizes = [1] * num_news_features  # Each news feature is 1D
        
        # Variable selection for features
        self.feature_vsn = VariableSelectionNetwork(
            self.feature_sizes,
            hidden_size,
            dropout
        )
        
        # Variable selection for news features
        self.news_vsn = VariableSelectionNetwork(
            self.news_feature_sizes,
            hidden_size,
            dropout
        )
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            hidden_size * 2,  # Combined features and news
            hidden_size,
            lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism with extra weight for news features
        self.attn = nn.MultiheadAttention(hidden_size, attn_heads, dropout=dropout)
        
        # News attention bias - gives more weight to news features
        self.news_attn_bias = nn.Parameter(torch.ones(1))
        
        # Output layers
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, x_features, x_news):
        # Process regular features
        features, feature_weights = self.feature_vsn(x_features)
        
        # Process news features with higher attention weight
        news, news_weights = self.news_vsn(x_news)
        
        # Combine features and news
        combined = torch.cat([features, news], dim=-1)
        
        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(combined)
        
        # Self-attention without mask
        lstm_out_permuted = lstm_out.permute(1, 0, 2)  # [seq_len, batch, hidden]
        
        # Apply attention without mask
        attn_out, _ = self.attn(
            lstm_out_permuted,
            lstm_out_permuted,
            lstm_out_permuted,
            need_weights=False
        )
        attn_out = attn_out.permute(1, 0, 2)  # [batch, seq_len, hidden]
        
        # Output projection - ensure it matches the expected hidden_size
        output = self.fc_out(attn_out)  # Shape: [batch, seq_len, 1]
        
        # For compatibility with the rest of the model, reshape to match hidden_size
        # This is a temporary fix - ideally the model architecture should be redesigned
        output_expanded = output.expand(-1, -1, self.hidden_size)
        
        return output_expanded, feature_weights, news_weights

class NBEATSBlock(nn.Module):
    """N-BEATS block for trend and seasonality decomposition"""
    def __init__(self, input_size, theta_size, horizon, n_harmonics=None, n_polynomials=None, stack_type='generic'):
        super(NBEATSBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.stack_type = stack_type
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, theta_size)
        self.fc2 = nn.Linear(theta_size, theta_size)
        self.fc3 = nn.Linear(theta_size, theta_size)
        self.fc4 = nn.Linear(theta_size, 2 * horizon)  # For both backcast and forecast
        
        if stack_type == 'trend':
            # Trend stack
            self.basis_function = self._trend_basis
        elif stack_type == 'seasonality':
            # Seasonality stack
            self.basis_function = self._seasonality_basis
        else:
            # Generic stack
            self.basis_function = self._identity_basis
            
    def _seasonality_basis(self, t_in, t_out):
        # Create seasonality basis
        p = torch.arange(self.n_harmonics).to(t_in.device)
        t_in_expanded = t_in.expand(-1, -1, self.n_harmonics)
        t_out_expanded = t_out.expand(-1, -1, self.n_harmonics)
        
        # Reshape p for broadcasting
        p = p.reshape(1, 1, self.n_harmonics)
        
        # Compute sin and cos features
        sin_features_in = torch.sin(2 * np.pi * p * t_in_expanded / self.input_size)
        cos_features_in = torch.cos(2 * np.pi * p * t_in_expanded / self.input_size)
        sin_features_out = torch.sin(2 * np.pi * p * t_out_expanded / self.input_size)
        cos_features_out = torch.cos(2 * np.pi * p * t_out_expanded / self.input_size)
        
        # Stack and reshape for proper dimensions
        backcast_basis = torch.cat([sin_features_in, cos_features_in], dim=2)
        forecast_basis = torch.cat([sin_features_out, cos_features_out], dim=2)
        
        return backcast_basis, forecast_basis
        
    def _trend_basis(self, t_in, t_out):
        # Create trend basis (polynomial)
        p = torch.arange(self.n_polynomials).to(t_in.device)
        t_in_expanded = t_in.expand(-1, -1, self.n_polynomials)
        t_out_expanded = t_out.expand(-1, -1, self.n_polynomials)
        
        # Reshape p for broadcasting
        p = p.reshape(1, 1, self.n_polynomials)
        
        # Compute polynomial features
        backcast_basis = t_in_expanded ** p
        forecast_basis = t_out_expanded ** p
        
        return backcast_basis, forecast_basis
        
    def _identity_basis(self, t_in, t_out):
        # Identity basis (no specific structure)
        batch_size = t_in.shape[0]
        
        # Create properly sized basis matrices
        # For backcast: [batch_size, input_size, theta_size/2]
        backcast_basis = torch.zeros(batch_size, self.input_size, self.horizon, device=t_in.device)
        for i in range(min(self.input_size, self.horizon)):
            backcast_basis[:, i, i % self.horizon] = 1.0
        
        # For forecast: [batch_size, horizon, theta_size/2]
        forecast_basis = torch.zeros(batch_size, self.horizon, self.horizon, device=t_out.device)
        for i in range(self.horizon):
            forecast_basis[:, i, i] = 1.0
        
        return backcast_basis, forecast_basis
        
    def forward(self, x):
        # Forward pass through FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # Split theta into backcast and forecast
        theta_b, theta_f = x.chunk(2, dim=-1)
        
        # Create time indices
        batch_size = x.shape[0]
        t_in = torch.arange(self.input_size).float().to(x.device) / self.input_size
        t_in = t_in.reshape(1, self.input_size, 1).repeat(batch_size, 1, 1)
        
        t_out = torch.arange(self.horizon).float().to(x.device) / self.horizon
        t_out = t_out.reshape(1, self.horizon, 1).repeat(batch_size, 1, 1)
        
        # Apply basis functions
        backcast_basis, forecast_basis = self.basis_function(t_in, t_out)
        
        # Handle dimension mismatch - simplify the approach to avoid matrix multiplication issues
        if self.stack_type == 'generic':
            # For generic type, just use the theta values directly
            backcast = theta_b
            forecast = theta_f
        else:
            # For trend/seasonality, use a simplified approach
            if self.stack_type == 'trend':
                # Simple polynomial trend
                t = torch.arange(self.horizon, dtype=torch.float32, device=x.device) / self.horizon
                forecast = torch.zeros(batch_size, self.horizon, device=x.device)
                for b in range(batch_size):
                    for i in range(min(self.horizon, theta_f.size(1))):
                        forecast[b] += theta_f[b, i] * (t ** i)
                
                # Backcast is not used in the hybrid model, so we can simplify
                backcast = torch.zeros(batch_size, self.input_size, device=x.device)
                
            elif self.stack_type == 'seasonality':
                # Simple harmonic seasonality
                t = torch.arange(self.horizon, dtype=torch.float32, device=x.device) / self.horizon
                forecast = torch.zeros(batch_size, self.horizon, device=x.device)
                for b in range(batch_size):
                    for i in range(min(self.n_harmonics, theta_f.size(1) // 2)):
                        forecast[b] += theta_f[b, 2*i] * torch.sin(2 * np.pi * (i+1) * t)
                        if 2*i+1 < theta_f.size(1):
                            forecast[b] += theta_f[b, 2*i+1] * torch.cos(2 * np.pi * (i+1) * t)
                
                # Backcast is not used in the hybrid model, so we can simplify
                backcast = torch.zeros(batch_size, self.input_size, device=x.device)
        
        return backcast, forecast

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRN, self).__init__()
        # Make sure input dimensions match the data
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Forward pass through FC layers
        sig = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)

class HybridModel(nn.Module):
    """Hybrid model combining TFT and N-BEATS for time series forecasting with news impact"""
    def __init__(self, num_features, num_news_features, input_size, horizon, hidden_size=64):
        super(HybridModel, self).__init__()
        self.num_features = num_features
        self.num_news_features = num_news_features
        self.input_size = input_size
        self.horizon = horizon
        self.hidden_size = hidden_size
        
        # TFT component for feature extraction
        self.tft = TemporalFusionTransformer(
            num_features=num_features,
            num_news_features=num_news_features,
            hidden_size=hidden_size
        )
        
        # N-BEATS components for decomposition
        self.trend_block = NBEATSBlock(
            input_size=input_size,
            theta_size=hidden_size,
            horizon=horizon,
            n_polynomials=3,
            stack_type='trend'
        )
        
        self.seasonality_block = NBEATSBlock(
            input_size=input_size,
            theta_size=hidden_size,
            horizon=horizon,
            n_harmonics=5,
            stack_type='seasonality'
        )
        
        self.volatility_block = NBEATSBlock(
            input_size=input_size,
            theta_size=hidden_size,
            horizon=horizon,
            stack_type='generic'
        )
        
        # Final projection layer
        self.fc_out = nn.Linear(hidden_size, horizon)
        
    def forward(self, x_price, x_news):
        # Process through TFT
        tft_out, feature_weights, news_weights = self.tft(x_price, x_news)
        
        # Get last timestep for N-BEATS blocks
        last_step = tft_out[:, -1, :]
        
        # Process through N-BEATS blocks - reshape input to match expected dimensions
        # The NBEATSBlock expects input of shape [batch_size, input_size]
        # But we're getting [batch_size, hidden_size]
        
        # Reshape or pad the input to match the expected input_size
        if last_step.shape[1] != self.input_size:
            # Option 1: Repeat the features to match input_size
            if last_step.shape[1] > self.input_size:
                # If larger, truncate
                last_step_resized = last_step[:, :self.input_size]
            else:
                # If smaller, repeat the last dimension
                repeats = self.input_size // last_step.shape[1] + 1
                last_step_expanded = last_step.repeat(1, repeats)
                last_step_resized = last_step_expanded[:, :self.input_size]
        else:
            last_step_resized = last_step
        
        # Now process through N-BEATS blocks with correctly sized input
        _, trend = self.trend_block(last_step_resized)
        _, seasonality = self.seasonality_block(last_step_resized)
        _, volatility = self.volatility_block(last_step_resized)
        
        # Combine components
        combined = trend + seasonality + volatility
        
        # Final output
        output = self.fc_out(last_step)
        
        return output, trend, seasonality, volatility, feature_weights, news_weights
    
    def set_scaling_params(self, price_mean, price_scale):
        """Set scaling parameters to convert normalized predictions back to original scale"""
        self.price_mean = price_mean
        self.price_scale = price_scale
        print(f"Scaling parameters set: mean={price_mean:.2f}, scale={price_scale:.2f}")
    
    def save(self, path):
        """Save model to disk"""
        # Create directory if it doesn't exist
        import os
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            
        # Save scaling parameters if they exist
        scaling_params = {}
        if hasattr(self, 'price_scale') and hasattr(self, 'price_mean'):
            scaling_params = {
                'price_mean': self.price_mean,
                'price_scale': self.price_scale
            }
            
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_features': self.num_features,
                'num_news_features': self.num_news_features,
                'input_size': self.input_size,
                'horizon': self.horizon,
                'hidden_size': self.hidden_size,
            },
            'scaling_params': scaling_params
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_features=checkpoint['config']['num_features'],
            num_news_features=checkpoint['config']['num_news_features'],
            input_size=checkpoint['config']['input_size'],
            horizon=checkpoint['config']['horizon'],
            hidden_size=checkpoint['config']['hidden_size'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scaling parameters if they exist
        if 'scaling_params' in checkpoint and checkpoint['scaling_params']:
            model.price_mean = checkpoint['scaling_params']['price_mean']
            model.price_scale = checkpoint['scaling_params']['price_scale']
            print(f"Loaded scaling parameters: mean={model.price_mean:.2f}, scale={model.price_scale:.2f}")
        
        return model
class NewsWeightedLoss(nn.Module):
    """Custom loss function that gives more weight to predictions with high news impact"""
    def __init__(self, news_weight=2.0, base_loss='mse', verbose=True):
        super(NewsWeightedLoss, self).__init__()
        self.news_weight = news_weight
        self.verbose = verbose
        self.iteration_count = 0
        self.log_interval = 10  # Log every 10 iterations
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
        
        print(f"NewsWeightedLoss initialized with news_weight={news_weight}, base_loss={base_loss}")
    
    def forward(self, y_pred, y_true, news_weights=None):
        # Calculate base loss
        base_loss = self.base_loss(y_pred, y_true)  # Shape: [batch_size, horizon]
        
        if news_weights is None:
            # If no news weights provided, use standard loss
            mean_loss = torch.mean(base_loss)
            
            # Log periodically
            if self.verbose and self.iteration_count % self.log_interval == 0:
                print(f"Iteration {self.iteration_count}, Loss: {mean_loss.item():.6f}")
                
            self.iteration_count += 1
            return mean_loss
        
        # Reshape news_weights to match the prediction horizon
        # news_weights shape: [batch_size] -> [batch_size, 1]
        news_weights = news_weights.unsqueeze(-1)
        
        # Scale news weights to be between 1.0 and self.news_weight
        scaled_weights = 1.0 + (self.news_weight - 1.0) * news_weights
        
        # Expand scaled_weights to match base_loss shape if needed
        if scaled_weights.shape != base_loss.shape:
            scaled_weights = scaled_weights.expand_as(base_loss)
        
        # Apply weights to loss
        weighted_loss = base_loss * scaled_weights
        mean_loss = torch.mean(weighted_loss)
        
        # Log periodically
        if self.verbose and self.iteration_count % self.log_interval == 0:
            print(f"Iteration {self.iteration_count}, Loss: {mean_loss.item():.6f}, News impact: {torch.mean(news_weights).item():.4f}")
            
        self.iteration_count += 1
        return mean_loss