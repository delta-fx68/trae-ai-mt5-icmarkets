# Hybrid Time Series Forecasting Model for Financial Markets

This project implements a hybrid deep learning model that combines Temporal Fusion Transformers (TFT) and Neural Basis Expansion Analysis for Time Series (N-BEATS) for financial market prediction with news sentiment analysis.

## Overview

The model is designed to forecast financial time series data (specifically for IC Markets) by incorporating both technical indicators and news sentiment. It uses a hybrid architecture that:

1. Processes technical features and news sentiment through a Temporal Fusion Transformer
2. Decomposes the time series into trend, seasonality, and volatility components using N-BEATS
3. Combines these components to make accurate predictions

## Features

- **News-Weighted Loss Function**: Gives higher importance to predictions during periods with significant news impact
- **Hybrid Architecture**: Combines the strengths of TFT (for feature selection) and N-BEATS (for time series decomposition)
- **Interpretable Components**: Separates predictions into trend, seasonality, and volatility components
- **Scalable Design**: Works with varying numbers of technical and news features

## Model Architecture

The model consists of several key components:

- **Temporal Fusion Transformer**: Processes and selects important features from both technical indicators and news sentiment
- **N-BEATS Blocks**: Specialized blocks for modeling trend, seasonality, and volatility
- **Variable Selection Networks**: Dynamically select the most relevant features for prediction
- **Attention Mechanism**: Focuses on the most important time steps and features

## Usage

### Training the Model

```bash
python train_hybrid_model.py
```