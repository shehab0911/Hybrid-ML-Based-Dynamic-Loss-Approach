
================================================================================
Hybrid ML-Based Dynamic Stop-Loss Approach for 1-Minute ES Futures Trading
================================================================================

📌 Project Description:
--------------------------------------------------------------------------------
This project implements a hybrid machine learning system for dynamic stop-loss 
prediction and adaptive trade entry for ES futures on a 1-minute timeframe.

It includes:
- LightGBM Regression to predict stop-loss levels dynamically
- LightGBM Classification to generate trade entry signals
- KMeans clustering to detect market regimes
- Volatility-aware technical indicators (ATR, RSI, MACD, BB, etc.)
- Dynamic position sizing based on predicted stop-loss
- Backtesting using vectorbt for realistic strategy evaluation

================================================================================
📂 Folder Structure (Recommended)
--------------------------------------------------------------------------------
ML/
│
├── algorithm.py                      # Main script with entry, SL, and backtesting
├── requirements.txt                  # Required Python packages
├── README.txt                        # This instruction file
├── .gitignore                        # Specifies ignored files/folders
└── SP candles/                       # Folder containing your 1-min candle CSV files

================================================================================
⚙️ Requirements
--------------------------------------------------------------------------------
Python >= 3.9

Install all dependencies using pip:
> pip install -r requirements.txt

================================================================================
📦 Virtual Environment Setup (Recommended)
--------------------------------------------------------------------------------
1. Create virtual environment:
> python -m venv myenv

2. Activate it:
- Windows:
> .\myenv\Scripts\activate
- Mac/Linux:
> source myenv/bin/activate

3. Install packages:
> pip install -r requirements.txt

================================================================================
🚀 How to Run the Project
--------------------------------------------------------------------------------
1. Place your 1-minute candle CSV files inside the folder: `SP candles/`

2. Run the main script:
> python algorithm.py

3. The script will:
   - Load and preprocess candle data
   - Engineer relevant features and detect market regimes
   - Train LightGBM models for stop-loss and entry signal prediction
   - Perform realistic backtesting using vectorbt
   - Print out:
     • Backtest performance (Sharpe Ratio, Drawdown, Return, Win Rate)
     • Stop-loss and entry signal prediction for the latest candle

================================================================================
📊 Sample Output
--------------------------------------------------------------------------------
--- Backtest Results ---
sharpe_ratio: 1.32
total_return: 14.87%
max_drawdown: -5.21%
win_rate: 63.45%

Latest Candle Predictions:
Predicted Stop-Loss Distance: 2.8500
Entry Signal Probability: 0.6821
Position Size: 4 contracts

================================================================================
💡 Additional Notes
--------------------------------------------------------------------------------
- Avoid pushing large raw files (CSV, models, environments) to GitHub.
- Use `.gitignore` to exclude `SP candles/`, `myenv/`, `.ipynb_checkpoints/`, etc.
- You can increase `n_trials` in Optuna for better model performance.
- This script can be integrated with live trading APIs for real-time execution.

================================================================================
🔗 Repository:
--------------------------------------------------------------------------------
GitHub: https://github.com/shehab0911/Hybrid-ML-Based-Dynamic-Loss-Approach

================================================================================
👤 Author: Shehab
--------------------------------------------------------------------------------
=======
# Complete Production ML Trading System

## Overview

This is a comprehensive machine learning-based trading system designed to address the challenges of static stop-loss and entry logic in 1-minute ES futures trading. The system implements dynamic risk management and entry confidence scoring using advanced ML models to optimize trading performance during volatility shifts.

## Problem Statement

The original trading algorithm suffered from:
- Static stop-loss levels that didn't adapt to market volatility
- Fixed entry logic that performed poorly during regime changes
- Lack of dynamic position sizing based on market conditions
- Poor performance metrics with negative Sharpe ratios

## Solution Architecture

This ML-enhanced system provides:

### Core ML Models
1. **Volatility Forecasting Model**: Predicts future volatility using GARCH-style features
2. **Dynamic Stop-Loss Model**: Calculates optimal stop-loss distances based on market conditions
3. **Entry Confidence Model**: Scores entry opportunities using regime-aware features
4. **Position Sizing Model**: Determines optimal position sizes based on predicted volatility
5. **Regime Detection Model**: Identifies market regimes using HMM or K-means clustering

### Key Features
- **Regime-Aware Trading**: Adapts strategy based on detected market regimes (low/medium/high volatility)
- **Advanced Feature Engineering**: 50+ technical and microstructure features
- **Walk-Forward Validation**: Time-series aware model validation
- **Real-Time Dashboard**: Live monitoring with Plotly/Dash
- **Production-Ready**: Includes live trading integration with Interactive Brokers
- **Comprehensive Risk Management**: Advanced drawdown control and position sizing

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies (for full functionality)
```bash
# For regime detection (HMM)
pip install hmmlearn

# For real-time dashboard
pip install dash plotly

# For live trading
pip install ib_insync

# For advanced analytics
pip install shap
```

## Data Structure

The system expects data in the following structure:
```
Safe code/
├── CandleIndicators/          # OHLCV candle data
│   └── SP500_candels_2010.csv
├── Downloads_SP500/           # Position/trade data (optional)
│   └── SP500_2010_1.csv
└── complete_production.py     # Main system file
```

### Data Format
- **Candle Data**: CSV files with OHLCV data and timestamps
- **Position Data**: Optional CSV files with actual trading positions
- **Timestamp Format**: Flexible parsing supports multiple datetime formats

## Usage

### Basic Usage
```python
from complete_production import CompleteProductionMLSystem

# Initialize system
system = CompleteProductionMLSystem()

# Load data
df = system.load_complete_dataset('CandleIndicators/', 'Downloads_SP500/')

# Engineer features
df = system.engineer_advanced_features(df)

# Train models
system.train_regime_detection_model(df)
system.train_volatility_forecasting_model(df)
system.tune_hyperparameters(df)

# Run backtest
backtest_results = system.enhanced_backtest_system(df)

# Generate report
system.generate_comprehensive_report(backtest_results)
```

### Advanced Usage
```python
# With walk-forward validation
walk_forward_results = system.walk_forward_validation(df, retrain_frequency=100)

# Create monitoring dashboard
dashboard = system.create_monitoring_dashboard(backtest_results)

# Live trading (requires IB connection)
ib_system = CompleteProductionMLSystem(broker_api=ib_connection)
trade_signal = {...}  # Your trade signal
ib_system.execute_live_trade(trade_signal)
```

## Model Architecture

### Feature Engineering
The system creates 50+ features including:
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Volatility Features**: Rolling volatility, GARCH-style features
- **Microstructure Features**: Volume-price relationships, tick analysis
- **Regime Features**: Regime-specific indicators and thresholds
- **Time Features**: Hour, day-of-week, market session indicators

### ML Models Used
- **Gradient Boosting**: Primary models for stop-loss and entry confidence
- **XGBoost/LightGBM**: Alternative ensemble methods
- **Hidden Markov Models**: For regime detection (when available)
- **K-Means Clustering**: Fallback regime detection

### Hyperparameter Optimization
- Grid search with time-series cross-validation
- Regime-specific parameter tuning
- Walk-forward validation for model selection

## Performance Metrics

### Target Metrics (from requirements)
- **Sharpe Ratio**: > 1.5 (target), > 1.0 (minimum)
- **Calmar Ratio**: > 2.0
- **Profit Factor**: > 1.7
- **Win Rate**: > 55% (target), > 50% (minimum)
- **Max Drawdown**: < 10% (target), < 15% (acceptable)

### Additional Metrics
- Maximum Adverse Excursion (MAE)
- Maximum Favorable Excursion (MFE)
- Regime-specific performance analysis
- Risk-adjusted returns

## Production Readiness Assessment

The system includes automated production readiness checks:
1. **Performance Metrics**: Sharpe ratio and drawdown thresholds
2. **Win Rate**: Minimum acceptable win rate
3. **Walk-Forward Validation**: Out-of-sample performance validation
4. **Real-Time Integration**: Live trading capabilities
5. **Data Handling**: Robust data processing and outlier handling
6. **System Reliability**: Error handling and monitoring

## Risk Management

### Dynamic Risk Controls
- **Volatility-Adjusted Position Sizing**: Based on predicted volatility
- **Regime-Aware Stop Losses**: Adaptive stop-loss distances
- **Drawdown Protection**: Automatic position reduction during drawdowns
- **Real-Time Monitoring**: Live risk metrics tracking

### Stress Testing
The system includes comprehensive stress testing:
- High volatility scenarios
- Extended drawdown periods
- Regime transition analysis
- Model degradation detection

## Monitoring and Maintenance

### Real-Time Dashboard
- Live P&L tracking
- Performance metrics visualization
- Regime detection status
- Model confidence indicators

### Automated Retraining
- Configurable retraining intervals
- Performance degradation detection
- Model versioning and rollback

## File Structure

```
Safe code/
├── main.py     # Main system implementation
├── requirements.txt          # Python dependencies
├── instructions.txt          # Original problem specification
├── README.md                 # This file
├── CandleIndicators/         # Market data directory
└── Downloads_SP500/          # Position data directory
```

## Configuration

### Key Parameters
```python
# Risk management
initial_capital = 100000
risk_budget = 0.02  # 2% risk per trade

# Model parameters
retrain_frequency = 100  # Retrain every 100 periods
initial_window = 1000    # Initial training window

# Regime thresholds
regime_thresholds = {
    'low': 0.15,
    'medium': 0.2, 
    'high': 0.25
}
```

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install optional packages for full functionality
2. **Data Format Issues**: Ensure CSV files have proper timestamp columns
3. **Insufficient Data**: Minimum 1000+ data points recommended for training
4. **Memory Issues**: Large datasets may require chunked processing

### Performance Optimization
- Use smaller retraining frequencies for faster execution
- Enable multiprocessing for hyperparameter tuning
- Consider data sampling for very large datasets

## Contributing

When contributing to this project:
1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new functions
3. Include unit tests for new features
4. Update this README for significant changes

## License

This project is for educational and research purposes. Please ensure compliance with relevant financial regulations before using in production trading.

## Disclaimer

This software is provided for educational purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always conduct thorough testing before deploying any trading system with real capital.

## Support

For technical issues or questions about the implementation, please refer to the code comments and docstrings within `main.py`.
>>>>>>> 7a4082f (Add ML trading system files: algorithm.py, README.md, requirements.txt)

