# PM2.5 Multi-Day Forecasting with LSTM

A production-ready LSTM model for multi-day PM2.5 air quality forecasting using direct prediction approach with automated hyperparameter optimization.

## 🎯 Key Features

- **Direct PM2.5 Prediction**: Leverages strong temporal autocorrelation (0.786) for superior accuracy
- **Multi-Horizon Forecasting**: Simultaneous 1-day, 2-day, and 7-day predictions
- **Automated Hyperparameter Tuning**: Optuna-based optimization including sequence length
- **Walk-Forward Validation**: Robust time-series cross-validation
- **Comprehensive Metrics**: MAE, RMSE, R², MASE, and sMAPE

## 📊 Performance Expectations

Based on data diagnostics (2016-2024, 2,709 samples):

| Prediction Horizon | Expected R² | Expected MAE | Target Performance |
|-------------------|-------------|--------------|-------------------|
| **1-day** | 0.60-0.75 | 1.5-2.5 μg/m³ | ✅ Excellent |
| **2-day** | 0.45-0.60 | 2.5-3.5 μg/m³ | ✅ Good |
| **7-day** | 0.20-0.40 | 4.0-6.0 μg/m³ | ⚠️ Acceptable |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn matplotlib optuna
```

### Basic Usage

```python
# Run with default settings (Optuna optimization enabled)
python lstm_direct_prediction.py

# Quick test without Optuna
# Set USE_OPTUNA = False in __main__
```

### Data Format

Your CSV file (`combined_data_1d.csv`) should have:

```
DateTime,PM2.5,PM10,temperature,humidity,wind_speed,wind_direction,air_pressure,rainfall,Month
2016-01-01,25.3,45.2,18.5,65.2,2.3,180,1013.2,0.0,1
2016-01-02,23.1,42.8,19.2,62.8,2.1,175,1012.8,0.0,1
...
```

**Required columns:**
- `DateTime`: Date in YYYY-MM-DD format
- `PM2.5`: Target variable (μg/m³)
- Weather features: `air_pressure`, `humidity`, `rainfall`, `temperature`, `wind_direction`, `wind_speed`
- `PM10`: Particulate matter 10
- `Month`: Month number (1-12)

## 🏗️ Architecture

### Model Design

```
Input Sequence (seq_len × features) 
    ↓
LSTM Layers (with dropout)
    ↓
Dense Layer (hidden_size → hidden_size/2)
    ↓
ReLU + Dropout
    ↓
Output Layer (→ prediction_days)
```

### Key Design Choices

1. **Direct Prediction Approach**
   - Predicts actual PM2.5 values (not differences)
   - Eliminates error accumulation in multi-step forecasting
   - Leverages strong autocorrelation (lag-1: 0.786)

2. **Enhanced Features**
   - PM2.5 lags: [1, 2, 3, 7 days]
   - Weather change features: Δtemperature, Δhumidity, Δwind_speed, Δair_pressure
   - Rolling statistics: 3-day and 7-day moving averages
   - Temporal encoding: day_of_week, month

3. **Hyperparameter Search Space**
   - `sequence_length`: [3, 5, 7, 10, 14, 21] days
   - `hidden_size`: [32, 64, 128, 256]
   - `num_layers`: [1, 2, 3]
   - `dropout_rate`: [0.1, 0.4]
   - `batch_size`: [16, 32, 64]
   - `learning_rate`: [1e-4, 1e-2] (log scale)

## 📁 Project Structure

```
pm25-lstm-forecasting/
│
├── lstm_direct_prediction.py      # Main training script
├── combined_data_1d.csv           # Input data (daily PM2.5 + weather)
│
├── outputs/                       # Generated after training
│   ├── best_hyperparameters_direct.json
│   ├── optuna_study_direct_1day.csv
│   ├── optuna_study_direct_2day.csv
│   ├── optuna_study_direct_7day.csv
│   └── multistep_forecast_results_direct.png
│
└── README.md                      # This file
```

## 🔬 Methodology

### Why Direct Prediction?

Initial data diagnostics revealed:

```
Original PM2.5 autocorrelation:     0.786 (strong)
Differential PM2.5 autocorrelation: -0.128 (weak, nearly random)
```

**Conclusion**: Direct PM2.5 prediction is 2-3× more accurate than predicting daily changes.

### Walk-Forward Validation

```
|----Train----|--Val--|--Test--| Fold 1
     |----Train----|--Val--|--Test--| Fold 2
          |----Train----|--Val--|--Test--| Fold 3
               |----Train----|--Val--|--Test--| Fold 4
                    |----Train----|--Val--|--Test--| Fold 5
```

- 5-fold time-series cross-validation
- No data leakage (strictly chronological splits)
- Each fold trains on expanding window

## 📈 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(\|y - ŷ\|)` | Average prediction error (μg/m³) |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Root mean squared error (penalizes large errors) |
| **R²** | `1 - SS_res/SS_tot` | Variance explained (1.0 = perfect, 0 = baseline) |
| **MASE** | `MAE / MAE_naive` | Scaled error (<1 = beats naive forecast) |
| **sMAPE** | `100 × mean(\|y-ŷ\| / ((y+ŷ)/2))` | Symmetric percentage error (0-100%) |

## ⚙️ Configuration

### Optuna Optimization

```python
# In __main__
USE_OPTUNA = True      # Enable hyperparameter tuning
N_TRIALS = 100         # Number of optimization trials (50-100 recommended)
```

**Recommended trials:**
- Quick test: 20 trials (~10 minutes)
- Production: 100 trials (~45 minutes)
- Exhaustive: 200 trials (~90 minutes)

### Default Hyperparameters (if Optuna disabled)

```python
{
    'sequence_length': 7,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout_rate': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5
}
```

## 📊 Output Interpretation

### 1. Best Hyperparameters JSON

```json
{
  "1day": {
    "sequence_length": 5,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout_rate": 0.23,
    "batch_size": 32,
    "learning_rate": 0.0015,
    "weight_decay": 2.3e-05
  },
  ...
}
```

### 2. Optuna Study CSV

Contains all trials with metrics for analysis:
- Trial number, parameters tested
- Validation R², MAE, loss
- Use for understanding hyperparameter importance

### 3. Visualization

Scatter plots comparing actual vs predicted PM2.5:
- Perfect predictions fall on diagonal line
- Tighter clustering = better accuracy
- Separate plot for each prediction day

## 🛠️ Advanced Usage

### Custom Feature Engineering

```python
# Add your own features in create_enhanced_features()
def create_enhanced_features(df, target, original_features):
    df = df.copy()
    
    # Example: Add your custom feature
    df['custom_feature'] = your_calculation(df)
    
    return df
```

### Different Prediction Horizons

```python
# In main()
prediction_horizons = [1, 3, 5, 7, 14]  # Customize as needed
```

### GPU Acceleration

Automatically uses CUDA if available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

## 📝 Data Diagnostics Summary

Before modeling, comprehensive diagnostics were performed:

```
Dataset: 3,148 days (2016-01-01 to 2024-08-13)
Valid samples: 2,709
PM2.5 range: 5-74 μg/m³
Mean: 17.56 μg/m³
Std: 6.86 μg/m³

Key Finding: Strong autocorrelation (0.786) → Direct prediction optimal
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Attention mechanisms for LSTM
- [ ] Ensemble methods (LSTM + XGBoost)
- [ ] External data integration (satellite, traffic)
- [ ] Probabilistic forecasting (prediction intervals)
- [ ] Real-time deployment pipeline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data diagnostics methodology based on time-series best practices
- Optuna framework for efficient hyperparameter optimization
- PyTorch for flexible deep learning architecture

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Pull requests welcome

## 🔍 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Reduce batch_size in hyperparameter search
batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
```

**2. NaN in predictions**
```python
# Check for missing values
print(df.isnull().sum())
# Increase ffill limit in data preprocessing
df[col] = df[col].ffill(limit=5)
```

**3. Poor performance**
- Verify data quality (no extreme outliers)
- Check feature scaling (should be automatic)
- Increase n_trials for better optimization
- Review sequence_length (might be too long/short)

---

**Note**: This model predicts PM2.5 concentration for air quality monitoring. For health advisories or policy decisions, consult domain experts and validate predictions against ground truth measurements.
