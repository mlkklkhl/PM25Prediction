import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
import optuna
from optuna.samplers import TPESampler
import warnings
import json
import seaborn as sns

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


# ============================================================================
# IMPROVED LSTM MODELS
# ============================================================================

class ImprovedLSTMModel(nn.Module):
    """Improved LSTM with residual connection"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Add residual connection for PM2.5 lag1
        self.use_residual = True

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = h_n[-1]
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)

        # Add residual: predict change from lag1 instead of absolute value
        if self.use_residual and x.shape[-1] > 0:
            # Assume first feature is PM2.5_lag1
            lag1_value = x[:, -1, 0:1]  # Last timestep, first feature
            out = out + lag1_value

        return out


# ============================================================================
# FEATURE ENGINEERING - FIXED
# ============================================================================

def create_enhanced_features(df, target):
    """Create features without future leakage"""
    df = df.copy()

    # Lagged values
    for lag in [1, 2, 3, 7, 14]:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)

    # Rolling statistics (SHIFT FIRST!)
    for window in [3, 7, 14]:
        df[f'{target}_rolling_mean_{window}'] = df[target].shift(1).rolling(window, min_periods=1).mean()
        df[f'{target}_rolling_std_{window}'] = df[target].shift(1).rolling(window, min_periods=1).std()
        df[f'{target}_rolling_min_{window}'] = df[target].shift(1).rolling(window, min_periods=1).min()
        df[f'{target}_rolling_max_{window}'] = df[target].shift(1).rolling(window, min_periods=1).max()

    # Difference features (trend)
    df[f'{target}_diff_1'] = df[target].diff(1)
    df[f'{target}_diff_7'] = df[target].diff(7)

    # Weather interactions
    for var in ['temperature', 'humidity', 'wind_speed', 'air_pressure']:
        if var in df.columns:
            df[f'{var}_lag1'] = df[var].shift(1)
            df[f'{var}_rolling_mean_3'] = df[var].shift(1).rolling(3, min_periods=1).mean()

    # PM10 ratio (if available)
    if 'PM10' in df.columns:
        df['PM2.5_PM10_ratio'] = df[target] / (df['PM10'].shift(1) + 1e-6)

    return df


def create_sequences(df, features, target, sequence_length, prediction_days):
    """Create sequences for prediction"""
    sequences = []
    targets = []
    sequence_dates = []

    for i in range(len(df) - sequence_length - prediction_days + 1):
        seq_data = df.iloc[i:i + sequence_length]
        target_data = df.iloc[i + sequence_length:i + sequence_length + prediction_days]

        # Skip if missing values
        if seq_data[features].isnull().any().any() or target_data[target].isnull().any():
            continue

        seq = seq_data[features].values
        target_vals = target_data[target].values

        sequences.append(seq.astype(np.float32))
        targets.append(target_vals.astype(np.float32))
        sequence_dates.append(seq_data.index[-1])

    return np.array(sequences), np.array(targets), sequence_dates


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs=100, patience=15, verbose=False):
    """Training with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_seq, batch_tgt in train_loader:
            batch_seq, batch_tgt = batch_seq.to(device), batch_tgt.to(device)

            optimizer.zero_grad()
            outputs = model(batch_seq)
            loss = criterion(outputs, batch_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_seq, batch_tgt in val_loader:
                batch_seq, batch_tgt = batch_seq.to(device), batch_tgt.to(device)
                outputs = model(batch_seq)
                loss = criterion(outputs, batch_tgt)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss, train_losses, val_losses


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_multistep(y_true, y_pred, prediction_days):
    """Evaluate predictions"""
    results = {}

    for day in range(prediction_days):
        true_day = y_true[:, day]
        pred_day = y_pred[:, day]

        mae = mean_absolute_error(true_day, pred_day)
        rmse = root_mean_squared_error(true_day, pred_day)
        r2 = r2_score(true_day, pred_day)

        results[f'day_{day + 1}'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

    return results


# ============================================================================
# OPTUNA OPTIMIZATION - FIXED WITH WALK-FORWARD VALIDATION
# ============================================================================

def objective(trial, df_clean, features, target, prediction_days):
    """
    Fixed objective using SINGLE walk-forward split
    This prevents overfitting to CV folds
    """
    # Hyperparameters
    sequence_length = trial.suggest_categorical('sequence_length', [7, 14, 21, 28])
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # Create sequences
    X, y, _ = create_sequences(df_clean, features, target, sequence_length, prediction_days)

    if len(X) < 300:
        return -1.0

    # FIXED: Use single 70-15-15 split (walk-forward)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    # CRITICAL FIX: Use RobustScaler and fit on ENTIRE training set
    # (not per-year, which causes distribution shift)
    feature_scaler = RobustScaler()  # More robust to outliers
    target_scaler = RobustScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)

    # Create dataloaders
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = ImprovedLSTMModel(
        input_size=X_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=prediction_days,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Train
    model, val_loss, _, _ = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        epochs=50, patience=10, verbose=False
    )

    # Predict
    model.eval()
    val_preds_scaled = []
    with torch.no_grad():
        for batch_seq, _ in val_loader:
            batch_seq = batch_seq.to(device)
            outputs = model(batch_seq)
            val_preds_scaled.extend(outputs.cpu().numpy())

    # Inverse transform
    val_preds = target_scaler.inverse_transform(np.array(val_preds_scaled))

    # Calculate R² for day 1
    r2_day1 = r2_score(y_val[:, 0], val_preds[:, 0])

    trial.set_user_attr("r2", r2_day1)
    trial.set_user_attr("val_loss", val_loss)

    return r2_day1


def optimize_hyperparameters(df_clean, features, target, prediction_days, n_trials=50):
    """Run Optuna optimization"""
    print(f"\n{'=' * 70}")
    print(f"HYPERPARAMETER OPTIMIZATION ({prediction_days}-DAY PREDICTION)")
    print("=" * 70)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f'lstm_pm25_{prediction_days}day'
    )

    print(f"Running {n_trials} trials with walk-forward validation...")
    study.optimize(
        lambda trial: objective(trial, df_clean, features, target, prediction_days),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nBest trial:")
    print(f"  R²: {study.best_value:.4f}")

    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


# ============================================================================
# FINAL TRAINING - FIXED
# ============================================================================

def train_with_best_params(df_clean, features, target, best_params, prediction_days):
    """
    FIXED: Use walk-forward validation instead of k-fold
    This better simulates real deployment
    """
    print(f"\n{'=' * 70}")
    print(f"FINAL EVALUATION ({prediction_days}-DAY)")
    print("=" * 70)

    sequence_length = best_params['sequence_length']
    X, y, sequence_dates = create_sequences(
        df_clean, features, target, sequence_length, prediction_days
    )

    print(f"Total sequences: {len(X)}")

    # FIXED: 70-15-15 split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Get date ranges
    train_dates = sequence_dates[:train_size]
    val_dates = sequence_dates[train_size:train_size + val_size]
    test_dates = sequence_dates[train_size + val_size:]

    print(f"\nTrain: {train_dates[0]} to {train_dates[-1]}")
    print(f"Val:   {val_dates[0]} to {val_dates[-1]}")
    print(f"Test:  {test_dates[0]} to {test_dates[-1]}")

    # CRITICAL: Single scaler for entire dataset
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)
    X_test_scaled = feature_scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    # Check for distribution shift
    print(f"\nDistribution check:")
    print(f"  Train target mean: {y_train.mean():.2f} ± {y_train.std():.2f}")
    print(f"  Val target mean:   {y_val.mean():.2f} ± {y_val.std():.2f}")
    print(f"  Test target mean:  {y_test.mean():.2f} ± {y_test.std():.2f}")

    # Create dataloaders
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    # Create model
    model = ImprovedLSTMModel(
        input_size=X_train.shape[-1],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        output_size=prediction_days,
        dropout_rate=best_params['dropout_rate']
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    criterion = nn.MSELoss()

    # Train
    model, _, train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        epochs=100, patience=15, verbose=True
    )

    # Predict on test set
    model.eval()
    test_preds_scaled = []
    with torch.no_grad():
        for batch_seq, _ in test_loader:
            batch_seq = batch_seq.to(device)
            outputs = model(batch_seq)
            test_preds_scaled.extend(outputs.cpu().numpy())

    # Inverse transform
    test_preds = target_scaler.inverse_transform(np.array(test_preds_scaled))

    # Evaluate
    results = evaluate_multistep(y_test, test_preds, prediction_days)

    print(f"\nTest Results:")
    for day_key, metrics in results.items():
        print(f"  {day_key}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")

    return results, test_preds, y_test, train_losses, val_losses


# ============================================================================
# MAIN FUNCTION - FIXED
# ============================================================================

def main(file_path, use_optuna=True, n_trials=50):
    """Main pipeline with all fixes"""
    print("=" * 70)
    print("FIXED PM2.5 FORECASTING")
    print("=" * 70)

    # Load data
    df = pd.read_csv(file_path)
    df.set_index('DateTime', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
    df = df.resample('D').mean()

    # Remove 2024 (insufficient data)
    if 2024 in df.index.year.unique():
        print("\nRemoving 2024 data (insufficient samples)")
        df = df[df.index.year < 2024]

    # CRITICAL FIX: Don't normalize by year!
    # Use raw PM2.5 values - let RobustScaler handle it globally
    target = 'PM2.5'

    # Add temporal features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['season'] = (df['month'] % 12 + 3) // 3  # 1=winter, 2=spring, etc.

    # Create enhanced features
    df = create_enhanced_features(df, target)

    # Select most important features
    features = [
        'PM2.5_lag1',
        'PM2.5_lag2',
        'PM2.5_lag7',
        'PM2.5_rolling_mean_7',
        'PM2.5_rolling_std_7',
        'PM2.5_diff_1',
        'PM10',
        'temperature',
        'humidity',
        'wind_speed',
        'air_pressure',
        'month',
        'is_weekend'
    ]

    # Fill gaps
    for col in features + [target]:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill', limit=3)

    df_clean = df.dropna(subset=features + [target])

    print(f"\nClean data: {len(df_clean)} samples")
    print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"PM2.5 stats: mean={df_clean[target].mean():.2f}, std={df_clean[target].std():.2f}")

    # # Train for 1-day prediction
    # prediction_days = 1

    # Train models for different prediction horizons
    prediction_horizons = [1, 2, 7]

    all_models = {}
    best_params_all = {}

    for prediction_days in prediction_horizons:

        print(f"\n{'=' * 70}")
        print(f"PROCESSING {prediction_days}-DAY PREDICTION")
        print("=" * 70)

        if use_optuna:
            study = optimize_hyperparameters(df_clean, features, target, prediction_days, n_trials)
            best_params = study.best_params
        else:
            best_params = {
                'sequence_length': 14,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': 0.2,
                'batch_size': 64,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            }

        # Final training
        results, preds, actuals, train_losses, val_losses = train_with_best_params(
            df_clean, features, target, best_params, prediction_days
        )

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Predictions scatter
        ax1.scatter(actuals[:, 0], preds[:, 0], alpha=0.5)
        ax1.plot([actuals.min(), actuals.max()],
                 [actuals.min(), actuals.max()], 'r--', lw=2)
        r2 = results['day_1']['R2']
        mae = results['day_1']['MAE']
        ax1.set_xlabel('Actual PM2.5')
        ax1.set_ylabel('Predicted PM2.5')
        ax1.set_title(f'{prediction_days}-Day Forecast\nR²={r2:.3f}, MAE={mae:.2f}')
        ax1.grid(True, alpha=0.3)

        # Training curves
        ax2.plot(train_losses, label='Train Loss')
        ax2.plot(val_losses, label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'pm25_fixed_results_{prediction_days}-Day Forecast.png', dpi=300)
        print(f"\nSaved: pm25_fixed_results{prediction_days}-Day Forecast.png")


if __name__ == "__main__":
    main(
        "combined_data_1d_preprocessed.csv",
        use_optuna=True,
        n_trials=100
    )
