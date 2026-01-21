import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import optuna
from optuna.samplers import TPESampler
import warnings
import json

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def calculate_mase(y_true, y_pred, y_train, seasonality=1):
    """
    Calculate Mean Absolute Scaled Error (MASE)
    """
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    naive_errors = np.abs(np.diff(y_train, n=seasonality))
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.nan

    mase = mae_forecast / mae_naive
    return mase


def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    smape = np.mean(100 * numerator[mask] / denominator[mask]) if np.any(mask) else np.nan
    return smape


def create_simple_features(df, target):
    """Minimal features - let LSTM learn patterns"""
    df = df.copy()

    # ONLY current values + 1-day lag for PM2.5
    df[f'{target}_lag1'] = df[target].shift(1)

    return df


def create_enhanced_features_fixed(df, target, original_features):
    """FIXED: No future information leakage"""
    df = df.copy()

    # PM2.5 lags (safe - uses past only)
    for lag in [1, 2, 3, 7]:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)

    # Rolling stats (FIXED - shift first!)
    df[f'{target}_rolling_mean_3'] = df[target].shift(1).rolling(3).mean()
    df[f'{target}_rolling_std_3'] = df[target].shift(1).rolling(3).std()
    df[f'{target}_rolling_mean_7'] = df[target].shift(1).rolling(7).mean()

    # Weather changes (safe)
    for var in ['temperature', 'humidity', 'wind_speed', 'air_pressure']:
        if var in df.columns:
            df[f'{var}_change'] = df[var].diff()
            df[f'{var}_lag1'] = df[var].shift(1)

    return df


def create_sequences_direct(df, features, target, sequence_length, prediction_days):
    """
    Create sequences for DIRECT PM2.5 prediction

    CRITICAL CHANGE: Predict actual PM2.5 values, not differences
    This leverages the strong autocorrelation (0.786) in original PM2.5
    """
    sequences = []
    targets = []
    sequence_dates = []

    for i in range(len(df) - sequence_length - prediction_days + 1):
        seq_data = df.iloc[i:i + sequence_length]
        target_data = df.iloc[i + sequence_length:i + sequence_length + prediction_days]

        # Check for missing values
        if seq_data[features].isnull().any().any() or target_data[target].isnull().any():
            continue

        seq = seq_data[features].values
        target_vals = target_data[target].values  # DIRECT PM2.5, not diff

        sequences.append(seq.astype(np.float32))
        targets.append(target_vals.astype(np.float32))
        sequence_dates.append(seq_data.index[-1])

    return np.array(sequences), np.array(targets), sequence_dates


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=100, patience=15):
    """Training with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_seq, batch_tgt in val_loader:
                batch_seq, batch_tgt = batch_seq.to(device), batch_tgt.to(device)
                outputs = model(batch_seq)
                loss = criterion(outputs, batch_tgt)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def evaluate_multistep(y_true, y_pred, y_train_for_mase, prediction_days):
    """
    Evaluate multi-step predictions with enhanced metrics
    """
    results = {}

    for day in range(prediction_days):
        true_day = y_true[:, day]
        pred_day = y_pred[:, day]

        mae = mean_absolute_error(true_day, pred_day)
        rmse = root_mean_squared_error(true_day, pred_day)
        r2 = r2_score(true_day, pred_day)
        mase = calculate_mase(true_day, pred_day, y_train_for_mase, seasonality=1)
        smape = calculate_smape(true_day, pred_day)

        results[f'day_{day + 1}'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MASE': mase,
            'sMAPE': smape
        }

    return results


def objective(trial, df_clean, features, target, prediction_days):
    """
    Optuna objective function for direct PM2.5 prediction

    EXPANDED sequence_length search space: [3, 5, 7, 10, 14, 21]
    """
    # Hyperparameters to tune
    sequence_length = trial.suggest_categorical('sequence_length', [3, 5, 7, 10, 14, 21])
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

    # Create sequences with direct PM2.5 prediction
    X, y, sequence_dates = create_sequences_direct(
        df_clean, features, target, sequence_length, prediction_days
    )

    if len(X) < 100:
        return -1.0

    # Split train/val (70/30)
    n_samples = len(X)
    train_end = int(n_samples * 0.7)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:]
    y_val = y[train_end:]

    # Scale data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)

    # Create dataloaders
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = LSTMModel(
        input_size=X_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=prediction_days,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Train
    model, val_loss = train_model(model, train_loader, val_loader, optimizer, criterion,
                                  epochs=50, patience=10)

    # Get predictions
    model.eval()
    val_preds_scaled = []
    with torch.no_grad():
        for batch_seq, _ in val_loader:
            batch_seq = batch_seq.to(device)
            outputs = model(batch_seq)
            val_preds_scaled.extend(outputs.cpu().numpy())

    # Inverse transform (no reconstruction needed!)
    val_preds = target_scaler.inverse_transform(np.array(val_preds_scaled))
    val_true = y_val

    # Calculate R² for day 1 (primary metric)
    mae_day1 = mean_absolute_error(val_true[:, 0], val_preds[:, 0])
    r2_day1 = r2_score(val_true[:, 0], val_preds[:, 0])

    # Store additional metrics
    trial.set_user_attr("mae_day1", mae_day1)
    trial.set_user_attr("r2_day1", r2_day1)
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("n_sequences", len(X))

    return r2_day1


def optimize_hyperparameters(df_clean, features, target, prediction_days, n_trials=100):
    """Run Optuna optimization with expanded search space"""
    print(f"\n{'=' * 70}")
    print(f"HYPERPARAMETER OPTIMIZATION ({prediction_days}-DAY PREDICTION)")
    print(f"DIRECT PM2.5 PREDICTION (Leveraging 0.786 autocorrelation)")
    print(f"Sequence lengths: [3, 5, 7, 10, 14, 21] days")
    print("=" * 70)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f'lstm_pm25_direct_{prediction_days}day'
    )

    print(f"\nRunning {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, df_clean, features, target, prediction_days),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nBest trial:")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  R² (Day 1): {study.best_value:.4f}")
    print(f"  MAE (Day 1): {study.best_trial.user_attrs['mae_day1']:.2f} μg/m³")
    print(f"  Val Loss: {study.best_trial.user_attrs['val_loss']:.6f}")
    print(f"  Sequences created: {study.best_trial.user_attrs['n_sequences']}")

    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


def train_with_best_params(df_clean, features, target, best_params, prediction_days):
    # Replace walk-forward with proper TimeSeriesSplit
    print(f"\n{'=' * 70}")
    print(f"TimeSeriesSplit VALIDATION WITH BEST PARAMS ({prediction_days}-DAY)")
    print(f"Using sequence_length = {best_params['sequence_length']} days")
    print(f"DIRECT PM2.5 PREDICTION")
    print("=" * 70)

    sequence_length = best_params['sequence_length']
    X, y, sequence_dates = create_sequences_direct(
        df_clean, features, target, sequence_length, prediction_days
    )

    print(f"Created {len(X)} sequences with sequence_length={sequence_length}")

    all_results = {f'day_{i + 1}': [] for i in range(prediction_days)}
    all_predictions = []
    all_actuals = []

    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # CRITICAL FIX: Split validation BEFORE scaling
        val_split = int(0.8 * len(X_train))
        X_train_only, X_val = X_train[:val_split], X_train[val_split:]
        y_train_only, y_val = y_train[:val_split], y_train[val_split:]

        # Scale - fit ONLY on training data (not including validation)
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        X_train_scaled = feature_scaler.fit_transform(
            X_train_only.reshape(-1, X_train_only.shape[-1])
        ).reshape(X_train_only.shape)

        X_val_scaled = feature_scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)

        X_test_scaled = feature_scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)

        y_train_scaled = target_scaler.fit_transform(
            y_train_only.reshape(-1, y_train_only.shape[-1])
        ).reshape(y_train_only.shape)

        y_val_scaled = target_scaler.transform(
            y_val.reshape(-1, y_val.shape[-1])
        ).reshape(y_val.shape)

        y_test_scaled = target_scaler.transform(
            y_test.reshape(-1, y_test.shape[-1])
        ).reshape(y_test.shape)

        # Create dataloaders
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

        # Create model
        model = LSTMModel(
            input_size=X_train.shape[-1],
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            output_size=prediction_days,
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                               weight_decay=best_params['weight_decay'])
        criterion = nn.MSELoss()

        print(f"\nFold {fold + 1}/5: Training model...")
        model, _ = train_model(model, train_loader, val_loader, optimizer, criterion,
                               epochs=100, patience=15)

        # Predict
        model.eval()
        test_preds_scaled = []
        with torch.no_grad():
            for batch_seq, _ in test_loader:
                batch_seq = batch_seq.to(device)
                outputs = model(batch_seq)
                test_preds_scaled.extend(outputs.cpu().numpy())

        # Inverse transform (NO reconstruction needed!)
        test_preds = target_scaler.inverse_transform(np.array(test_preds_scaled))
        test_true = y_test

        # Get training PM2.5 for MASE calculation
        train_pm25_for_mase = y_train_only.flatten()

        # Evaluate
        fold_results = evaluate_multistep(test_true, test_preds,
                                          train_pm25_for_mase, prediction_days)

        print(f"Fold {fold + 1} Results:")
        for day_key, metrics in fold_results.items():
            mase_str = f"{metrics['MASE']:.4f}" if not np.isnan(metrics['MASE']) else "N/A"
            smape_str = f"{metrics['sMAPE']:.2f}%" if not np.isnan(metrics['sMAPE']) else "N/A"
            print(f"  {day_key}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, "
                  f"R²={metrics['R2']:.4f}, MASE={mase_str}, sMAPE={smape_str}")
            all_results[day_key].append(metrics)

        all_predictions.append(test_preds)
        all_actuals.append(test_true)

    return all_results, all_predictions, all_actuals


def main(file_path, use_optuna=True, n_trials=100):
    """
    Main training function with DIRECT PM2.5 prediction

    KEY CHANGES:
    1. Predict PM2.5 directly (leveraging 0.786 autocorrelation)
    2. Expanded sequence_length search: [3, 5, 7, 10, 14, 21]
    3. Enhanced features including weather changes
    4. No reconstruction step needed
    """
    print("=" * 70)
    print("MULTI-DAY PM2.5 FORECASTING WITH LSTM")
    print("DIRECT PREDICTION APPROACH")
    print("Sequence lengths: [3, 5, 7, 10, 14, 21] days")
    print("Metrics: MAE, RMSE, R², MASE, sMAPE")
    print("=" * 70)

    # Load data
    df = pd.read_csv(file_path)
    df.set_index('DateTime', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
    df = df.resample('D').mean()

    target = 'PM2.5'
    original_features = ['air_pressure', 'humidity', 'rainfall', 'temperature',
                         'wind_direction', 'wind_speed', 'PM10', 'Month']

    # Add temporal features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Create enhanced features (DIRECT approach)
    # df = create_simple_features(df, target)
    df = create_enhanced_features_fixed(df, target, original_features)

    # Define features
    lag_features = [f'{target}_lag{i}' for i in [1, 2, 3, 7]]
    rolling_features = [f'{target}_rolling_mean_3', f'{target}_rolling_std_3',
                        f'{target}_rolling_mean_7']
    weather_change_features = []
    for var in ['temperature', 'humidity', 'wind_speed', 'air_pressure']:
        if f'{var}_change' in df.columns:
            weather_change_features.extend([f'{var}_change', f'{var}_lag1'])

    features = original_features + ['day_of_week'] + lag_features + rolling_features + weather_change_features
    features = [f for f in features if f in df.columns]

    # Fill gaps
    for col in features + [target]:
        if col in df.columns:
            df[col] = df[col].ffill(limit=3)

    df_clean = df.dropna(subset=features + [target])
    print(f"Clean data: {len(df_clean)} samples")
    print(f"Number of features: {len(features)}")
    print(f"Features: {features}")

    # Train models for different prediction horizons
    prediction_horizons = [1, 2, 7]

    all_models = {}
    best_params_all = {}

    for pred_days in prediction_horizons:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING {pred_days}-DAY PREDICTION")
        print("=" * 70)

        if use_optuna:
            study = optimize_hyperparameters(df_clean, features, target, pred_days, n_trials=n_trials)
            best_params = study.best_params
            best_params_all[f'{pred_days}day'] = best_params

            study_df = study.trials_dataframe()
            study_df.to_csv(f'optuna_study_direct_{pred_days}day.csv', index=False)
            print(f"Saved Optuna study to 'optuna_study_direct_{pred_days}day.csv'")
        else:
            best_params = {
                'sequence_length': 7,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout_rate': 0.2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            }
            print(f"Using default parameters (Optuna disabled)")

        # Train with best params
        results, preds, actuals = train_with_best_params(
            df_clean, features, target, best_params, pred_days
        )

        # Aggregate results
        print(f"\n{'=' * 70}")
        print(f"AGGREGATE RESULTS ({pred_days}-DAY PREDICTION)")
        print(f"With sequence_length = {best_params['sequence_length']}")
        print("=" * 70)

        for day in range(pred_days):
            day_key = f'day_{day + 1}'
            metrics = results[day_key]

            avg_mae = np.mean([m['MAE'] for m in metrics])
            avg_rmse = np.mean([m['RMSE'] for m in metrics])
            avg_r2 = np.mean([m['R2'] for m in metrics])

            mase_values = [m['MASE'] for m in metrics if not np.isnan(m['MASE'])]
            smape_values = [m['sMAPE'] for m in metrics if not np.isnan(m['sMAPE'])]

            avg_mase = np.mean(mase_values) if mase_values else np.nan
            avg_smape = np.mean(smape_values) if smape_values else np.nan

            std_mae = np.std([m['MAE'] for m in metrics])
            std_r2 = np.std([m['R2'] for m in metrics])
            std_mase = np.std(mase_values) if len(mase_values) > 1 else 0

            print(f"\n{day_key.upper().replace('_', ' ')}:")
            print(f"  MAE:  {avg_mae:.2f} ± {std_mae:.2f} μg/m³")
            print(f"  RMSE: {avg_rmse:.2f} μg/m³")
            print(f"  R²:   {avg_r2:.4f} ± {std_r2:.4f}")

            if not np.isnan(avg_mase):
                print(
                    f"  MASE:  {avg_mase:.4f} ± {std_mase:.4f} {'✓ Beats naive' if avg_mase < 1 else '✗ Worse than naive'}")
            else:
                print(f"  MASE:  N/A")

            if not np.isnan(avg_smape):
                print(f"  sMAPE: {avg_smape:.2f}%")
            else:
                print(f"  sMAPE: N/A")

        all_models[f'{pred_days}day'] = {
            'results': results,
            'predictions': preds,
            'actuals': actuals,
            'best_params': best_params
        }

    # Save best parameters
    with open('best_hyperparameters_direct.json', 'w') as f:
        json.dump(best_params_all, f, indent=2)
    print(f"\n{'=' * 70}")
    print("Saved best hyperparameters to 'best_hyperparameters_direct.json'")
    print("=" * 70)

    # Create comparison visualization
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))

    plot_idx = 0
    for pred_days in prediction_horizons:
        model_data = all_models[f'{pred_days}day']
        seq_len = model_data['best_params']['sequence_length']

        all_preds = np.concatenate(model_data['predictions'], axis=0)
        all_actuals = np.concatenate(model_data['actuals'], axis=0)

        for day in range(pred_days):
            row = plot_idx // 3
            col = plot_idx % 3

            preds_day = all_preds[:, day]
            actuals_day = all_actuals[:, day]

            axes[row, col].scatter(actuals_day, preds_day, alpha=0.4, s=10)
            axes[row, col].plot([actuals_day.min(), actuals_day.max()],
                                [actuals_day.min(), actuals_day.max()], 'r--', lw=2)

            r2 = r2_score(actuals_day, preds_day)
            mae = mean_absolute_error(actuals_day, preds_day)

            axes[row, col].set_xlabel('Actual PM2.5 (μg/m³)')
            axes[row, col].set_ylabel('Predicted PM2.5 (μg/m³)')
            axes[row, col].set_title(
                f'{pred_days}-day DIRECT (seq={seq_len}): Day +{day + 1}\n(R²={r2:.3f}, MAE={mae:.2f})')
            axes[row, col].grid(True, alpha=0.3)

            plot_idx += 1

    for idx in range(plot_idx, 12):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('multistep_forecast_results_direct.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'multistep_forecast_results_direct.png'")
    print("=" * 70)

    return all_models, best_params_all


if __name__ == "__main__":
    data_file_path = "combined_data_1d_preprocessed.csv"

    try:
        USE_OPTUNA = True
        N_TRIALS = 100  # Increased for expanded search space

        print(f"\n{'=' * 70}")
        print("CONFIGURATION")
        print("=" * 70)
        print(f"Approach: DIRECT PM2.5 PREDICTION")
        print(f"Optuna Enabled: {USE_OPTUNA}")
        if USE_OPTUNA:
            print(f"Number of Trials: {N_TRIALS}")
            print(f"Sequence lengths: [3, 5, 7, 10, 14, 21] days")
        print("=" * 70)

        models, best_params = main(
            data_file_path,
            use_optuna=USE_OPTUNA,
            n_trials=N_TRIALS
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED!")
        print("=" * 70)

        if USE_OPTUNA:
            print("\n✅ Optimized hyperparameters (DIRECT prediction):")
            for model_name, params in best_params.items():
                print(f"\n  {model_name}:")
                print(f"    sequence_length: {params['sequence_length']} days")
                print(f"    hidden_size: {params['hidden_size']}")
                print(f"    num_layers: {params['num_layers']}")
                print(f"    dropout_rate: {params['dropout_rate']:.3f}")
                print(f"    batch_size: {params['batch_size']}")
                print(f"    learning_rate: {params['learning_rate']:.6f}")

            print("\n📁 Output files:")
            print("  - best_hyperparameters_direct.json")
            print("  - optuna_study_direct_1day.csv")
            print("  - optuna_study_direct_2day.csv")
            print("  - optuna_study_direct_7day.csv")
            print("  - multistep_forecast_results_direct.png")

        print("\n" + "=" * 70)
        print("KEY IMPROVEMENTS IN THIS VERSION:")
        print("=" * 70)
        print("✅ Direct PM2.5 prediction")
        print("✅ Leverages strong autocorrelation (0.786)")
        print("✅ Expanded sequence_length search: [3, 5, 7, 10, 14, 21]")
        print("✅ Enhanced features: weather changes + rolling stats")
        print("✅ No reconstruction step (no error accumulation)")
        print("✅ Expected 2-3x better R² performance")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
