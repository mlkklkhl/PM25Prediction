import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

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

# Load data
df = pd.read_csv('combined_data_1d_preprocessed.csv', index_col='DateTime', parse_dates=True)

# Check PM2.5 statistics by year
df['year'] = df.index.year
yearly_stats = df.groupby('year')['PM2.5'].agg(['mean', 'std', 'count'])

print("PM2.5 Statistics by Year:")
print(yearly_stats)
print("\n")

# Visual inspection
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: PM2.5 over time with yearly means
axes[0].plot(df.index, df['PM2.5'], alpha=0.3, linewidth=0.5, label='Daily PM2.5')
for year in df['year'].unique():
    year_data = df[df['year'] == year]
    axes[0].axhline(y=year_data['PM2.5'].mean(),
                    xmin=(year-df['year'].min())/(df['year'].max()-df['year'].min()),
                    xmax=(year+1-df['year'].min())/(df['year'].max()-df['year'].min()),
                    color='red', linewidth=2, alpha=0.7)
    axes[0].text(year_data.index[len(year_data)//2], year_data['PM2.5'].mean() + 2,
                 f'{year_data["PM2.5"].mean():.1f}', fontsize=8)

axes[0].set_title('PM2.5 Time Series with Yearly Averages', fontweight='bold')
axes[0].set_ylabel('PM2.5 (μg/m³)')
axes[0].grid(True, alpha=0.3)

# Plot 2: Distribution by year (boxplot)
df.boxplot(column='PM2.5', by='year', ax=axes[1])
axes[1].set_title('PM2.5 Distribution by Year', fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('PM2.5 (μg/m³)')

plt.tight_layout()
plt.savefig('temporal_distribution_check.png', dpi=300)
plt.show()

# Check for distribution shifts
print("\nDistribution Shift Analysis:")
early_years = df[df['year'] <= 2017]['PM2.5']
recent_years = df[df['year'] >= 2020]['PM2.5']

print(f"2016-2017 PM2.5: mean={early_years.mean():.2f}, std={early_years.std():.2f}")
print(f"2020-2024 PM2.5: mean={recent_years.mean():.2f}, std={recent_years.std():.2f}")
print(f"Difference: {abs(early_years.mean() - recent_years.mean()):.2f} μg/m³")

from scipy import stats
t_stat, p_value = stats.ttest_ind(early_years.dropna(), recent_years.dropna())
print(f"T-test p-value: {p_value:.6f}")
if p_value < 0.05:
    print("⚠️ SIGNIFICANT DISTRIBUTION SHIFT DETECTED!")
else:
    print("✓ No significant distribution shift")


# Check What Fold 1 Contains
df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
df = df.resample('D').mean()

# Normalize PM2.5 within each year to remove temporal drift
df['year'] = df.index.year
df['PM2.5'] = df.groupby('year')['PM2.5'].transform(lambda x: (x - x.mean()) / x.std())

# Use PM2.5_normalized as target instead of PM2.5
target = 'PM2.5'

original_features = ['air_pressure', 'humidity', 'rainfall', 'temperature',
                         'wind_direction', 'wind_speed', 'PM10', 'Month', 'year']

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
# features = [f for f in features if f in df.columns]

features = [
        'PM2.5_lag1',  # Yesterday's PM2.5 (0.79 correlation)
        'PM2.5_rolling_mean_7',  # Weekly trend (0.60 correlation)
        'PM10',  # Related pollutant (0.67 correlation)
        'temperature',  # Weather impact
        'humidity',  # Moisture
        'wind_speed',  # Dispersion
        'wind_direction',  # Source direction
        'year'
]

# Fill gaps
for col in features + [target]:
        if col in df.columns:
            df[col] = df[col].ffill(limit=3)

df_clean = df.dropna(subset=features + [target])

# Create sequences
X, y, dates = create_sequences_direct(df_clean, features, 'PM2.5',
                                       sequence_length=7, prediction_days=1)

# Check fold splits
tscv = TimeSeriesSplit(n_splits=9)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    train_dates = [dates[i] for i in train_idx]
    test_dates = [dates[i] for i in test_idx]

    print(f"\nFold {fold + 1}:")
    print(f"  Train: {min(train_dates)} to {max(train_dates)}")
    print(f"  Test:  {min(test_dates)} to {max(test_dates)}")
    print(f"  Train PM2.5: {y[train_idx].mean():.2f} ± {y[train_idx].std():.2f}")
    print(f"  Test PM2.5:  {y[test_idx].mean():.2f} ± {y[test_idx].std():.2f}")

    # Check if distributions are similar
    diff = abs(y[train_idx].mean() - y[test_idx].mean())
    if diff > 3.0:
        print(f"  ⚠️ WARNING: Mean difference = {diff:.2f} μg/m³ (distribution shift!)")