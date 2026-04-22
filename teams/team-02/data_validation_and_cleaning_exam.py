import pandas as pd

# Load dataset
df = pd.read_csv(r'DATA/RAW/renewable_energy_raw.csv')

# Print dataset shape
print("Dataset Shape:")
print(df.shape)
print("\n" + "="*50 + "\n")

# Print column names
print("Column Names:")
print(df.columns.tolist())
print("\n" + "="*50 + "\n")

# Print data types
print("Data Types:")
print(df.dtypes)
print("\n" + "="*50 + "\n")

# Display missing values per column
print("Missing Values per Column:")
missing_values = df.isnull().sum()
print(missing_values)
print(f"\nTotal missing values: {missing_values.sum()}")
print("\n" + "="*50 + "\n")

# Display number of duplicate rows
print("Duplicate Rows:")
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")
print("\n" + "="*50 + "\n")

# Data Integrity Checks
print("Data Integrity Checks:")
negative_irradiance = (df['irradiance_wm2'] < 0).sum()
print(f"Negative values in irradiance_wm2: {negative_irradiance}")

negative_power = (df['power_kw'] < 0).sum()
print(f"Negative values in power_kw: {negative_power}")

humidity_over_100 = (df['humidity_pct'] > 100).sum()
print(f"Values in humidity_pct greater than 100: {humidity_over_100}")
print("\n" + "="*50 + "\n")

# TASK2 - DATA CLEANING
print("TASK2 - DATA CLEANING")
print("="*50 + "\n")

# Store original shape for comparison
original_shape = df.shape
print(f"Original dataset shape: {original_shape}")

# Replace missing numerical values with median
print("\nReplacing missing values...")
numerical_cols = ['irradiance_wm2', 'temp_c', 'humidity_pct', 'wind_speed_ms', 'power_kw']
for col in numerical_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    print(f"Filled {col} missing values with median: {median_value}")

# Replace missing categorical values with "unknown"
categorical_cols = ['site_id', 'timestamp', 'inverter_status']
for col in categorical_cols:
    df[col] = df[col].fillna('unknown')
    print(f"Filled {col} missing values with 'unknown'")

# Remove duplicate rows
df = df.drop_duplicates()
print(f"\nRemoved duplicate rows")

# Print cleaned dataset shape
cleaned_shape = df.shape
print(f"\nCleaned dataset shape: {cleaned_shape}")
print(f"Rows removed (duplicates): {original_shape[0] - cleaned_shape[0]}")
print(f"Columns: {cleaned_shape[1]}")

# Verify no missing values remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\n" + "="*50 + "\n")

# TASK3 - DATA INTEGRITY FIXES
print("TASK3 - DATA INTEGRITY FIXES")
print("="*50 + "\n")

# Count negative values before fixing
negative_irradiance_before = (df['irradiance_wm2'] < 0).sum()
negative_power_before = (df['power_kw'] < 0).sum()

print(f"Before fixing:")
print(f"Negative irradiance values: {negative_irradiance_before}")
print(f"Negative power values: {negative_power_before}")

# Fix negative irradiance values
df.loc[df['irradiance_wm2'] < 0, 'irradiance_wm2'] = 0
print(f"\nFixed negative irradiance values to 0")

# Fix negative power values
df.loc[df['power_kw'] < 0, 'power_kw'] = 0
print(f"Fixed negative power values to 0")

# Verify fixes
negative_irradiance_after = (df['irradiance_wm2'] < 0).sum()
negative_power_after = (df['power_kw'] < 0).sum()

print(f"\nAfter fixing:")
print(f"Negative irradiance values: {negative_irradiance_after}")
print(f"Negative power values: {negative_power_after}")

print(f"\nIntegrity fixes applied successfully!")

print("\n" + "="*50 + "\n")

# Display first few rows
print("First 5 rows (after integrity fixes):")
print(df.head())

print("\n" + "="*50 + "\n")

# Save cleaned dataset to CSV
output_path = r'DATA/PROCESSED/renewable_energy_cleaned_exam.csv'
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
print(f"Final dataset shape: {df.shape}")
print(f"Total rows saved: {len(df)}")
print(f"Total columns saved: {len(df.columns)}")
