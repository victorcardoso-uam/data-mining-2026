"""
Data Preparation - Chihuahua Climate Dataset
Course: Data Mining | Universidad Anáhuac Mayab

This script cleans and prepares the Chihuahua climate dataset
for use across all predictive models in the final project.

Dataset source: NASA POWER / CONAGUA - Chihuahua State
Variables: Surface Irradiance, Precipitation, Wind Speed, Earth Skin Temp
Period: 2020–2025 | Spatial resolution: 0.5° grid
"""

import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RAW_PATH = "data/raw/Data_Chihuahua.csv"
OUTPUT_PATH = "data/processed/chihuahua_dataset.csv"

os.makedirs("data/processed", exist_ok=True)

# --------------------------------------------------
# 1. LOAD RAW DATA
# --------------------------------------------------

print("=" * 60)
print("DATA PREPARATION - CHIHUAHUA CLIMATE DATASET")
print("=" * 60)

df = pd.read_csv(RAW_PATH, skiprows=1)

# Keep only valid parameter rows
valid_params = ['Surface Irradiance', 'Precipitation', 'Wind Speed', 'Earth Skin Temp']
df = df[df['PARAMETER'].isin(valid_params)].copy()

# Convert columns to numeric
numeric_cols = ['YEAR', 'LAT', 'LON',
                'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[pd.to_numeric(df['YEAR'], errors='coerce').notna()].copy()

print(f"\nRaw rows loaded (after filtering): {len(df)}")
print(f"Parameters present: {df['PARAMETER'].unique()}")

# --------------------------------------------------
# 2. RESHAPE: MELT MONTHS → LONG FORMAT → PIVOT
# --------------------------------------------------

months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
month_num = {m: i + 1 for i, m in enumerate(months)}

melted = df.melt(
    id_vars=['PARAMETER', 'YEAR', 'LAT', 'LON'],
    value_vars=months,
    var_name='MONTH',
    value_name='VALUE'
)
melted['MONTH_NUM'] = melted['MONTH'].map(month_num)

pivot = melted.pivot_table(
    index=['YEAR', 'LAT', 'LON', 'MONTH', 'MONTH_NUM'],
    columns='PARAMETER',
    values='VALUE'
).reset_index()

pivot.columns.name = None
pivot.columns = [c.replace(' ', '_') for c in pivot.columns]

# --------------------------------------------------
# 3. CLEAN & FEATURE ENGINEERING
# --------------------------------------------------

clean = pivot.dropna().copy()
clean = clean.rename(columns={
    'Earth_Skin_Temp': 'temperature_c',
    'Surface_Irradiance': 'solar_irradiance_wm2',
    'Precipitation': 'precipitation_mm',
    'Wind_Speed': 'wind_speed_ms'
})

# Season feature
def get_season(m):
    if m in [12, 1, 2]:   return 'Winter'
    elif m in [3, 4, 5]:  return 'Spring'
    elif m in [6, 7, 8]:  return 'Summer'
    else:                  return 'Fall'

clean['season'] = clean['MONTH_NUM'].apply(get_season)

# Final column order
final_cols = [
    'YEAR', 'LAT', 'LON', 'MONTH', 'MONTH_NUM', 'season',
    'solar_irradiance_wm2', 'precipitation_mm', 'wind_speed_ms',
    'temperature_c'
]
clean = clean[final_cols].reset_index(drop=True)

# --------------------------------------------------
# 4. SUMMARY
# --------------------------------------------------

print(f"\nFinal dataset shape: {clean.shape}")
print("\nColumn descriptions:")
print("  YEAR, LAT, LON         → Spatiotemporal identifiers")
print("  MONTH, MONTH_NUM        → Month name and number (1–12)")
print("  season                  → Meteorological season")
print("  solar_irradiance_wm2   → Surface irradiance (W/m²) — INPUT")
print("  precipitation_mm        → Monthly precipitation (mm) — INPUT")
print("  wind_speed_ms           → Wind speed (m/s) — INPUT")
print("  temperature_c           → Earth Skin Temperature (°C) — TARGET (y)")

print("\nBasic statistics:")
print(clean[['solar_irradiance_wm2', 'precipitation_mm', 'wind_speed_ms', 'temperature_c']].describe().round(3))

print(f"\nMissing values: {clean.isnull().sum().sum()}")

# --------------------------------------------------
# 5. SAVE
# --------------------------------------------------

clean.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataset saved to: {OUTPUT_PATH}")
print("=" * 60)
