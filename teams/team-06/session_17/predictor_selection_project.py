import os
import pandas as pd

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Usamos el nombre exacto de tu archivo según la captura
DATA_PATH = os.path.join(BASE_DIR, "amazon.csv") 
TARGET_COL = "rating"

def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        print("Asegúrate de que el archivo se llame amazon.csv y esté en la misma carpeta.")
        return

    # 1. Cargar datos
    df = pd.read_csv(DATA_PATH)

    # 2. LIMPIEZA DE DATOS (Interna, no crea archivos nuevos)
    # Quitamos el símbolo ₹ y las comas para que Python pueda hacer cálculos
    cols_to_fix = ['discounted_price', 'actual_price', 'rating_count']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Limpieza de la columna rating (el target)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL].astype(str).replace('|', '0'), errors='coerce')
    
    # Borramos filas que hayan quedado vacías tras la limpieza
    df = df.dropna(subset=[TARGET_COL] + [c for c in cols_to_fix if c in df.columns])

    print("\n=== DATASET OVERVIEW (AMAZON) ===")
    print(f"Total rows after cleaning: {df.shape[0]}")
    print(f"Target: {TARGET_COL}")

    # 3. Selección de Variables
    exclude_cols = ["product_id", "product_name", "category", "user_id", "review_id"]
    candidate_cols = ["discounted_price", "actual_price", "rating_count"]

    # 4. Mostrar Correlación
    print("\n=== NUMERIC CORRELATION WITH RATING ===")
    correlations = df[candidate_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    print(correlations.sort_values(ascending=False))

    # --- TEAM 06 FINAL ANALYSIS (ENGLISH) ---
    print("\n" + "="*50)
    print("         TEAM 06 - AMAZON ANALYSIS")
    print("="*50)
    
    print(f"\n1. Target Variable: {TARGET_COL}")
    print(f"2. Independent Variables: {candidate_cols}")
    print(f"3. Excluded Variables: IDs, Names, and URLs (Rule 5).")
    
    print("\n4. Logic:")
    print("   We selected prices and rating count because they represent")
    print("   the financial value and the popularity of the product,")
    print("   which are logical predictors for the overall rating.")
    
    print("\n5. Redundancy:")
    print("   'discounted_price' and 'actual_price' show high collinearity.")
    print("   In the final model, we should choose only one to avoid noise.")
    
    print("\n6. Direction of Causality:")
    print("   If the target changed to 'discounted_price', the 'rating'")
    print("   would then become an independent variable to help predict price.")
    print("="*50)

if __name__ == "__main__":
    main()