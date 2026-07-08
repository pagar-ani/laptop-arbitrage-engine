import pandas as pd
import json
import sys
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# Mode options: "SINGLE_DICT", "BATCH_DICTS", "CSV_FILE", "INTERACTIVE"
ACTIVE_MODE = "CSV_FILE"

CSV_FILE_PATH = "Kaggle_Master_Dataset.csv"

# ==========================================
# REQUIRED SCHEMA (Hard Constraint)
# ==========================================
REQUIRED_COLUMNS = [
    "brand", "cpu", "gpu", "ram_size_gb", "storage_size_gb", 
    "model_series", "display_size", "description", 
    "fingerprint", "touch", "warranty", "urgent_sell", 
    "city", "state", "images_count"
]
# Note: storage_tech is natively enforced as "SSD" by the engine.
# created_at is automatically handled by the engine to compute listing_age_days.

# ==========================================
# PRESETS
# ==========================================

# Preset 1: Single Entry Dictionary
SINGLE_LAPTOP_DICT = {
    "brand": "Dell",
    "cpu": "Intel Core i7 12700H",
    "gpu": "NVIDIA RTX 3060",
    "ram_size_gb": 16,
    "storage_size_gb": 512,
    "model_series": "Inspiron",
    "display_size": "15.6",
    "description": "Used for 6 months, minor scratch on lid. Battery backup is good. Need urgent cash.",
    "fingerprint": "Yes",
    "touch": "No",
    "warranty": "Yes",
    "urgent_sell": "Yes",
    "city": "Mumbai",
    "state": "Maharashtra",
    "images_count": 6
}

# Preset 2: Batch Entry (List of Dictionaries)
BATCH_LAPTOP_DICTS = [
    SINGLE_LAPTOP_DICT,
    {
        "brand": "Lenovo",
        "cpu": "Ryzen 7 5800H",
        "gpu": "RTX 3050 Ti",
        "ram_size_gb": 8,
        "storage_size_gb": 256,
        "model_series": "Legion",
        "display_size": "15.6",
        "description": "Mint condition, flawless.",
        "fingerprint": "No",
        "touch": "No",
        "warranty": "No",
        "urgent_sell": "No",
        "city": "Delhi",
        "state": "Delhi",
        "images_count": 3
    }
]

# ==========================================
# IMPLEMENTATION
# ==========================================

def _interactive_prompt():
    """Generates a dict via terminal prompts for live market entry."""
    print("\n[Data Supplier] Interactive Market Entry Mode")
    print("-" * 50)
    data = {}
    for col in REQUIRED_COLUMNS:
        val = input(f"Enter {col}: ").strip()
        
        # Coerce numeric types
        if col in ["ram_size_gb", "storage_size_gb", "images_count"]:
            try:
                val = float(val) if "gb" in col else int(val)
            except ValueError:
                print(f"[ERROR] Invalid numeric input for {col}. Defaulting to 0.")
                val = 0
        data[col] = val
    return [data]

def get_laptops() -> pd.DataFrame:
    """
    Main ingestion function called by the Arbitrage Engine.
    Returns a Pandas DataFrame formatted for the engine's strict pipeline.
    """
    if ACTIVE_MODE == "SINGLE_DICT":
        df = pd.DataFrame([SINGLE_LAPTOP_DICT])
    elif ACTIVE_MODE == "BATCH_DICTS":
        df = pd.DataFrame(BATCH_LAPTOP_DICTS)
    elif ACTIVE_MODE == "CSV_FILE":
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            # Filter for records that have warranty AND exclude Apple/Asus
            if 'warranty' in df.columns:
                mask = df['warranty'].fillna('no').astype(str).str.lower().str.strip() == 'yes'
                if 'brand' in df.columns:
                    brand_mask = ~df['brand'].fillna('').astype(str).str.lower().str.contains('apple|mac|asus|rog|strix|tuf|zenbook|vivobook', regex=True)
                    mask = mask & brand_mask
                df = df[mask].copy()
        except Exception as e:
            print(f"[FATAL] Failed to read {CSV_FILE_PATH}: {e}")
            sys.exit(1)
    elif ACTIVE_MODE == "INTERACTIVE":
        df = pd.DataFrame(_interactive_prompt())
    else:
        raise ValueError(f"Unknown ACTIVE_MODE: {ACTIVE_MODE}")

    # Validation Gate
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[FATAL] Data feed missing required columns: {missing}")

    print(f"[SUPPLIER] Pipeline populated with {len(df)} records. Forwarding to engine...")
    return df

if __name__ == "__main__":
    # Test execution and CSV export
    df = get_laptops()
    print(df.head())
    
    out_file = "Warranty_Filtered_Data.csv"
    df.to_csv(out_file, index=False)
    print(f"[SUPPLIER] Extracted data saved to {out_file}")


