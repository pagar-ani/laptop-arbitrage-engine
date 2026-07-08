import os
import sys
import logging
import pandas as pd
from arbitrage_engine import predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_pipeline(margin=0.0, price_min=0.0, **kwargs):
    logging.info("=== STEP 1: Loading Dataset ===")
    candidate_paths = [
        "../data/Kaggle_Master_Engineered_State.csv",
        "../../Kaggle_Master_Engineered_State.csv",
        "../Kaggle_Master_Engineered_State.csv",
        "Kaggle_Master_Engineered_State.csv"
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        logging.error("Dataset Kaggle_Master_Engineered_State.csv not found in candidate paths.")
        return
        
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} records from {csv_path}.")
    
    logging.info("=== STEP 2: Engine Prediction ===")
    if not df.empty:
        predict(output_csv="../predict/Arbitrage_Opportunities.csv", input_df=df, margin=margin, price_min=price_min)
        logging.info("Prediction complete. Results saved to ../predict/Arbitrage_Opportunities.csv")
    else:
        logging.warning("No data to predict.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Arbitrage Inference Pipeline.")
    parser.add_argument("--date", type=str, default=None, help="Ignored in public release.")
    parser.add_argument("--margin", type=float, default=0.0, help="Negotiation buffer percentage.")
    parser.add_argument("--price-min", type=int, default=0, help="Minimum price filter.")
    parser.add_argument("--no-warranty-filter", action="store_true", help="Ignored in public release.")
    parser.add_argument("--clear-cache", action="store_true", help="Ignored in public release.")
    args = parser.parse_args()
    
    run_pipeline(margin=args.margin, price_min=args.price_min)


