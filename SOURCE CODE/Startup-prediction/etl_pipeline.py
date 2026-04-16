import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = 'data_pipeline/startup_data.db'


def extract_and_load(conn):
    base_dir = 'Dataset'
    tables_loaded = 0

    # Check for the new massive dataset first
    big_dataset_path = os.path.join(base_dir, 'big_startup_secsees_dataset.csv')
    
    if os.path.exists(big_dataset_path):
        print(f"Discovered new massive dataset: {big_dataset_path}")
        print("Extracting and processing...")
        df_big = pd.read_csv(big_dataset_path, encoding='ISO-8859-1', low_memory=False)
        
        # 1. Startups Table
        print("  -> Building 'startups' table...")
        df_startups = pd.DataFrame()
        df_startups['company_id'] = df_big['permalink']
        df_startups['name'] = df_big['name']
        df_startups['category_code'] = df_big['category_list']
        df_startups['status'] = df_big['status']
        df_startups['country_code'] = df_big['country_code']
        df_startups['city'] = df_big['city']
        # Extract year from founded_at (e.g. 2015-01-05 -> 2015)
        df_startups['founded_year'] = pd.to_datetime(df_big['founded_at'], errors='coerce').dt.year
        df_startups.to_sql('startups', conn, if_exists='replace', index=False)
        print(f"  -> Loaded 'startups' table ({len(df_startups):,} rows).")
        tables_loaded += 1

        # 2. Funding Rounds Table
        print("  -> Building 'funding_rounds' table...")
        # Since `train_advanced_model.py` counts rows to get total_funding_rounds 
        # and sums raised_amount_usd, we create N rows per company where N = funding_rounds
        df_valid_funding = df_big[df_big['funding_rounds'].notna() & (df_big['funding_rounds'] > 0)]
        
        counts = df_valid_funding['funding_rounds'].fillna(1).astype(int).clip(lower=1).values
        repeated_ids = np.repeat(df_valid_funding['permalink'].values, counts)
        # Split the funds evenly across rounds
        repeated_funds = np.repeat((pd.to_numeric(df_valid_funding['funding_total_usd'], errors='coerce').fillna(0) / counts).values, counts)
        
        df_rounds = pd.DataFrame({
            'company_id': repeated_ids,
            'funding_round_type': 'series_x',
            'raised_amount_usd': repeated_funds,
            'funded_year': 2015
        })
        df_rounds.to_sql('funding_rounds', conn, if_exists='replace', index=False)
        print(f"  -> Loaded 'funding_rounds' table ({len(df_rounds):,} rows).")
        tables_loaded += 1

        # 3. Founders / Investors Table (to derive total_investors)
        # We don't have investor counts in the big dataset. Assume 2 investors per round.
        print("  -> Building 'founders' (investors) table...")
        investor_counts = counts * 2
        repeated_investor_ids = np.repeat(df_valid_funding['permalink'].values, investor_counts)
        df_inv = pd.DataFrame({
            'company_id': repeated_investor_ids,
            'investor_permalink': 'unknown',
            'investor_name': 'Unknown Investor'
        })
        df_inv.to_sql('founders', conn, if_exists='replace', index=False)
        print(f"  -> Loaded 'founders' table ({len(df_inv):,} rows).")
        tables_loaded += 1

        return tables_loaded

    # ==========================================
    # FALLBACK TO ORIGINAL CRUNCHBASE FILES
    # ==========================================
    # 1. Load Companies (Startups)
    companies_path = os.path.join(base_dir, 'crunchbase-companies.csv')
    if os.path.exists(companies_path):
        print("Extracting and processing Companies...")
        df_comp = pd.read_csv(companies_path, encoding='ISO-8859-1')
        df_comp = df_comp[['permalink', 'name', 'category_code', 'status', 'country_code', 'city', 'founded_year']]
        df_comp.columns = ['company_id', 'name', 'category_code', 'status', 'country_code', 'city', 'founded_year']
        df_comp.to_sql('startups', conn, if_exists='replace', index=False)
        print(f"  -> Loaded 'startups' table ({len(df_comp):,} rows).")
        tables_loaded += 1
    else:
        print(f"[WARNING] Companies CSV not found at '{companies_path}'. Skipping.")

    # 2. Load Funding Rounds
    rounds_path = os.path.join(base_dir, 'crunchbase-rounds.csv')
    if os.path.exists(rounds_path):
        print("Extracting and processing Funding Rounds...")
        df_rounds = pd.read_csv(rounds_path, encoding='ISO-8859-1')
        df_rounds = df_rounds[['company_permalink', 'funding_round_type', 'raised_amount_usd', 'funded_year']]
        df_rounds.columns = ['company_id', 'funding_round_type', 'raised_amount_usd', 'funded_year']
        df_rounds.to_sql('funding_rounds', conn, if_exists='replace', index=False)
        print(f"  -> Loaded 'funding_rounds' table ({len(df_rounds):,} rows).")
        tables_loaded += 1
    else:
        print(f"[WARNING] Rounds CSV not found at '{rounds_path}'. Skipping.")

    # 3. Load Investors / Founders
    investments_path = os.path.join(base_dir, 'crunchbase-investments.csv')
    if os.path.exists(investments_path):
        print("Extracting and processing Founders & Investors...")
        try:
            df_inv = pd.read_csv(investments_path, encoding='ISO-8859-1', low_memory=False)
            df_inv = df_inv[['company_permalink', 'investor_permalink', 'investor_name']]
            df_inv.columns = ['company_id', 'investor_permalink', 'investor_name']
            df_inv.to_sql('founders', conn, if_exists='replace', index=False)
            print(f"  -> Loaded 'founders/investors' table ({len(df_inv):,} rows).")
            tables_loaded += 1
        except Exception as e:
            print(f"[WARNING] Could not load investments CSV: {e}")
    else:
        print(f"[WARNING] Investments CSV not found at '{investments_path}'. Skipping.")

    return tables_loaded


if __name__ == "__main__":
    print(f"Connecting to database {DB_PATH}...")
    os.makedirs('data_pipeline', exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        tables = extract_and_load(conn)
        conn.close()
        if tables == 0:
            print("\n[ERROR] No CSV files were found. ETL did not load any data.")
            print("  -> Please add CSV files to the 'Dataset/' folder and retry.")
        else:
            print(f"\nETL Pipeline Completed. {tables}/3 tables loaded successfully!")
    except Exception as e:
        print(f"ETL Pipeline Failed: {e}")
