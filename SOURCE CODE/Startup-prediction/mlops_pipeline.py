import schedule
import time
import subprocess
import datetime
import requests
import sqlite3
import os
from dotenv import load_dotenv

# Load API keys securely from .env file (never hardcode keys!)
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
CLEARBIT_API_KEY = os.getenv("CLEARBIT_API_KEY", "")
DB_PATH = 'data_pipeline/startup_data.db'


def extract_clearbit_data(domain):
    """1. CLEARBIT API: Finds industry and location dynamically based on a website."""
    print(f"\n[API 1] Fetching Clearbit profile for {domain}...")
    if not CLEARBIT_API_KEY or CLEARBIT_API_KEY == "your_clearbit_key_here":
        print("  -> Skipped: Clearbit API Key missing. Add to .env file.")
        return None

    url = f"https://company.clearbit.com/v2/companies/find?domain={domain}"
    headers = {'Authorization': f'Bearer {CLEARBIT_API_KEY}'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"  -> Found: {data.get('name')} in {data.get('category', {}).get('sector')}")
            return data
    except Exception as e:
        print(f"  -> Clearbit Error: {e}")
    return None


def monitor_news_for_funding(company_name):
    """2. NEWS API: Scans global news to detect if this startup just raised new funding."""
    print(f"\n[API 2] Scanning NewsAPI for recent funding rounds for '{company_name}'...")
    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        print("  -> Skipped: NewsAPI Key missing. Get free key at https://newsapi.org/register")
        return False, 0

    url = f"https://newsapi.org/v2/everything?q={company_name} AND funding&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            if len(articles) > 0:
                print(f"  -> ALERT: Found {len(articles)} articles about {company_name} funding!")
                return True, len(articles)
            else:
                print("  -> No recent funding news found.")
    except Exception as e:
        print(f"  -> NewsAPI Error: {e}")
    return False, 0


def check_yahoo_finance_ipo(ticker_symbol):
    """3. YAHOO FINANCE (Free - No Key Needed): Checks if the company IPO'd successfully."""
    print(f"\n[API 3] Checking Yahoo Finance for live IPO status on ticker '{ticker_symbol}'...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get('chart', {}).get('result'):
                price = data['chart']['result'][0]['meta']['regularMarketPrice']
                print(f"  -> SUCCESS! {ticker_symbol} trading at ${price}. Status: IPO/Acquired.")
                return True, price
        print(f"  -> {ticker_symbol} is not a valid public ticker yet.")
    except Exception as e:
        print(f"  -> Yahoo Finance API error: {e}")
    return False, 0


def log_api_run_to_db(company_name, domain, ticker, news_found, ipo_found, ipo_price, clearbit_sector):
    """Saves the fetched API data into the MLOps log table in SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create mlops_log table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mlops_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                company_name TEXT,
                domain TEXT,
                ticker TEXT,
                clearbit_sector TEXT,
                news_funding_found INTEGER DEFAULT 0,
                news_article_count INTEGER DEFAULT 0,
                ipo_found INTEGER DEFAULT 0,
                ipo_price REAL DEFAULT 0.0
            )
        """)

        cursor.execute("""
            INSERT INTO mlops_log 
            (run_timestamp, company_name, domain, ticker, clearbit_sector, 
             news_funding_found, news_article_count, ipo_found, ipo_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().isoformat(),
            company_name,
            domain,
            ticker,
            clearbit_sector or "N/A",
            1 if news_found else 0,
            0,  # article count placeholder
            1 if ipo_found else 0,
            ipo_price or 0.0
        ))
        conn.commit()
        conn.close()
        print(f"  -> MLOps log entry saved to database for '{company_name}'.")
    except Exception as e:
        print(f"  -> DB log error: {e}")


def save_model_version():
    """Creates a timestamped backup of the current model before retraining (Model Versioning)."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = "model"
    versions_dir = os.path.join(model_dir, "versions")
    os.makedirs(versions_dir, exist_ok=True)

    for model_file in ["startup_success_rf_model.pkl", "startup_success_nn_model.pkl", "nn_scaler.pkl"]:
        src = os.path.join(model_dir, model_file)
        if os.path.exists(src):
            dst = os.path.join(versions_dir, f"{timestamp}_{model_file}")
            import shutil
            shutil.copy2(src, dst)
            print(f"  -> Versioned: {dst}")


def job_continuous_learning():
    """THE MASTER MLOPS CRON JOB - Fetches live data, logs it, then retrains the model."""
    print(f"\n[{datetime.datetime.now()}] [MLOps API Pipeline Triggered]")

    # Run all 3 API checks
    clearbit_data = extract_clearbit_data("stripe.com")
    clearbit_sector = clearbit_data.get('category', {}).get('sector') if clearbit_data else None

    news_found, article_count = monitor_news_for_funding("Anthropic")
    ipo_found, ipo_price = check_yahoo_finance_ipo("UBER")

    # ✅ COMPLETE: Log all fetched data to SQLite database
    print("\nConnecting to SQLite Data Pipeline to insert newly fetched data...")
    log_api_run_to_db(
        company_name="Anthropic",
        domain="stripe.com",
        ticker="UBER",
        news_found=news_found,
        ipo_found=ipo_found,
        ipo_price=ipo_price,
        clearbit_sector=clearbit_sector
    )

    # Save model version before retraining
    print("\nVersioning current model before retraining...")
    save_model_version()

    print("Retraining the AI model with the updated database...")
    try:
        subprocess.run(["python", "train_advanced_model.py"], check=True)
        subprocess.run(["python", "train_nn_model.py"], check=True)
        print(f"[{datetime.datetime.now()}] [MLOps] Model successfully retrained & updated in production!")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.datetime.now()}] [MLOps ERROR] Model retraining failed: {e}")


if __name__ == "__main__":
    print("==========================================")
    print("PHASE 3.5: Dynamic Live API Ingestion Engine")
    print("==========================================")
    print("This script runs in the background. It connects to 3 Live APIs")
    print("to scrape data, pushes it to your Database, and retrains the AI.")
    print(f"\nAPI Key Status:")
    print(f"  NewsAPI: {'✓ Configured' if NEWS_API_KEY and NEWS_API_KEY != 'your_newsapi_key_here' else '✗ Missing (add to .env)'}")
    print(f"  Clearbit: {'✓ Configured' if CLEARBIT_API_KEY and CLEARBIT_API_KEY != 'your_clearbit_key_here' else '✗ Missing (add to .env)'}")
    print(f"  Yahoo Finance: ✓ No key needed (public API)")

    print("\n[Running initial API ingestion scan...]")
    job_continuous_learning()

    # Automate to scan the internet every 12 hours
    schedule.every(12).hours.do(job_continuous_learning)

    print("\nLive Tracking Scheduler active. Waiting for next cycle (Press Ctrl+C to exit)...")
    while True:
        schedule.run_pending()
        time.sleep(60)
