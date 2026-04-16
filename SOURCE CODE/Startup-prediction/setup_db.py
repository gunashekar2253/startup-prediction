import sqlite3
import os

def create_database():
    # Ensure the directory exists
    os.makedirs('data_pipeline', exist_ok=True)
    db_path = 'data_pipeline/startup_data.db'
    
    # Connect to SQLite database (this creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Startups Table (Core Info including Industry & Location)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS startups (
        company_id TEXT PRIMARY KEY,
        name TEXT,
        category_code TEXT,
        status TEXT,
        country_code TEXT,
        city TEXT,
        founded_year INTEGER
    )
    ''')

    # 2. Funding Rounds Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS funding_rounds (
        funding_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id TEXT,
        funding_round_type TEXT,
        raised_amount_usd REAL,
        funded_year INTEGER,
        FOREIGN KEY(company_id) REFERENCES startups(company_id)
    )
    ''')

    # 3. Founders Profile Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS founders (
        founder_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id TEXT,
        first_name TEXT,
        last_name TEXT,
        title TEXT,
        is_founder BOOLEAN,
        FOREIGN KEY(company_id) REFERENCES startups(company_id)
    )
    ''')

    # 4. Digital Footprint Metrics Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS digital_footprint (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id TEXT,
        twitter_username TEXT,
        homepage_url TEXT,
        FOREIGN KEY(company_id) REFERENCES startups(company_id)
    )
    ''')

    conn.commit()
    conn.close()
    print(f"Database successfully created at: {db_path}")
    print("Tables created: startups, funding_rounds, founders, digital_footprint")

if __name__ == "__main__":
    create_database()
