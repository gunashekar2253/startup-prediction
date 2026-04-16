# Phase 1: Next Steps & Data Download Instructions

I have successfully laid the foundation for your Data Engineering pipeline! 
In the `SOURCE CODE` directory, you will now see:
1. [setup_db.py](file:///c:/Users/Dell/Desktop/13/SOURCE%20CODE/An%20Efficient%20Novel%20Approach%20for%20Prediction%20of%20Start-Up%20Company%20Success%20Rates%20through%20ML%20Paradigms/setup_db.py): A Python script that creates a robust SQLite database schema for the startups, funding rounds, founders, and digital footprints. (I have already run this for you, resulting in the creation of the `data_pipeline/startup_data.db` file).
2. [etl_pipeline.py](file:///c:/Users/Dell/Desktop/13/SOURCE%20CODE/An%20Efficient%20Novel%20Approach%20for%20Prediction%20of%20Start-Up%20Company%20Success%20Rates%20through%20ML%20Paradigms/etl_pipeline.py): The Extract, Transform, and Load script that can take a raw CSV file, clean up missing values, do feature engineering (like calculating startup age), and insert the data into the SQL database.

## Your Task: Download the Free Rich Dataset

To make this pipeline work exactly like a real-world system without paying for expensive APIs, we are going to use a massive free public dataset.

**Step 1:** Go to Kaggle and download the **Crunchbase 2013 Snapshot** dataset. You can search for "Crunchbase Startups Dataset" on Kaggle (it's a very popular free dataset containing thousands of companies, funding rounds, and acquisitions).
**Step 2:** Extract the downloaded ZIP file.
**Step 3:** Take the main companies CSV file (usually named `companies.csv` or `crunchbase_companies.csv`) and place it inside your project's `Dataset/` folder. Ensure it is named `crunchbase_companies.csv`.
**Step 4:** Once the file is in place, you can run the ETL pipeline by executing the following command in your terminal while inside the `SOURCE CODE` directory:

```bash
python etl_pipeline.py
```

Once this script finishes, your SQLite database (`startup_data.db`) will be populated with the rich dataset. It is then ready for Phase 2 (Advanced Machine Learning)!
