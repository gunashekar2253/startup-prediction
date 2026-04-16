import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import datetime

try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    print("SHAP not installed. Install via `pip install shap` for full XAI support.")

DB_PATH = 'data_pipeline/startup_data.db'
MODEL_DIR = 'model'


def get_training_data():
    """Extracts and merges data from SQLite to create the Feature Matrix (X) and Target (y)"""
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    # 1. Load Startups (Target and core features) — now including category and country
    query_startups = """
        SELECT company_id, category_code, status, country_code, founded_year 
        FROM startups 
        WHERE status IS NOT NULL
    """
    df_startups = pd.read_sql(query_startups, conn)
    df_startups['is_success'] = df_startups['status'].apply(lambda x: 0 if x == 'closed' else 1)

    # 2. Aggregate Funding Data
    query_funding = """
        SELECT company_id, 
               COUNT(funding_round_type) as total_funding_rounds, 
               SUM(raised_amount_usd) as total_raised_usd 
        FROM funding_rounds 
        GROUP BY company_id
    """
    df_funding = pd.read_sql(query_funding, conn)

    # 3. Aggregate Investors
    query_founders = """
        SELECT company_id, COUNT(investor_name) as total_investors 
        FROM founders 
        GROUP BY company_id
    """
    df_founders = pd.read_sql(query_founders, conn)
    conn.close()

    # Merge datasets
    print("Merging data...")
    df = df_startups.merge(df_funding, on='company_id', how='left')
    df = df.merge(df_founders, on='company_id', how='left')

    return df


def feature_engineering(df):
    print("Engineering features (including category & country)...")

    # Numeric features
    df['total_funding_rounds'] = df['total_funding_rounds'].fillna(0)
    df['total_raised_usd'] = df['total_raised_usd'].fillna(0)
    df['total_investors'] = df['total_investors'].fillna(0)
    df['startup_age'] = 2013 - df['founded_year']
    df['startup_age'] = df['startup_age'].fillna(df['startup_age'].median())

    # ✅ NEW: Encode category_code (industry) as numeric feature
    df['category_code'] = df['category_code'].fillna('unknown')
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category_code'].astype(str))

    # ✅ NEW: Encode country_code as numeric feature
    df['country_code'] = df['country_code'].fillna('unknown')
    le_country = LabelEncoder()
    df['country_encoded'] = le_country.fit_transform(df['country_code'].astype(str))

    # Save encoders for use during prediction
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(le_category, os.path.join(MODEL_DIR, 'le_category.pkl'))
    joblib.dump(le_country, os.path.join(MODEL_DIR, 'le_country.pkl'))
    print("  -> Label encoders saved for category & country.")

    features = [
        'total_funding_rounds',
        'total_raised_usd',
        'total_investors',
        'startup_age',
        'category_encoded',   # ✅ NEW FEATURE
        'country_encoded'     # ✅ NEW FEATURE
    ]

    X = df[features]
    y = df['is_success']
    return X, y, features


def train_model(X, y):
    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"  -> Original: {len(y)} samples | After SMOTE: {len(y_resampled)} samples")

    print("Training Advanced Random Forest Model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    return model, X_train, acc


def generate_shap_explanations(model, X_train):
    if not shap_available:
        print("Skipping SHAP Explainability (library not installed).")
        return

    print("\nGenerating SHAP Values for Explainable AI (XAI)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train.head(100))
    print("SHAP explainer successfully initialized.")


def save_artifacts(model, features, acc):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ✅ Model Versioning: Backup old model before overwriting
    versions_dir = os.path.join(MODEL_DIR, "versions")
    os.makedirs(versions_dir, exist_ok=True)
    old_model_path = os.path.join(MODEL_DIR, 'startup_success_rf_model.pkl')
    if os.path.exists(old_model_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        import shutil
        shutil.copy2(old_model_path, os.path.join(versions_dir, f"{timestamp}_startup_success_rf_model.pkl"))
        print(f"  -> Old model backed up to versions/{timestamp}_startup_success_rf_model.pkl")

    model_path = os.path.join(MODEL_DIR, 'startup_success_rf_model.pkl')
    joblib.dump(model, model_path)

    # Save feature names used during training
    joblib.dump(features, os.path.join(MODEL_DIR, 'feature_names.pkl'))

    # Save model metadata
    metadata = {
        'trained_at': datetime.datetime.now().isoformat(),
        'accuracy': round(acc * 100, 2),
        'features': features,
        'n_estimators': 150
    }
    import json
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProduction model saved. Accuracy: {acc * 100:.2f}%")
    print("Phase 2 Model Training Complete!")


if __name__ == "__main__":
    print("==========================================")
    print("PHASE 2: Advanced Machine Learning Pipeline")
    print("==========================================\n")

    try:
        df_raw = get_training_data()
        X, y, features = feature_engineering(df_raw)
        trained_model, X_train, acc = train_model(X, y)
        generate_shap_explanations(trained_model, X_train)
        save_artifacts(trained_model, features, acc)
    except Exception as e:
        print(f"Error during Phase 2 Training: {e}")
        raise
