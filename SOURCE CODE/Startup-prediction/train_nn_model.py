import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import datetime

DB_PATH = 'data_pipeline/startup_data.db'
MODEL_DIR = 'model'


def get_training_data():
    conn = sqlite3.connect(DB_PATH)

    query_startups = "SELECT company_id, status, founded_year, category_code, country_code FROM startups WHERE status IS NOT NULL"
    df_startups = pd.read_sql(query_startups, conn)
    df_startups['is_success'] = df_startups['status'].apply(lambda x: 0 if x == 'closed' else 1)

    query_funding = "SELECT company_id, COUNT(funding_round_type) as total_funding_rounds, SUM(raised_amount_usd) as total_raised_usd FROM funding_rounds GROUP BY company_id"
    df_funding = pd.read_sql(query_funding, conn)

    query_founders = "SELECT company_id, COUNT(investor_name) as total_investors FROM founders GROUP BY company_id"
    df_founders = pd.read_sql(query_founders, conn)

    conn.close()

    df = df_startups.merge(df_funding, on='company_id', how='left')
    df = df.merge(df_founders, on='company_id', how='left')

    df['total_funding_rounds'] = df['total_funding_rounds'].fillna(0)
    df['total_raised_usd'] = df['total_raised_usd'].fillna(0)
    df['total_investors'] = df['total_investors'].fillna(0)
    df['startup_age'] = 2013 - df['founded_year']
    df['startup_age'] = df['startup_age'].fillna(df['startup_age'].median())

    # Use saved label encoders if they exist (for consistent encoding with RF model)
    le_category_path = os.path.join(MODEL_DIR, 'le_category.pkl')
    le_country_path = os.path.join(MODEL_DIR, 'le_country.pkl')

    df['category_code'] = df['category_code'].fillna('unknown')
    df['country_code'] = df['country_code'].fillna('unknown')

    if os.path.exists(le_category_path):
        le_category = joblib.load(le_category_path)
        # Handle unseen labels gracefully
        known = set(le_category.classes_)
        df['category_code'] = df['category_code'].apply(lambda x: x if x in known else 'unknown')
        df['category_encoded'] = le_category.transform(df['category_code'])
    else:
        le_category = LabelEncoder()
        df['category_encoded'] = le_category.fit_transform(df['category_code'].astype(str))

    if os.path.exists(le_country_path):
        le_country = joblib.load(le_country_path)
        known = set(le_country.classes_)
        df['country_code'] = df['country_code'].apply(lambda x: x if x in known else 'unknown')
        df['country_encoded'] = le_country.transform(df['country_code'])
    else:
        le_country = LabelEncoder()
        df['country_encoded'] = le_country.fit_transform(df['country_code'].astype(str))

    features = ['total_funding_rounds', 'total_raised_usd', 'total_investors', 'startup_age', 'category_encoded', 'country_encoded']
    X = df[features]
    y = df['is_success']
    return X, y


def train_neural_network(X, y):
    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"  -> Original: {len(y)} samples | After SMOTE: {len(y_resampled)} samples")

    print("Scaling Features for Deep Learning...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

    print("Training Multi-Layer Perceptron (Deep Neural Network)...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_train, y_train)

    preds = nn_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Neural Network Accuracy: {acc * 100:.2f}%")

    return nn_model, scaler


if __name__ == "__main__":
    print("==========================================")
    print("PHASE 5: Deep Learning (Neural Networks)")
    print("==========================================\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = get_training_data()
    nn_model, scaler = train_neural_network(X, y)

    # Version backup of old NN model
    old_path = os.path.join(MODEL_DIR, 'startup_success_nn_model.pkl')
    if os.path.exists(old_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        versions_dir = os.path.join(MODEL_DIR, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        import shutil
        shutil.copy2(old_path, os.path.join(versions_dir, f"{timestamp}_startup_success_nn_model.pkl"))

    joblib.dump(nn_model, os.path.join(MODEL_DIR, 'startup_success_nn_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'nn_scaler.pkl'))
    print("Deep Learning Model and Scaler successfully saved!")
