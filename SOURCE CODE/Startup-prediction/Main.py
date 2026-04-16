import os
import json
import sqlite3
import datetime
import subprocess
import threading
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
nlp_analyzer = SentimentIntensityAnalyzer()

try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False

# Initialize Flask App
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secure_key_123')

# Paths
MODEL_PATH = 'model/startup_success_rf_model.pkl'
NN_MODEL_PATH = 'model/startup_success_nn_model.pkl'
NN_SCALER_PATH = 'model/nn_scaler.pkl'
FEATURE_NAMES_PATH = 'model/feature_names.pkl'
LE_CATEGORY_PATH = 'model/le_category.pkl'
LE_COUNTRY_PATH = 'model/le_country.pkl'
DB_PATH = 'data_pipeline/startup_data.db'
METADATA_PATH = 'model/model_metadata.json'

# =========================================================
# Model Loading
# =========================================================
retraining_in_progress = False

def load_models():
    global rf_model, nn_model, nn_scaler, explainer, feature_names, le_category, le_country
    print("Loading Machine Learning Models into memory...")
    try:
        rf_model = joblib.load(MODEL_PATH)
        nn_model = joblib.load(NN_MODEL_PATH)
        nn_scaler = joblib.load(NN_SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH) if os.path.exists(FEATURE_NAMES_PATH) else \
            ['total_funding_rounds', 'total_raised_usd', 'total_investors', 'startup_age']
        le_category = joblib.load(LE_CATEGORY_PATH) if os.path.exists(LE_CATEGORY_PATH) else None
        le_country = joblib.load(LE_COUNTRY_PATH) if os.path.exists(LE_COUNTRY_PATH) else None
        print("All models loaded successfully!")
    except Exception as e:
        rf_model = nn_model = nn_scaler = le_category = le_country = None
        feature_names = ['total_funding_rounds', 'total_raised_usd', 'total_investors', 'startup_age']
        print(f"WARNING: Models not found or failed to load. {e}")

    if rf_model and shap_available:
        try:
            explainer = shap.TreeExplainer(rf_model)
        except Exception:
            explainer = None
    else:
        explainer = None

load_models()


# =========================================================
# PREDICTION HISTORY - Database Setup
# =========================================================
def init_history_db():
    """Create prediction_history table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_funding_rounds REAL,
                total_raised_usd REAL,
                total_investors REAL,
                startup_age REAL,
                founder_bio TEXT,
                prediction TEXT,
                ensemble_confidence REAL,
                rf_confidence REAL,
                nn_confidence REAL,
                nlp_sentiment TEXT,
                xai_explanation TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB init warning: {e}")

init_history_db()


def log_prediction(data, result):
    """Saves a prediction result to the history table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO prediction_history 
            (timestamp, total_funding_rounds, total_raised_usd, total_investors, startup_age,
             founder_bio, prediction, ensemble_confidence, rf_confidence, nn_confidence, nlp_sentiment, xai_explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().isoformat(),
            data.get('total_funding_rounds', 0),
            data.get('total_raised_usd', 0),
            data.get('total_investors', 0),
            data.get('startup_age', 0),
            data.get('founder_bio', ''),
            result.get('prediction'),
            result.get('ensemble_confidence_percent'),
            result.get('rf_confidence_score_percent'),
            result.get('nn_confidence_score_percent'),
            result.get('nlp_sentiment'),
            result.get('xai_explanation')
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"History log error: {e}")


# =========================================================
# INPUT VALIDATION
# =========================================================
def validate_inputs(data):
    """Validates and cleans prediction inputs. Returns (cleaned_data, error_message)."""
    errors = []

    try:
        funding_rounds = float(data.get('total_funding_rounds', 0))
        if funding_rounds < 0:
            errors.append("total_funding_rounds must be >= 0")
        if funding_rounds > 100:
            errors.append("total_funding_rounds seems unrealistic (max 100)")
    except (ValueError, TypeError):
        errors.append("total_funding_rounds must be a number")
        funding_rounds = 0

    try:
        raised_usd = float(data.get('total_raised_usd', 0))
        if raised_usd < 0:
            errors.append("total_raised_usd must be >= 0")
    except (ValueError, TypeError):
        errors.append("total_raised_usd must be a number")
        raised_usd = 0

    try:
        investors = float(data.get('total_investors', 0))
        if investors < 0:
            errors.append("total_investors must be >= 0")
        if investors > 10000:
            errors.append("total_investors seems unrealistic (max 10000)")
    except (ValueError, TypeError):
        errors.append("total_investors must be a number")
        investors = 0

    try:
        age = float(data.get('startup_age', 0))
        if age < 0:
            errors.append("startup_age must be >= 0")
        if age > 200:
            errors.append("startup_age seems unrealistic (max 200 years)")
    except (ValueError, TypeError):
        errors.append("startup_age must be a number")
        age = 0

    if errors:
        return None, errors

    cleaned = {
        **data,
        'total_funding_rounds': funding_rounds,
        'total_raised_usd': raised_usd,
        'total_investors': investors,
        'startup_age': age,
    }
    return cleaned, None


# =========================================================
# FEATURE BUILDING (supports both 4-feature and 6-feature models)
# =========================================================
def build_features(data):
    """Builds the feature DataFrame for prediction, handling optional category/country."""
    base_features = {
        'total_funding_rounds': float(data.get('total_funding_rounds', 0)),
        'total_raised_usd': float(data.get('total_raised_usd', 0)),
        'total_investors': float(data.get('total_investors', 0)),
        'startup_age': float(data.get('startup_age', 0)),
    }

    if le_category and le_country and 'category_encoded' in feature_names:
        cat = data.get('category_code', 'unknown')
        country = data.get('country_code', 'unknown')

        # Handle unseen labels gracefully
        known_cats = set(le_category.classes_)
        known_countries = set(le_country.classes_)
        cat = cat if cat in known_cats else 'unknown'
        country = country if country in known_countries else 'unknown'

        base_features['category_encoded'] = int(le_category.transform([cat])[0])
        base_features['country_encoded'] = int(le_country.transform([country])[0])

    df = pd.DataFrame([base_features], columns=feature_names)
    return df


# =========================================================
# CORE PREDICTION LOGIC (shared by single + batch)
# =========================================================
def run_prediction(data):
    """Runs the full ensemble prediction pipeline on a single data dict."""
    df_features = build_features(data)

    # 1. Random Forest
    rf_probs = rf_model.predict_proba(df_features)[0]
    rf_success_prob = rf_probs[1] * 100

    # 2. Neural Network
    features_scaled = nn_scaler.transform(df_features)
    nn_probs = nn_model.predict_proba(features_scaled)[0]
    nn_success_prob = nn_probs[1] * 100

    # 3. Ensemble
    avg_success_prob = (rf_success_prob + nn_success_prob) / 2

    # 4. NLP Sentiment
    founder_bio = data.get('founder_bio', '').strip()
    nlp_score = 0
    nlp_sentiment = "Neutral"
    nlp_impact_text = "No bio provided; NLP Engine bypassed."

    if founder_bio:
        sentiment_dict = nlp_analyzer.polarity_scores(founder_bio)
        compound = sentiment_dict['compound']
        nlp_score = compound * 10

        if compound >= 0.05:
            nlp_sentiment = "Positive"
            nlp_impact_text = f"Founder bio reflects strong confidence (+{round(nlp_score, 1)}% boost)."
        elif compound <= -0.05:
            nlp_sentiment = "Negative"
            nlp_impact_text = f"Founder bio contains uncertainty ({round(nlp_score, 1)}% penalty)."
        else:
            nlp_sentiment = "Neutral"
            nlp_impact_text = "Founder bio is neutral. Minor impact on prediction."

    # 5. Final confidence
    final_confidence = max(0, min(100, avg_success_prob + nlp_score))
    ensemble_status = "Success" if final_confidence >= 50 else "Failure"

    # 6. SHAP Explanation
    explanation = "N/A"
    shap_values_list = []
    if explainer:
        try:
            sv = explainer.shap_values(df_features)
            feature_shap = sv[0][0] if isinstance(sv, list) else sv[0]
            top_idx = abs(feature_shap).argmax()
            top_feature = df_features.columns[top_idx]
            impact = "positively" if feature_shap[top_idx] > 0 else "negatively"
            explanation = f"'{top_feature}' {impact} impacted the prediction most."
            # Build SHAP chart data
            shap_values_list = [
                {"feature": col, "value": round(float(feature_shap[i]), 4)}
                for i, col in enumerate(df_features.columns)
            ]
        except Exception as e:
            explanation = f"SHAP unavailable: {e}"

    return {
        "prediction": ensemble_status,
        "rf_confidence_score_percent": round(rf_success_prob, 2),
        "nn_confidence_score_percent": round(nn_success_prob, 2),
        "ensemble_confidence_percent": round(final_confidence if final_confidence >= 50 else 100 - final_confidence, 2),
        "raw_success_probability": round(final_confidence, 2),
        "nlp_sentiment": nlp_sentiment,
        "nlp_impact_text": nlp_impact_text,
        "xai_explanation": explanation,
        "shap_values": shap_values_list,
        "input_used": data
    }


# =========================================================
# API ENDPOINTS
# =========================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint — tells the frontend if the backend is alive and models are loaded."""
    model_loaded = rf_model is not None and nn_model is not None
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    return jsonify({
        "status": "online",
        "models_loaded": model_loaded,
        "shap_available": shap_available,
        "features_count": len(feature_names),
        "features": feature_names,
        "model_metadata": metadata,
        "retraining_in_progress": retraining_in_progress
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """Single startup prediction endpoint."""
    if not rf_model or not nn_model:
        return jsonify({"error": "ML Models are not loaded. Train Phase 2 and Phase 5 first."}), 500

    data = request.json
    cleaned_data, errors = validate_inputs(data)
    if errors:
        return jsonify({"error": "Input validation failed", "details": errors}), 400

    try:
        result = run_prediction(cleaned_data)
        log_prediction(cleaned_data, result)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction — accepts a list of startups and returns predictions for all."""
    if not rf_model or not nn_model:
        return jsonify({"error": "ML Models are not loaded."}), 500

    payload = request.json
    if not isinstance(payload, list):
        return jsonify({"error": "Request body must be a JSON array of startup objects."}), 400

    results = []
    for i, item in enumerate(payload):
        cleaned, errors = validate_inputs(item)
        if errors:
            results.append({"index": i, "error": errors})
            continue
        try:
            result = run_prediction(cleaned)
            log_prediction(cleaned, result)
            results.append({"index": i, **result})
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify({"count": len(results), "results": results}), 200


@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns the last N predictions from the history log."""
    limit = request.args.get('limit', 20, type=int)
    limit = min(limit, 200)  # Cap at 200

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            f"SELECT * FROM prediction_history ORDER BY id DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return jsonify({"count": len(df), "history": df.to_dict(orient='records')}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Triggers model retraining in a background thread."""
    global retraining_in_progress

    if retraining_in_progress:
        return jsonify({"message": "Retraining already in progress. Please wait."}), 409

    def retrain_task():
        global retraining_in_progress
        retraining_in_progress = True
        print("[RETRAIN] Starting model retraining...")
        try:
            subprocess.run(["python", "train_advanced_model.py"], check=True)
            subprocess.run(["python", "train_nn_model.py"], check=True)
            load_models()  # Hot-reload updated models into memory
            print("[RETRAIN] Done! Models updated in memory.")
        except subprocess.CalledProcessError as e:
            print(f"[RETRAIN ERROR] {e}")
        finally:
            retraining_in_progress = False

    thread = threading.Thread(target=retrain_task, daemon=True)
    thread.start()

    return jsonify({"message": "Model retraining started in background. Check /api/health for status."}), 202


@app.route('/api/model/versions', methods=['GET'])
def list_model_versions():
    """Lists all saved model versions."""
    versions_dir = os.path.join('model', 'versions')
    if not os.path.exists(versions_dir):
        return jsonify({"versions": []}), 200

    files = sorted(os.listdir(versions_dir), reverse=True)
    return jsonify({"versions": files}), 200


# =========================================================
# LEGACY ROUTES (kept for compatibility)
# =========================================================
@app.route('/PredictAction', methods=['POST'])
def old_predict_action():
    return "This route is deprecated. Use /api/predict with JSON.", 410

@app.route('/index', methods=['GET'])
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', data='')

@app.route('/PredictSuccess', methods=['GET'])
def PredictSuccess():
    return render_template('PredictSuccess.html', data='')

@app.route('/Logout')
def Logout():
    return render_template('index.html', data='')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
