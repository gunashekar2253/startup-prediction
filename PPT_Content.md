# PPT Content — Startup Success Prediction System

---

## SLIDE 1: Title Slide

**Title:** AI-Powered Startup Success Prediction System with Ensemble Machine Learning, Explainable AI, and MLOps Automation

**Subtitle:** B.Tech Final Year Project — Department of Computer Science & Engineering

**Team Members:** [Your Names & Roll Numbers]

**Guide:** [Guide Name, Designation]

**College:** [College Name, Affiliation, Year 2021-2025]

---

## SLIDE 2: Abstract

- 90% of startups fail within their first few years causing massive financial losses
- This project builds an AI-powered system that predicts startup success or failure using ensemble machine learning
- Two models used: Random Forest (86.86% accuracy) + Neural Network (79.05% accuracy)
- Trained on 66,000+ real-world startups from Crunchbase dataset
- SMOTE applied to handle class imbalance (66K → 120K balanced samples)
- NLP sentiment analysis (VADER) on founder descriptions adjusts confidence by ±10%
- SHAP explainability shows which feature impacted each prediction most
- Flask REST API backend with 6 endpoints
- React + Vite interactive dashboard with 3 tabs
- MLOps pipeline auto-fetches data from 3 live APIs and retrains every 12 hours
- Keywords: Ensemble ML, Random Forest, Neural Network, NLP, SHAP, MLOps, SMOTE

---

## SLIDE 3: Introduction — Problem Statement

**The Problem:**
- ~90% of startups fail, costing investors and founders billions annually
- Current evaluation methods are subjective — investors rely on gut feeling and personal bias
- No transparency — existing ML tools give predictions without explaining WHY
- Models are trained once and never updated — they go stale as markets change
- Only numerical data is used — qualitative signals like founder confidence are completely ignored
- Datasets are imbalanced — models learn to always predict "Success" and appear accurate but are useless

**The Solution:**
- Ensemble of 2 fundamentally different ML models for robust predictions
- SHAP explainability for every prediction — users see which factor mattered most
- NLP sentiment analysis adds qualitative founder assessment
- SMOTE balances the dataset so the model learns from both successes AND failures
- Automated MLOps pipeline retrains models every 12 hours with live API data
- Modern React dashboard makes it accessible to non-technical users

---

## SLIDE 4: Literature Survey

| Paper | Author(s) | Method | Limitation Our System Solves |
|-------|-----------|--------|------------------------------|
| Predicting Startup Outcomes | Krishna et al. (2016) | Random Forest + Gradient Boosting on Crunchbase data | Static model, no continuous learning, no qualitative features |
| Company Acquisition Prediction | Xiang et al. (2012) | SVM + Logistic Regression | No explainability, poor on imbalanced data |
| Web-based Startup Prediction | Sharchilev et al. (2018) | Gradient Boosting + web traffic features | Data availability issues, privacy concerns |
| Deep Learning for Funding Prediction | Ang et al. (2021) | MLP Neural Network with dropout | Black-box predictions, no explanations |
| Explainable AI for Business | Bhat & Reddy (2022) | SHAP + Gradient Boosting | High computational overhead, single model only |

**Our Contribution:** We combine ensemble ML + NLP + SHAP + automated MLOps — no existing system does all four together.

---

## SLIDE 5: Existing System & Limitations

**Existing System:**
- Manual evaluation by VCs and angel investors based on subjective criteria
- Basic ML models using 3-4 numerical features only
- Single model predictions with no explanation
- Static models that are never retrained after deployment
- No integration of qualitative data like founder vision

**Limitations:**
1. Subjectivity & Bias — Different investors evaluate the same startup differently
2. Limited Features — Only funding amount and round count; ignores industry and geography
3. Class Imbalance — 85%+ startups are "operating" → model trivially predicts "Success"
4. Black-Box — User gets a prediction but zero explanation of reasoning
5. No Continuous Learning — Model trained once, never updated as markets shift
6. No NLP — Founder confidence and vision completely ignored

---

## SLIDE 6: Proposed System & Advantages

**Proposed System:**
- Dual-model ensemble: Random Forest (150 trees) + Neural Network (128→64→32 layers)
- 6 engineered features: funding_rounds, raised_usd, investors, startup_age, category_encoded, country_encoded
- SMOTE oversampling: 66,368 → 120,260 balanced samples
- VADER NLP: Analyzes founder bio, adjusts confidence ±10%
- SHAP TreeExplainer: Per-prediction feature importance
- Flask API: 6 endpoints (predict, batch, history, health, retrain, versions)
- React Dashboard: 3 tabs — Predict, History, Model Management
- MLOps Pipeline: Clearbit + NewsAPI + Yahoo Finance → auto-retrain every 12 hours
- Model Versioning: Timestamped backups before every retrain

**Advantages:**
1. 86.86% accuracy with balanced precision/recall
2. Every prediction is explainable via SHAP
3. Qualitative + quantitative analysis combined
4. Continuous learning prevents model drift
5. Batch prediction for portfolio screening
6. One-click launcher (RunProject.bat)
7. Prediction audit trail in SQLite

---

## SLIDE 7: Functional Requirements

1. Accept 7 inputs: funding rounds, raised USD, investors, age, category, country, founder bio
2. Return Success/Failure prediction with ensemble confidence percentage
3. Combine Random Forest + Neural Network predictions into single ensemble score
4. Analyze founder bio text using VADER NLP and adjust confidence ±10%
5. Generate SHAP feature importance values for every individual prediction
6. Support batch prediction — predict multiple startups in one API call
7. Log every prediction with inputs, outputs, timestamps to SQLite database
8. Support on-demand model retraining via API without server restart
9. Expose health endpoint showing model status, accuracy, feature count
10. Provide interactive dashboard with prediction form, history and model management

---

## SLIDE 8: Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| Accuracy | ≥ 80% on test set with balanced precision and recall (achieved 86.86%) |
| Response Time | < 2 seconds for single prediction |
| Batch Performance | < 10 seconds for 50 startups |
| Security | API keys in .env file, never in source code; .gitignore protects sensitive files |
| Availability | MLOps pipeline runs 24/7 with 12-hour retrain cycles |
| Usability | Dashboard usable by non-technical users with zero training |
| Data Volume | Handles 66,000+ startup records and 114,000+ funding round records |
| Portability | Runs on Windows/Linux, no GPU required, standard CPU hardware |

---

## SLIDE 9: System Architecture

**3-Tier Architecture + MLOps Layer:**

**Layer 1 — Presentation (React + Vite, Port 5173):**
- Predict Tab: Input form → Confidence Gauge → SHAP Chart → NLP Badge
- History Tab: Past predictions with Success/Failure badges
- Model Tab: Accuracy, features, retrain button
- Health Badge: Auto-polls backend every 10 seconds

**Layer 2 — Application (Flask API, Port 5000):**
- 6 REST endpoints: /api/health, /api/predict, /api/predict/batch, /api/history, /api/retrain, /api/model/versions
- Ensemble prediction engine (RF + NN + NLP + SHAP)
- Input validation, prediction history logging

**Layer 3 — Data (SQLite):**
- startups table: 66,368 rows
- funding_rounds table: 114,984 rows
- founders table: 229,968 rows
- prediction_history table: dynamic
- mlops_log table: dynamic

**Layer 4 — MLOps (Background Process):**
- Clearbit API → company industry data
- NewsAPI → funding news articles
- Yahoo Finance → IPO/stock status
- Auto-retrain every 12 hours with model versioning

---

## SLIDE 10: UML — Use Case Diagram

**Actors:**
- User (Investor/Entrepreneur)
- Flask API System
- ML Models (RF + NN)
- MLOps Pipeline (automated)

**Use Cases:**
1. Enter Startup Metrics
2. Validate Input
3. Build Feature Vector (6 features + LabelEncoder)
4. Run Random Forest Prediction
5. Run Neural Network Prediction
6. Compute Ensemble Average
7. Analyze Founder Bio (VADER NLP)
8. Generate SHAP Explanation
9. View Prediction Result (Gauge + Chart)
10. Log Prediction to Database
11. View Prediction History
12. Trigger Model Retrain
13. Fetch Live API Data (Clearbit, NewsAPI, Yahoo Finance)
14. Auto-Retrain Models
15. Version Old Models

[INSERT USE CASE DIAGRAM IMAGE]

---

## SLIDE 11: UML — Class Diagram

**Classes:**

1. **ETLPipeline** (etl_pipeline.py)
   - Attributes: DB_PATH, base_dir
   - Methods: extract_and_load(conn)

2. **RFModelTrainer** (train_advanced_model.py)
   - Attributes: DB_PATH, MODEL_DIR
   - Methods: get_training_data(), feature_engineering(df), train_model(X, y), generate_shap_explanations(), save_artifacts()

3. **NNModelTrainer** (train_nn_model.py)
   - Attributes: DB_PATH, MODEL_DIR
   - Methods: get_training_data(), train_neural_network(X, y)

4. **PredictionEngine** (Main.py)
   - Attributes: rf_model, nn_model, nn_scaler, explainer, le_category, le_country, feature_names
   - Methods: load_models(), validate_inputs(data), build_features(data), run_prediction(data), log_prediction(), init_history_db()

5. **MLOpsPipeline** (mlops_pipeline.py)
   - Attributes: NEWS_API_KEY, CLEARBIT_API_KEY
   - Methods: extract_clearbit_data(), monitor_news_for_funding(), check_yahoo_finance_ipo(), log_api_run_to_db(), save_model_version(), job_continuous_learning()

6. **Dashboard** (App.jsx)
   - Components: HealthBadge, ConfidenceGauge, ShapChart, HistoryPanel
   - State: formData, result, health, history, activeTab

[INSERT CLASS DIAGRAM IMAGE]

---

## SLIDE 12: UML — Sequence Diagram

**Flow: User → Prediction Result**

1. User → React Dashboard: Enters metrics and clicks "Predict Success"
2. React → Flask API: POST /api/predict with JSON body
3. Flask API → validate_inputs(): Checks ranges and types
4. Flask API → build_features(): Constructs 6-feature DataFrame with LabelEncoders
5. Flask API → RF Model: predict_proba() → returns P(success) = 82.45%
6. Flask API → StandardScaler: Scales features for NN
7. Flask API → NN Model: predict_proba() → returns P(success) = 76.30%
8. Flask API: Ensemble average = (82.45 + 76.30) / 2 = 79.375%
9. Flask API → VADER NLP: polarity_scores(founder_bio) → compound = +0.65
10. Flask API: NLP adjustment = 0.65 × 10 = +6.5%
11. Flask API: Final = 79.375 + 6.5 = 85.875% → "Success"
12. Flask API → SHAP TreeExplainer: Computes 6 feature contribution values
13. Flask API → SQLite: INSERT into prediction_history
14. Flask API → React: Returns JSON with prediction, confidence, SHAP, NLP
15. React: Renders gauge, bars, SHAP chart, NLP badge

[INSERT SEQUENCE DIAGRAM IMAGE]

---

## SLIDE 13: UML — Activity Diagram

**Activity Flow:**

START → User Opens Dashboard → Enters Startup Metrics → Clicks Predict
→ [Decision: Valid Input?]
   → NO → Display Error Messages → Return to Form
   → YES → Build 6-Feature Vector
→ Run Random Forest (150 trees) → Get RF Probability
→ Scale Features → Run Neural Network (128→64→32) → Get NN Probability
→ Average RF + NN → Base Confidence
→ [Decision: Founder Bio Provided?]
   → YES → VADER Sentiment Analysis → Adjust ±10%
   → NO → Skip NLP
→ SHAP TreeExplainer → Compute Feature Importances
→ [Decision: Confidence ≥ 50%?]
   → YES → Classify as "Success"
   → NO → Classify as "Failure"
→ Log to prediction_history Table
→ Display Result (Gauge + SHAP Chart + NLP Badge)
→ END

[INSERT ACTIVITY DIAGRAM IMAGE]

---

## SLIDE 14: UML — Component Diagram

**Components and Dependencies:**

1. **Dataset Component** (CSV files)
   → provides data to → ETL Pipeline

2. **ETL Pipeline** (etl_pipeline.py)
   → loads into → SQLite Database

3. **SQLite Database** (startup_data.db)
   → supplies training data to → RF Trainer + NN Trainer

4. **RF Trainer** (train_advanced_model.py)
   → produces → RF Model (.pkl) + LabelEncoders (.pkl) + SHAP Explainer

5. **NN Trainer** (train_nn_model.py)
   → produces → NN Model (.pkl) + Scaler (.pkl)

6. **Flask API** (Main.py)
   → loads → RF Model, NN Model, Scaler, LabelEncoders, SHAP
   → reads/writes → SQLite (prediction_history)
   → serves → REST API endpoints

7. **React Dashboard** (App.jsx)
   → calls → Flask API via HTTP

8. **MLOps Pipeline** (mlops_pipeline.py)
   → fetches from → Clearbit API, NewsAPI, Yahoo Finance
   → writes to → SQLite (mlops_log)
   → triggers → RF Trainer + NN Trainer

[INSERT COMPONENT DIAGRAM IMAGE]

---

## SLIDE 15: UML — Deployment Diagram

**Nodes:**

1. **Client Machine**
   - Web Browser
   - Accesses: http://localhost:5173 (React Dashboard)

2. **Application Server (localhost)**
   - React Dev Server (Vite, Port 5173) — serves frontend
   - Flask API Server (Port 5000) — serves backend
   - MLOps Pipeline — background process

3. **File System**
   - SQLite Database: data_pipeline/startup_data.db
   - Model Files: model/*.pkl (RF, NN, Scaler, Encoders)
   - Model Versions: model/versions/ (timestamped backups)
   - Configuration: .env (API keys)

4. **External Services (Internet)**
   - Clearbit API (https://company.clearbit.com)
   - NewsAPI (https://newsapi.org)
   - Yahoo Finance (https://query1.finance.yahoo.com)

**Connections:**
- Browser ↔ React (HTTP, port 5173)
- React ↔ Flask API (HTTP REST, port 5000)
- Flask API ↔ SQLite (SQL queries)
- Flask API ↔ Model .pkl files (Joblib load)
- MLOps Pipeline ↔ External APIs (HTTPS)
- MLOps Pipeline ↔ SQLite (SQL insert)

[INSERT DEPLOYMENT DIAGRAM IMAGE]

---

## SLIDE 16: Results — Model Performance

**Random Forest Classifier:**
- Accuracy: 86.86%
- Precision (Class 0 — Failure): 0.90
- Recall (Class 0 — Failure): 0.83
- Precision (Class 1 — Success): 0.84
- Recall (Class 1 — Success): 0.91
- F1-Score (weighted avg): 0.87
- Test samples: 24,052

**Neural Network (MLP):**
- Accuracy: 79.05%
- Architecture: 128 → 64 → 32 neurons
- Early stopping at validation convergence

**Data Processing:**
- Original dataset: 66,368 samples (imbalanced)
- After SMOTE: 120,260 samples (balanced)
- Features used: 6 (funding_rounds, raised_usd, investors, age, category, country)
- Training split: 80% train, 20% test

---

## SLIDE 17: Results — UI Screenshots

**Screenshot 1:** Dashboard home screen with health badge showing "Online | 6 Features | Acc: 86.86%"

**Screenshot 2:** Prediction input form with all 7 fields filled (funding rounds, raised USD, investors, age, category, country, founder bio)

**Screenshot 3:** Prediction result showing:
- Large Success/Failure badge
- SVG circular confidence gauge (e.g., 85.88%)
- Random Forest progress bar (e.g., 82.45%)
- Neural Network progress bar (e.g., 76.30%)
- NLP Sentiment badge: "Positive" with boost text
- SHAP Feature Importance bar chart showing 6 features

**Screenshot 4:** History tab showing list of past predictions with color-coded badges, funding amounts, round counts, timestamps

**Screenshot 5:** Model Management tab showing accuracy, trained date, feature count, SHAP status, retrain button

[INSERT ACTUAL SCREENSHOTS]

---

## SLIDE 18: Conclusion

- Built a complete end-to-end AI prediction platform — not just a model
- Ensemble of RF (86.86%) + NN (79.05%) provides robust predictions
- SHAP makes every prediction transparent and explainable
- VADER NLP adds qualitative founder assessment to quantitative model
- SMOTE solved class imbalance — model detects both success AND failure
- MLOps pipeline automates continuous learning from 3 live APIs
- Flask REST API with 6 production endpoints including batch prediction and history
- React dashboard makes the system accessible to non-technical users
- Model versioning enables safe rollbacks after every retrain
- ETL pipeline processes 66,000+ startups into normalized SQLite database
- One-click RunProject.bat launches the entire system

---

## SLIDE 19: Future Work

1. **XGBoost / LightGBM** — Add gradient boosting models for potentially higher accuracy
2. **Hyperparameter Tuning (Optuna)** — Automated search for best model parameters
3. **Docker Containerization** — Package entire system into containers for one-command deployment
4. **PostgreSQL** — Replace SQLite with production-grade database for concurrency
5. **MLflow Model Registry** — Track experiments, compare versions, manage model lifecycle
6. **CI/CD Pipeline (GitHub Actions)** — Automate testing and deployment on every code push
7. **Model Drift Detection (Evidently AI)** — Detect when model accuracy degrades over time
8. **Kubernetes** — Auto-scaling pods based on user traffic
9. **PDF Report Export** — Generate downloadable prediction reports
10. **Startup Comparison Mode** — Compare 2+ startups side-by-side

---

## SLIDE 20: Bibliography

[1] A. Krishna et al., "Predicting the outcome of startups: less failure, more success," Proc. IEEE ICDMW, 2016, pp. 798-805.

[2] G. Xiang et al., "A supervised approach to predict company acquisition," Proc. AAAI ICWSM, 2012, pp. 607-610.

[3] B. Sharchilev et al., "Web-based startup success prediction," Proc. ACM CIKM, 2018, pp. 2283-2291.

[4] Y. N. Ang et al., "Deep learning for startup funding prediction," Proc. IEEE Big Data, 2021, pp. 4741-4748.

[5] H. S. Bhat and S. R. Reddy, "Explainable AI for business analytics," J. Business Analytics, vol. 5, no. 2, 2022.

[6] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, vol. 12, 2011, pp. 2825-2830.

[7] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," NeurIPS, 2017, pp. 4765-4774.

[8] C. J. Hutto and E. Gilbert, "VADER: Sentiment analysis of social media text," Proc. AAAI ICWSM, 2014, pp. 216-225.

[9] N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," JAIR, vol. 16, 2002, pp. 321-357.

[10] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, 2001, pp. 5-32.

---

## SLIDE 21: Thank You

**Thank You!**

**Project Title:** AI-Powered Startup Success Prediction System

**Technologies:** Python · Flask · React · SQLite · Scikit-learn · SHAP · VADER · SMOTE

**Key Achievement:** 86.86% Accuracy with Full Explainability

**Questions?**
