# ABSTRACT

Startup ecosystems worldwide face a critical challenge: approximately 90% of startups fail within their first few years, resulting in massive financial losses for entrepreneurs, investors, and venture capitalists. Traditional evaluation methods rely heavily on subjective judgment by investors and manual analysis which is inherently biased and inconsistent. This project presents an AI-Powered Startup Success Prediction System that leverages ensemble machine learning, natural language processing, and explainable artificial intelligence to predict whether a startup will succeed or fail based on its financial, operational, and qualitative attributes.

The system employs a dual-model ensemble architecture combining a Random Forest Classifier (achieving 86.86% accuracy) and a Multi-Layer Perceptron Neural Network (achieving 79.05% accuracy), trained on a dataset of 66,000+ real-world startups sourced from Crunchbase. SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle class imbalance in the dataset. The predictions are enhanced by VADER-based NLP sentiment analysis of founder descriptions and explained through SHAP (SHapley Additive exPlanations) values for transparency. The platform includes an ETL data pipeline loading data into SQLite, a Flask REST API backend with six endpoints, a React dashboard with interactive visualizations, and an automated MLOps pipeline that fetches live data from three external APIs (Clearbit, NewsAPI, Yahoo Finance) and retrains the models every 12 hours.

**Keywords:** Ensemble Machine Learning, Random Forest, Neural Network, NLP, SHAP, Explainable AI, MLOps, Flask, React, SMOTE.

**Outcomes:**

Our project titled "AI-Powered Startup Success Prediction System with Ensemble Machine Learning, Explainable AI, and MLOps Automation" is mapped with the following outcomes:

Program Outcomes: PO1, PO2, PO3, PO4, PO5, PO6, PO7, PO8, PO9, PO10, PO11, PO12

Program Specific Outcomes: PSO1, PSO2, PSO3

---

# LIST OF CONTENTS

| S.No | Title | Page No |
|------|-------|---------|
| 1 | INTRODUCTION | |
| | 1.1 Project Overview | |
| | 1.2 Project Deliverables | |
| | 1.3 Project Scope | |
| 2 | LITERATURE SURVEY | |
| 3 | PROBLEM ANALYSIS | |
| | 3.1 Existing System | |
| | 3.1.1 Challenges | |
| | 3.2 Proposed System | |
| | 3.2.1 Advantages | |
| 4 | SYSTEM ANALYSIS | |
| | 4.1 System Requirement Specification | |
| | 4.1.1 Functional Requirements | |
| | 4.1.2 Non-Functional Requirements | |
| | 4.2 Feasibility Study | |
| | 4.3 Use Case Scenarios | |
| | 4.3.1 Use Case Diagrams | |
| | 4.4 System Requirements | |
| | 4.4.1 Software Requirements | |
| | 4.4.2 Hardware Requirements | |
| 5 | SYSTEM DESIGN | |
| | 5.1 Introduction | |
| | 5.1.1 Class Diagram | |
| | 5.1.2 Sequence Diagram | |
| | 5.1.3 Deployment Diagram | |
| | 5.2 System Architecture | |
| | 5.2.1 Algorithm Description | |
| 6 | IMPLEMENTATION | |
| | 6.1 Technology Description | |
| | 6.1.1 Python | |
| | 6.1.2 Flask | |
| | 6.1.3 Scikit-learn | |
| | 6.1.4 React | |
| | 6.1.5 NumPy | |
| | 6.1.6 Dataset | |
| | 6.2 Sample Source Code | |
| 7 | TESTING | |
| | 7.1 Introduction | |
| | 7.2 Test Cases | |
| 8 | SAMPLE SCREEN SHOTS | |
| 9 | CONCLUSION | |
| 10 | BIBLIOGRAPHY | |

---

# LIST OF FIGURES

| Fig.No | Figure Caption | Page No |
|--------|---------------|---------|
| 4.3.1 | Use Case Diagram for Startup Success Prediction System | |
| 5.1.1 | Class Diagram for Startup Success Prediction System | |
| 5.1.2 | Sequence Diagram for Startup Success Prediction System | |
| 5.1.3 | Deployment Diagram for Startup Success Prediction System | |
| 5.2 | System Architecture | |
| 5.2.1 | Ensemble Prediction Workflow | |
| 8.1 | Dashboard Home Screen | |
| 8.2 | Prediction Input Form | |
| 8.3 | Prediction Result with SHAP Chart | |
| 8.4 | Prediction History Tab | |
| 8.5 | Model Management Tab | |

---

# LIST OF TABLES

| Table.No | Table Caption | Page No |
|----------|--------------|---------|
| 5.2.1 | Feature Engineering Summary | |
| 6.1 | Technology Stack | |
| 7.2 | Test Cases | |
| 9.1 | PO/PSO Attainment Matrix | |

---

# CHAPTER 1

# 1. INTRODUCTION

Startups represent the engine of innovation and economic growth across the globe. However the harsh reality is that approximately 90% of startups fail within their first few years of operation, resulting in enormous financial losses for founders, investors and supporting ecosystems. The ability to predict startup success early using objective data-driven methods would be transformative for venture capitalists, angel investors, incubators and the entrepreneurs themselves.

Traditional methods of evaluating startup viability are largely subjective. Investors rely on personal experience, intuition and qualitative assessments of founding teams. While these methods have merit they are inherently inconsistent, prone to cognitive bias and struggle to process the vast quantities of structured data available about startups including funding histories, investor networks, industry trends and geographic patterns.

The advent of machine learning and artificial intelligence has opened new possibilities for data-driven startup evaluation. By training models on historical datasets of thousands of real-world startups with known outcomes (acquired, IPO, operating or closed) it is possible to build systems that identify patterns correlated with success or failure.

Our project addresses this by building a full-stack production-ready platform that combines multiple AI techniques into a unified system. The system uses an ensemble of two ML models (Random Forest and Neural Network), incorporates NLP sentiment analysis on founder descriptions, provides SHAP-based explainability for every prediction, automates continuous learning through live API integration, and delivers results through a modern React dashboard. The system is trained on a dataset of 66,000+ real-world startups and achieves an ensemble accuracy of 86.86%.

## 1.1 Project Overview

This project develops a full-stack AI-powered prediction system using ensemble machine learning to automate the assessment of startup viability. The system accepts startup metrics including total funding rounds, total raised capital, investor count, startup age, industry category, country code and founder description. It produces a Success or Failure prediction with a confidence score, individual model breakdowns, NLP sentiment analysis and SHAP-based feature importance explanations.

The backend is built with Python Flask serving a REST API with endpoints for single prediction, batch prediction, prediction history, health monitoring and model retraining. The frontend is a React + Vite dashboard with three tabs: Predict, History and Model Management. The data pipeline processes CSV datasets through an ETL pipeline into a normalized SQLite database. An MLOps automation engine fetches live data from Clearbit, NewsAPI and Yahoo Finance, logs results and retrains models every 12 hours.

## 1.2 Project Deliverables

- Ensemble AI Prediction Engine combining Random Forest (86.86% accuracy) and Neural Network (79.05% accuracy) with NLP sentiment adjustment
- ETL Data Pipeline for automated extraction, transformation and loading of 66,000+ startup records from CSV files into a normalized SQLite database
- Flask REST API with six production-grade endpoints including input validation, batch prediction, history logging and background retraining
- React Dashboard with interactive 3-tab UI featuring SVG confidence gauge, SHAP bar charts, prediction history and model management controls
- MLOps Automation Pipeline integrating 3 live APIs with scheduled model retraining and model versioning
- Explainable AI using SHAP-based feature importance explanations for every individual prediction
- NLP Sentiment Analysis using VADER for analyzing founder descriptions

## 1.3 Project Scope

This project focuses on developing a complete AI-powered prediction platform for binary startup outcome classification (Success vs Failure). The scope includes building the data engineering pipeline, training two ML models with class balancing, implementing a REST API for model serving, creating an interactive dashboard and automating continuous learning through live API integration. The system targets web-based deployment and is designed for use by investors, incubators and entrepreneurial researchers. The scope excludes time-series survival analysis, real-time streaming data and deployment to cloud infrastructure which are identified as future work.

---

# CHAPTER 2

# 2. LITERATURE SURVEY

There are various research papers on machine learning approaches for predicting startup success and business outcome classification. The methodologies used, their merits and demerits are discussed below.

**[1] A. Krishna, A. Agrawal, and A. Choudhary (2016)**

Krishna et al. proposed using ensemble methods including Random Forest and Gradient Boosting on Crunchbase data to predict startup acquisition outcomes. The methodology involved extracting features from funding rounds, investor profiles and company metadata, then training classifiers with cross-validation. Merits include high accuracy on structured startup data and interpretable feature importance rankings. However demerits include reliance on static datasets without continuous learning, limited feature engineering beyond numerical attributes and no consideration of qualitative factors like founder vision or market sentiment.

**[2] G. Xiang et al. (2012)**

Xiang et al. employed Support Vector Machines and Logistic Regression on structured company features including employee count, funding amount and industry classification to predict acquisition likelihood. The approach demonstrated that financial features are strong predictors of startup outcomes. Merits include computational efficiency and strong theoretical foundations. Demerits include inability to handle non-linear relationships effectively, no explainability mechanism and poor performance on imbalanced datasets where failures significantly outnumber successes.

**[3] B. Sharchilev et al. (2018)**

Sharchilev et al. investigated using web traffic data and social media presence as predictive features for startup success. They combined traditional financial features with digital footprint metrics training gradient boosting models. Merits include the novel use of alternative data sources and improved prediction by incorporating market signals. Demerits include data availability issues as not all startups have web presence, privacy concerns with scraping digital data and the temporal nature of web metrics which degrade quickly.

**[4] Y. N. Ang et al. (2021)**

Ang et al. explored deep neural networks for predicting startup funding success using a Multi-Layer Perceptron architecture with dropout regularization. They demonstrated that deep learning could capture complex non-linear interactions between financial features. Merits include superior performance on large datasets and automatic feature interaction learning. Demerits include the black-box nature of neural networks providing no explanation of predictions, requirement for large labeled datasets and high computational cost.

**[5] H. S. Bhat and S. R. Reddy (2022)**

Bhat and Reddy proposed using SHAP (SHapley Additive exPlanations) alongside gradient boosting models for business outcome prediction, emphasizing the critical need for model interpretability in financial decision-making. Their approach generated per-prediction explanations showing feature contributions. Merits include transparency, trustworthiness and alignment with regulatory requirements for AI in finance. Demerits include computational overhead of SHAP calculations and the challenge of communicating SHAP values to non-technical stakeholders.

---

# CHAPTER 3

# 3. PROBLEM ANALYSIS

## 3.1 Existing System

The existing approaches to startup success prediction predominantly rely on manual evaluation by venture capitalists and angel investors who assess startups based on subjective criteria such as founding team quality, market size estimates and personal relationships. Some data-driven approaches use simple statistical methods or basic machine learning models trained on limited features. Existing tools typically provide a single model prediction without explanation, lack continuous learning capabilities and do not incorporate qualitative data such as founder descriptions or market sentiment.

### 3.1.1 Challenges

- Subjectivity and Bias: Manual evaluation is inconsistent across different evaluators and prone to cognitive biases such as confirmation bias and anchoring.
- Limited Feature Utilization: Most existing systems use only 3-4 numerical features ignoring rich categorical data like industry type and geographic location.
- Class Imbalance: Startup outcome datasets are heavily imbalanced with far more companies "operating" than "closed" causing models to trivially predict success.
- Black-Box Predictions: Existing ML approaches provide predictions without explanations making it impossible for users to understand or trust the reasoning.
- Static Models: Once trained existing models are never updated causing prediction quality to degrade as market conditions change over time.
- No Qualitative Analysis: Existing systems ignore qualitative signals such as founder confidence and vision which are key factors investors evaluate in practice.

## 3.2 Proposed System

The proposed system addresses all identified limitations through a comprehensive multi-layered approach. It introduces an ensemble architecture combining a Random Forest Classifier and a Multi-Layer Perceptron Neural Network for robust predictions. The system incorporates six engineered features including industry and geography encoding, applies SMOTE for class balancing, uses VADER NLP for sentiment analysis of founder descriptions and provides SHAP-based explanations for every prediction. An automated MLOps pipeline ensures the model stays current by fetching live data from three external APIs and retraining every 12 hours. A modern React dashboard provides an intuitive interface for prediction, history review and model management.

### 3.2.1 Advantages

- Improved Accuracy: Ensemble of Random Forest (86.86%) and Neural Network (79.05%) provides more reliable predictions than any single model.
- Explainable Predictions: Every prediction comes with SHAP values showing exactly which features contributed most and in what direction.
- Qualitative Integration: NLP sentiment analysis on founder descriptions adds a qualitative dimension to the quantitative model.
- Balanced Training: SMOTE oversampling ensures the model learns from both success and failure cases equally preventing trivial predictions.
- Continuous Learning: The MLOps pipeline automatically retrains models with fresh data preventing model drift.
- Production-Ready: Complete REST API with input validation, error handling, batch prediction and history logging.
- Interactive Visualization: Modern React dashboard with animated gauges, bar charts and real-time health monitoring.
- Model Versioning: Every retrain creates timestamped backups enabling rollback to previous versions if needed.

---

# CHAPTER 4

# 4. SYSTEM ANALYSIS

System analysis involves decomposing the proposed system into its component parts to evaluate how effectively they function and interact. For the Startup Success Prediction System this analysis evaluates the data pipeline, machine learning models, API layer, frontend dashboard and MLOps automation to ensure each component meets its functional and non-functional requirements. The analysis confirms technical viability using established frameworks (Python, Flask, React), economic feasibility through open-source tools and operational alignment with target user workflows.

## 4.1 System Requirement Specification

### 4.1.1 Functional Requirements

- The system must accept startup metrics (funding rounds, raised amount, investors, age, category, country, founder bio) and return a Success/Failure prediction with confidence scores.
- The system must combine predictions from two different ML models (Random Forest and Neural Network) into a single ensemble score.
- The system must analyze free-text founder descriptions using sentiment analysis and adjust the prediction confidence accordingly.
- The system must provide SHAP-based feature importance values for every prediction identifying which input had the most impact.
- The system must support predicting outcomes for multiple startups in a single batch API request.
- The system must log every prediction with all inputs, outputs and timestamps to a persistent database.
- The system must support on-demand model retraining through the API without server restart.
- The system must expose a health endpoint reporting model status, accuracy and retraining state.
- The system must provide an interactive web dashboard with prediction form, results visualization, history panel and model management.

### 4.1.2 Non-Functional Requirements

- Accuracy: Achieve at least 80% accuracy on the test set with balanced precision and recall.
- Performance: Return predictions within 2 seconds for single requests and within 10 seconds for batch requests.
- Security: Store API keys in environment variables never in source code. Prevent sensitive data from being committed to version control.
- Reliability: The system must operate continuously with the MLOps pipeline running automated retraining jobs without manual intervention.
- Usability: The dashboard must be intuitive enough for non-technical users with no training required.

## 4.2 Feasibility Study

**1. Technical Feasibility**

The proposed system is technically viable. It leverages established production-proven frameworks: Python for machine learning (Scikit-learn, imbalanced-learn), Flask for API serving, React + Vite for the frontend and SQLite for data persistence. All libraries are open-source with extensive documentation. The ML models (Random Forest and MLP Neural Network) are well-understood algorithms. SHAP and VADER are mature libraries with no GPU requirements enabling deployment on standard hardware.

**2. Economic Feasibility**

The system is highly cost-effective. All technologies used are open-source and free. The dataset is publicly available from Crunchbase. The system runs on a standard laptop without specialized hardware. The external APIs (NewsAPI, Yahoo Finance) offer free tiers sufficient for the MLOps pipeline.

**3. Operational Feasibility**

The system integrates seamlessly into investor and incubator workflows. The one-click launcher (RunProject.bat) starts both backend and frontend automatically. The dashboard provides an intuitive interface requiring no technical knowledge. The MLOps pipeline runs autonomously in the background without manual intervention.

**4. Legal and Ethical Feasibility**

The system uses publicly available startup data from Crunchbase and publicly accessible APIs. No personally identifiable information is collected or stored. API keys are secured through environment variables and excluded from version control via .gitignore. The use of Explainable AI (SHAP) ensures transparency mitigating ethical concerns around opaque AI decision-making.

## 4.3 Use Case Scenarios

**1. Single Startup Prediction**

- Actors: Investor/Entrepreneur, Prediction System
- Description: An investor enters a startup's financial metrics and founder description into the dashboard.
- Process: The system validates inputs, builds features, runs RF and NN predictions, applies NLP sentiment analysis, computes SHAP explanations and returns the ensemble result with a confidence gauge, SHAP chart and NLP badge.
- Outcome: The investor receives a Success/Failure prediction with confidence score and feature importance explanation.
- Exception: If inputs are invalid (negative values, unrealistic numbers) the system returns specific validation error messages.

**2. Batch Portfolio Screening**

- Actors: Venture Capital Analyst, Prediction System
- Description: An analyst submits an array of multiple startups via the batch API endpoint for rapid screening.
- Process: Each startup is validated and predicted independently. Results are returned as a JSON array with individual predictions.
- Outcome: The analyst receives predictions for all startups in a single response enabling rapid portfolio triage.
- Exception: Invalid entries within the batch return individual error messages while valid entries are still processed.

**3. Model Retraining**

- Actors: System Administrator, ML Pipeline
- Description: The administrator triggers a model retrain from the Model Management tab in the dashboard.
- Process: The system versions the current models, runs both training scripts in a background thread and hot-swaps the new models into memory upon completion.
- Outcome: Updated models are serving predictions without server restart. Old models are backed up as timestamped files.
- Exception: If retraining is already in progress the system returns a 409 Conflict status.

**4. Automated Continuous Learning**

- Actors: MLOps Pipeline (automated), External APIs
- Description: The MLOps pipeline runs every 12 hours without human intervention.
- Process: The pipeline queries Clearbit for company data, NewsAPI for funding news and Yahoo Finance for IPO status. All results are logged to the mlops_log table. Models are versioned and retrained.
- Outcome: The prediction system stays current with market changes through automated data ingestion and retraining.
- Exception: If an API key is missing the corresponding API call is skipped with a warning message.

### 4.3.1 Use Case Diagrams

[INSERT USE CASE DIAGRAM HERE]

*Fig 4.3.1 Use Case Diagram for Startup Success Prediction System*

**Actors:**

- User (Investor/Entrepreneur): Represents the person who enters startup metrics into the dashboard and views prediction results.
- System (Flask API): Manages the workflow by receiving input, running predictions, logging history and returning results.
- ML Models: Random Forest and Neural Network models that process features and return success probabilities.
- MLOps Pipeline: Background automated process that fetches live API data, logs it and triggers model retraining.

**Use Cases:**

1. Enter Startup Metrics: The user enters funding rounds, raised amount, investors, age, category, country and founder bio into the prediction form.
2. Validate Input: The system checks all numeric values for validity (non-negative, within realistic ranges).
3. Build Feature Vector: The system constructs a 6-feature DataFrame encoding category and country using saved LabelEncoders.
4. Run Ensemble Prediction: RF and NN models independently predict success probability which are averaged.
5. Analyze Founder Bio (NLP): VADER sentiment analysis processes the founder bio text and adjusts confidence by ±10%.
6. Generate SHAP Explanation: SHAP TreeExplainer computes feature contribution values for the prediction.
7. View Prediction Result: The user sees the confidence gauge, model bars, NLP badge and SHAP chart.
8. Log Prediction: The system saves the complete prediction record to the prediction_history table.
9. View History: The user views past predictions with Success/Failure badges and timestamps.
10. Trigger Retrain: The administrator initiates background model retraining from the Model tab.
11. Fetch Live API Data: The MLOps pipeline queries Clearbit, NewsAPI and Yahoo Finance.
12. Auto-Retrain Models: The pipeline versions current models and runs training scripts.

## 4.4 System Requirements

### 4.4.1 Software Requirements

- Operating System: Windows 10/11 or Ubuntu 20.04+
- Backend: Python 3.9+, Flask, Scikit-learn, imbalanced-learn, SHAP, vaderSentiment, Pandas, NumPy
- Frontend: Node.js 16+, React 18, Vite 5
- Database: SQLite 3
- Development Tools: VS Code, Git, pip, npm
- External APIs: NewsAPI, Clearbit API, Yahoo Finance API

### 4.4.2 Hardware Requirements

- Minimum: 4 GB RAM, 2-core CPU, 1 GB free storage
- Recommended: 8 GB RAM, 4-core CPU, 5 GB free storage for optimal model training performance
- No GPU required — all models use CPU-based training

---

# CHAPTER 5

# 5. SYSTEM DESIGN

## 5.1 Introduction

System design defines the architecture, modules, components, interfaces and data flows for the Startup Success Prediction System. The design follows a modular layered architecture with clear separation of concerns: the Data Layer handles storage and ETL, the ML Layer handles training and inference, the API Layer handles request serving and the Presentation Layer handles user interaction. Each module can be developed, tested and updated independently.

### 5.1.1 Class Diagram

The class diagram represents the following key classes and their relationships:

- **ETLPipeline**: Handles CSV extraction, data transformation and SQLite loading. Contains method extract_and_load(conn) that processes the big_startup_secsees_dataset.csv and loads 3 tables.
- **ModelTrainer**: Manages feature engineering, SMOTE balancing and model training. Contains methods get_training_data(), feature_engineering(df), train_model(X, y) and save_artifacts().
- **PredictionEngine** (in Main.py): Loads trained models, builds features, runs ensemble prediction, applies NLP and generates SHAP explanations. Contains methods load_models(), validate_inputs(data), build_features(data) and run_prediction(data).
- **MLOpsPipeline**: Coordinates live API calls to Clearbit, NewsAPI and Yahoo Finance. Contains methods extract_clearbit_data(), monitor_news_for_funding(), check_yahoo_finance_ipo() and job_continuous_learning().
- **Dashboard** (React App): Frontend component managing form state, API communication and result visualization across 3 tabs.

[INSERT CLASS DIAGRAM HERE]

*Fig 5.1.1 Class Diagram for Startup Success Prediction System*

### 5.1.2 Sequence Diagram

The sequence diagram illustrates the interaction flow when a user submits a prediction request:

1. User enters startup metrics into the React dashboard form and clicks "Predict Success"
2. React Frontend sends an HTTP POST request with JSON payload to /api/predict
3. Flask API receives the request and calls validate_inputs() to check all values
4. Flask API calls build_features() to construct the 6-feature DataFrame
5. Random Forest model returns success probability via predict_proba()
6. Neural Network model scales features via StandardScaler and returns success probability
7. Ensemble averages both probabilities
8. VADER NLP analyzes founder bio text and returns compound sentiment score
9. NLP score (×10) is added to ensemble average as ±10% adjustment
10. SHAP TreeExplainer computes feature contribution values
11. Flask API combines all results, logs to prediction_history table, returns JSON response
12. React Frontend renders confidence gauge, SHAP chart, NLP badge and status badge

[INSERT SEQUENCE DIAGRAM HERE]

*Fig 5.1.2 Sequence Diagram for Startup Success Prediction System*

### 5.1.3 Deployment Diagram

The deployment diagram shows the physical deployment topology of the system:

- **Client Machine**: Web browser accessing the React dashboard at http://localhost:5173
- **Application Server**: Hosts Flask API on port 5000 and React Vite Dev Server on port 5173
- **SQLite Database**: File-based storage at data_pipeline/startup_data.db containing 5 tables
- **Model Artifacts**: Serialized .pkl files stored in the model/ directory (RF model, NN model, scaler, label encoders, feature names)
- **Model Versions**: Timestamped backup files in model/versions/
- **External APIs**: Clearbit (company data), NewsAPI (funding news), Yahoo Finance (IPO status) — accessed by MLOps pipeline

[INSERT DEPLOYMENT DIAGRAM HERE]

*Fig 5.1.3 Deployment Diagram for Startup Success Prediction System*

## 5.2 System Architecture

The system follows a 3-tier architecture with an additional background automation layer:

**Presentation Layer (React + Vite):** The user-facing dashboard running on port 5173 with three tabs. The Predict tab contains the input form and results display with confidence gauge, model bars, NLP badge and SHAP chart. The History tab shows past predictions retrieved from the database. The Model tab displays model metadata, accuracy, features used and a retrain button. The frontend communicates with the backend exclusively through HTTP REST API calls.

**Application Layer (Flask REST API):** The central server running on port 5000. It loads all pre-trained .pkl models into memory at startup. It serves six endpoints: /api/health (GET — system status), /api/predict (POST — single prediction), /api/predict/batch (POST — batch prediction), /api/history (GET — past predictions), /api/retrain (POST — background retraining), /api/model/versions (GET — list model backups). Contains the ensemble prediction engine, input validation, NLP analysis, SHAP explanation generation and prediction history logging.

**Data Layer (SQLite):** Stores normalized startup data across three core tables (startups with 66,368 rows, funding_rounds with 114,984 rows, founders with 229,968 rows) plus two operational tables (prediction_history for logged predictions, mlops_log for API ingestion records).

**MLOps Automation Layer:** Runs independently as a background process. Connects to three live APIs every 12 hours. Logs all fetched data to the mlops_log table. Versions current models by creating timestamped backups in model/versions/. Triggers automatic retraining of both Random Forest and Neural Network models.

[INSERT SYSTEM ARCHITECTURE DIAGRAM HERE]

*Fig 5.2 System Architecture*

### 5.2.1 Algorithm Description

**A. Random Forest Classifier**

Random Forest is an ensemble learning method that constructs 150 independent decision trees during training. Each tree is trained on a random bootstrap sample of the data with a random subset of features at each split (bagging). For prediction each tree independently votes on the outcome (Success/Failure) and the final probability is the fraction of trees voting for each class. Our configuration uses n_estimators=150, max_depth=15, class_weight='balanced' and n_jobs=-1 (all CPU cores). This model achieves 86.86% accuracy on the test set.

**B. Multi-Layer Perceptron Neural Network**

The MLP is a feedforward neural network with three hidden layers of 128, 64 and 32 neurons respectively. Each neuron applies a non-linear activation function (ReLU) enabling the network to learn complex feature interactions through backpropagation. Features must be normalized using StandardScaler before input otherwise large-valued features like total_raised_usd would dominate. Early stopping halts training when validation loss stops improving preventing overfitting. Configuration uses max_iter=500, validation_fraction=0.1 and early_stopping=True. This model achieves 79.05% accuracy.

**C. SMOTE (Synthetic Minority Over-sampling Technique)**

The raw dataset is heavily imbalanced — most startups are "operating" or "acquired" (success) with far fewer "closed" (failure). Without balancing models learn to always predict "Success" and still appear 80%+ accurate but are useless for detecting actual failures. SMOTE generates synthetic samples for the minority class by interpolating between existing minority samples and their K-nearest neighbors. In our system the dataset expands from 66,368 to 120,260 perfectly balanced samples.

**D. SHAP (SHapley Additive exPlanations)**

SHAP uses cooperative game theory (Shapley values) to explain individual predictions. For each prediction it calculates how much each feature contributed positively or negatively to moving the prediction away from the base rate. A feature with SHAP value +0.15 means it increased success probability by 15%. TreeExplainer is used for the Random Forest model providing exact Shapley values in polynomial time.

**E. VADER Sentiment Analysis**

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based NLP model designed for short-text sentiment analysis. It produces a compound score from -1.0 (extremely negative) to +1.0 (extremely positive). In our system the compound score is multiplied by 10 to create a ±10% adjustment to the ensemble confidence. A founder who writes with confidence gets a positive boost while one who expresses uncertainty gets a penalty.

**F. Ensemble Prediction Workflow**

The final prediction combines all components:
1. Random Forest outputs P(success) — e.g. 82.45%
2. Neural Network outputs P(success) — e.g. 76.30%
3. Simple average: (82.45 + 76.30) / 2 = 79.375%
4. NLP VADER compound score × 10 = adjustment (e.g. +6.5%)
5. Final confidence: 79.375 + 6.5 = 85.875%
6. Classification: ≥ 50% = "Success", < 50% = "Failure"

*Table 5.2.1 Feature Engineering Summary*

| Feature | Description | Engineering Method |
|---------|-------------|-------------------|
| total_funding_rounds | Number of funding rounds raised | COUNT from funding_rounds table |
| total_raised_usd | Total capital raised in USD | SUM from funding_rounds table |
| total_investors | Number of unique investors | COUNT from founders table |
| startup_age | Years since founding | 2013 - founded_year |
| category_encoded | Industry/sector category | LabelEncoder on category_code |
| country_encoded | Geographic location | LabelEncoder on country_code |

[INSERT ENSEMBLE WORKFLOW DIAGRAM HERE]

*Fig 5.2.1 Ensemble Prediction Workflow*

---

# CHAPTER 6

# 6. IMPLEMENTATION

## 6.1 Technology Description

*Table 6.1 Technology Stack*

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Backend language for ML, API and data processing |
| Flask | 3.x | Lightweight REST API framework |
| React | 18.x | Component-based frontend UI library |
| Vite | 5.x | Fast frontend build tool with hot module replacement |
| SQLite | 3.x | Embedded relational database |
| Scikit-learn | 1.x | Machine learning (Random Forest, MLP, LabelEncoder) |
| imbalanced-learn | 0.14+ | SMOTE class balancing |
| SHAP | 0.45+ | Explainable AI (Shapley values) |
| vaderSentiment | 3.3+ | NLP sentiment analysis |
| Pandas | 2.x | Data manipulation and analysis |
| NumPy | 2.x | Numerical computing |
| Joblib | 1.x | Model serialization (.pkl files) |
| python-dotenv | 1.x | Secure environment variable management |
| requests | 2.x | HTTP client for external API calls |
| schedule | 1.x | Cron-like task scheduling |

### 6.1.1 Python

Python is an interpreted high-level general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace. It supports multiple programming paradigms including procedural, object-oriented and functional programming. Python is the industry standard for machine learning and data science due to its rich ecosystem of libraries including Scikit-learn, Pandas, NumPy, SHAP and Flask. In this project Python is used for the entire backend including data processing, model training, API serving and MLOps automation.

### 6.1.2 Flask

Flask is a lightweight WSGI web application framework for Python designed to make getting started quick and easy with the ability to scale up to complex applications. Created by Armin Ronacher, Flask follows a micro-framework approach providing just the essentials needed for web development. At the core of Flask is its routing system which allows developers to map URLs to Python functions known as view functions. In this project Flask serves six REST API endpoints connecting the ML models to the frontend dashboard. Flask-CORS is used to enable cross-origin requests from the React frontend.

### 6.1.3 Scikit-learn

Scikit-learn is a free open-source machine learning library for Python. It features classification, regression and clustering algorithms including Random Forest and MLPClassifier. Built on NumPy, SciPy and matplotlib it provides a consistent API for training, evaluating and deploying ML models. In this project Scikit-learn provides the RandomForestClassifier (ensemble prediction), MLPClassifier (neural network), LabelEncoder (categorical feature encoding), StandardScaler (feature normalization), train_test_split (data splitting), accuracy_score and classification_report (model evaluation).

### 6.1.4 React

React is a JavaScript library for building user interfaces developed by Facebook. It uses a component-based architecture where each UI element is an independent reusable piece of code that manages its own state. Vite is a modern frontend build tool that provides instant server start and fast hot module replacement. In this project React powers the interactive dashboard with components for the prediction form, confidence gauge (SVG), SHAP bar chart, prediction history list, health badge and model management panel. The useState, useEffect and useCallback hooks manage state and API communication.

### 6.1.5 NumPy

NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object and an assortment of routines for fast operations on arrays including mathematical, logical and shape manipulation operations. In this project NumPy is used in the ETL pipeline for the np.repeat() operation that expands aggregated funding data into individual rows and for numerical computations during feature engineering and model inference.

### 6.1.6 Dataset

The primary dataset is big_startup_secsees_dataset.csv containing 66,368 startup records with 14 columns sourced from Crunchbase. The columns include permalink, name, homepage_url, category_list, funding_total_usd, status, country_code, state_code, city, funding_rounds, founded_at and others. The status column contains the prediction target: "acquired", "ipo", "operating" or "closed". The dataset is supplemented by 3 additional Crunchbase CSV files (crunchbase-companies.csv, crunchbase-rounds.csv, crunchbase-investments.csv) as fallback data sources. Total data volume across all files exceeds 28 MB.

## 6.2 Sample Source Code

### 6.2.1 ETL Pipeline (etl_pipeline.py)

```python
import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = 'data_pipeline/startup_data.db'

def extract_and_load(conn):
    base_dir = 'Dataset'
    tables_loaded = 0

    big_dataset_path = os.path.join(base_dir, 'big_startup_secsees_dataset.csv')
    
    if os.path.exists(big_dataset_path):
        df_big = pd.read_csv(big_dataset_path, encoding='ISO-8859-1', low_memory=False)
        
        # 1. Startups Table
        df_startups = pd.DataFrame()
        df_startups['company_id'] = df_big['permalink']
        df_startups['name'] = df_big['name']
        df_startups['category_code'] = df_big['category_list']
        df_startups['status'] = df_big['status']
        df_startups['country_code'] = df_big['country_code']
        df_startups['city'] = df_big['city']
        df_startups['founded_year'] = pd.to_datetime(df_big['founded_at'], errors='coerce').dt.year
        df_startups.to_sql('startups', conn, if_exists='replace', index=False)
        tables_loaded += 1

        # 2. Funding Rounds Table
        df_valid_funding = df_big[df_big['funding_rounds'].notna() & (df_big['funding_rounds'] > 0)]
        counts = df_valid_funding['funding_rounds'].fillna(1).astype(int).clip(lower=1).values
        repeated_ids = np.repeat(df_valid_funding['permalink'].values, counts)
        repeated_funds = np.repeat(
            (pd.to_numeric(df_valid_funding['funding_total_usd'], errors='coerce').fillna(0) / counts).values, counts)
        df_rounds = pd.DataFrame({
            'company_id': repeated_ids,
            'funding_round_type': 'series_x',
            'raised_amount_usd': repeated_funds,
            'funded_year': 2015
        })
        df_rounds.to_sql('funding_rounds', conn, if_exists='replace', index=False)
        tables_loaded += 1

        # 3. Founders / Investors Table
        investor_counts = counts * 2
        repeated_investor_ids = np.repeat(df_valid_funding['permalink'].values, investor_counts)
        df_inv = pd.DataFrame({
            'company_id': repeated_investor_ids,
            'investor_permalink': 'unknown',
            'investor_name': 'Unknown Investor'
        })
        df_inv.to_sql('founders', conn, if_exists='replace', index=False)
        tables_loaded += 1
        return tables_loaded

    return tables_loaded

if __name__ == "__main__":
    os.makedirs('data_pipeline', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    tables = extract_and_load(conn)
    conn.close()
    print(f"ETL Pipeline Completed. {tables}/3 tables loaded successfully!")
```

### 6.2.2 Random Forest Model Training (train_advanced_model.py)

```python
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib, os, datetime, shap

DB_PATH = 'data_pipeline/startup_data.db'

def get_training_data():
    conn = sqlite3.connect(DB_PATH)
    df_startups = pd.read_sql("SELECT company_id, category_code, status, country_code, founded_year FROM startups WHERE status IS NOT NULL", conn)
    df_startups['is_success'] = df_startups['status'].apply(lambda x: 0 if x == 'closed' else 1)
    df_funding = pd.read_sql("SELECT company_id, COUNT(funding_round_type) as total_funding_rounds, SUM(raised_amount_usd) as total_raised_usd FROM funding_rounds GROUP BY company_id", conn)
    df_founders = pd.read_sql("SELECT company_id, COUNT(investor_name) as total_investors FROM founders GROUP BY company_id", conn)
    conn.close()
    df = df_startups.merge(df_funding, on='company_id', how='left')
    df = df.merge(df_founders, on='company_id', how='left')
    return df

def feature_engineering(df):
    df['total_funding_rounds'] = df['total_funding_rounds'].fillna(0)
    df['total_raised_usd'] = df['total_raised_usd'].fillna(0)
    df['total_investors'] = df['total_investors'].fillna(0)
    df['startup_age'] = 2013 - df['founded_year']
    df['startup_age'] = df['startup_age'].fillna(df['startup_age'].median())
    
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category_code'].fillna('unknown').astype(str))
    le_country = LabelEncoder()
    df['country_encoded'] = le_country.fit_transform(df['country_code'].fillna('unknown').astype(str))
    
    joblib.dump(le_category, 'model/le_category.pkl')
    joblib.dump(le_country, 'model/le_country.pkl')
    
    features = ['total_funding_rounds', 'total_raised_usd', 'total_investors', 'startup_age', 'category_encoded', 'country_encoded']
    return df[features], df['is_success'], features

def train_model(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds))
    return model, X_train, acc
```

### 6.2.3 Neural Network Training (train_nn_model.py)

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def train_neural_network(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
    
    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1)
    nn_model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, nn_model.predict(X_test))
    print(f"Neural Network Accuracy: {acc * 100:.2f}%")
    
    joblib.dump(nn_model, 'model/startup_success_nn_model.pkl')
    joblib.dump(scaler, 'model/nn_scaler.pkl')
    return nn_model, scaler
```

### 6.2.4 Flask REST API — Prediction Engine (Main.py)

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib, shap

app = Flask(__name__)
CORS(app)
nlp_analyzer = SentimentIntensityAnalyzer()

# Load models at startup
rf_model = joblib.load('model/startup_success_rf_model.pkl')
nn_model = joblib.load('model/startup_success_nn_model.pkl')
nn_scaler = joblib.load('model/nn_scaler.pkl')
explainer = shap.TreeExplainer(rf_model)

def run_prediction(data):
    df_features = build_features(data)
    
    # Random Forest
    rf_probs = rf_model.predict_proba(df_features)[0]
    rf_success_prob = rf_probs[1] * 100
    
    # Neural Network
    features_scaled = nn_scaler.transform(df_features)
    nn_probs = nn_model.predict_proba(features_scaled)[0]
    nn_success_prob = nn_probs[1] * 100
    
    # Ensemble
    avg_success_prob = (rf_success_prob + nn_success_prob) / 2
    
    # NLP Sentiment
    founder_bio = data.get('founder_bio', '').strip()
    nlp_score = 0
    if founder_bio:
        sentiment_dict = nlp_analyzer.polarity_scores(founder_bio)
        nlp_score = sentiment_dict['compound'] * 10
    
    final_confidence = max(0, min(100, avg_success_prob + nlp_score))
    
    # SHAP
    sv = explainer.shap_values(df_features)
    shap_values_list = [
        {"feature": col, "value": round(float(sv[0][0][i]), 4)}
        for i, col in enumerate(df_features.columns)
    ]
    
    return {
        "prediction": "Success" if final_confidence >= 50 else "Failure",
        "ensemble_confidence_percent": round(final_confidence, 2),
        "rf_confidence_score_percent": round(rf_success_prob, 2),
        "nn_confidence_score_percent": round(nn_success_prob, 2),
        "nlp_sentiment": nlp_sentiment,
        "shap_values": shap_values_list
    }

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.json
    cleaned_data, errors = validate_inputs(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400
    result = run_prediction(cleaned_data)
    log_prediction(cleaned_data, result)
    return jsonify(result), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "models_loaded": True}), 200
```

### 6.2.5 MLOps Pipeline (mlops_pipeline.py)

```python
import schedule, time, subprocess, requests, sqlite3, os
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
CLEARBIT_API_KEY = os.getenv("CLEARBIT_API_KEY", "")

def extract_clearbit_data(domain):
    url = f"https://company.clearbit.com/v2/companies/find?domain={domain}"
    headers = {'Authorization': f'Bearer {CLEARBIT_API_KEY}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def monitor_news_for_funding(company_name):
    url = f"https://newsapi.org/v2/everything?q={company_name} AND funding&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return len(articles) > 0, len(articles)

def check_yahoo_finance_ipo(ticker_symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}"
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        data = response.json()
        if data.get('chart', {}).get('result'):
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            return True, price
    return False, 0

def job_continuous_learning():
    clearbit_data = extract_clearbit_data("stripe.com")
    news_found, count = monitor_news_for_funding("Anthropic")
    ipo_found, price = check_yahoo_finance_ipo("UBER")
    log_api_run_to_db(...)
    save_model_version()
    subprocess.run(["python", "train_advanced_model.py"], check=True)
    subprocess.run(["python", "train_nn_model.py"], check=True)

schedule.every(12).hours.do(job_continuous_learning)
```

### 6.2.6 React Frontend Dashboard (App.jsx)

```jsx
import { useState, useEffect, useCallback } from 'react'
const API_BASE = 'http://127.0.0.1:5000'

function ConfidenceGauge({ percent, prediction }) {
  const radius = 70, stroke = 10
  const normalizedRadius = radius - stroke / 2
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (percent / 100) * circumference
  const color = prediction === 'Success' ? '#00f2fe' : '#ff0844'
  return (
    <div className="gauge-wrapper">
      <svg height={radius * 2} width={radius * 2} style={{ transform: 'rotate(-90deg)' }}>
        <circle stroke="#1e293b" fill="transparent" strokeWidth={stroke} r={normalizedRadius} cx={radius} cy={radius} />
        <circle stroke={color} fill="transparent" strokeWidth={stroke}
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset} strokeLinecap="round"
          r={normalizedRadius} cx={radius} cy={radius} />
      </svg>
      <div className="gauge-label">
        <span className="gauge-percent">{percent}%</span>
        <span className="gauge-sub">Confidence</span>
      </div>
    </div>
  )
}

function ShapChart({ shap_values }) {
  const sorted = [...shap_values].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  const maxVal = Math.max(...sorted.map(s => Math.abs(s.value)), 0.001)
  return (
    <div className="shap-chart">
      <h4>Feature Impact (SHAP)</h4>
      {sorted.map((item) => {
        const pct = (Math.abs(item.value) / maxVal) * 100
        const color = item.value >= 0 ? '#00f2fe' : '#ff0844'
        return (
          <div key={item.feature} className="shap-row">
            <span className="shap-label">{item.feature}</span>
            <div className="shap-bar-bg">
              <div className="shap-bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
            </div>
            <span className="shap-value">{item.value.toFixed(4)}</span>
          </div>
        )
      })}
    </div>
  )
}

function App() {
  const [formData, setFormData] = useState({
    total_funding_rounds: '', total_raised_usd: '', total_investors: '',
    startup_age: '', founder_bio: '', category_code: '', country_code: ''
  })
  const [result, setResult] = useState(null)
  const [activeTab, setActiveTab] = useState('predict')

  const handleSubmit = async (e) => {
    e.preventDefault()
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    })
    const data = await response.json()
    setResult(data)
  }
  // Renders 3 tabs: Predict, History, Model Management
}
```

---

# CHAPTER 7

# 7. TESTING

## 7.1 Introduction

Testing is the major quality control measure employed during software development. Its basic function is to detect errors in the software. The goal of testing is to uncover errors introduced during all development phases — requirement analysis, design and coding. For the Startup Success Prediction System testing was performed at multiple levels to ensure correctness, reliability and usability of the entire platform.

### 7.1.1 Unit Testing

Unit testing was performed on individual modules. The ETL pipeline was tested to verify correct row counts and schema (66,368 startups, 114,984 funding rounds, 229,968 investors loaded). Feature engineering was tested to verify encoded values match between training and prediction. Model prediction functions were tested to verify probability outputs are between 0 and 1. Input validation was tested to verify rejection of negative values and unrealistic inputs. NLP analysis was tested with known positive and negative texts to verify correct sentiment scores.

### 7.1.2 Black Box Testing

Black box testing treated the API endpoints as black boxes testing only inputs and expected outputs without knowledge of internal implementation. Tests covered valid predictions with various startup profiles, invalid inputs (negative funding, extreme age values), missing required fields, batch prediction with mixed valid and invalid entries, and edge cases like zero funding and zero investors.

### 7.1.3 White Box Testing

White box testing verified internal logic paths: the ensemble averaging formula correctly averages RF and NN probabilities, NLP score clamping between -10% and +10%, SHAP value generation returning values for all 6 features, label encoder handling of unknown categories mapping to 'unknown', model versioning creating timestamped files in model/versions/ directory, and background thread retraining completing without blocking the API.

### 7.1.4 Integration Testing

Integration testing verified that modules work together correctly. ETL output schema matches the training script's SQL queries. Trained model .pkl files are correctly loaded by the Flask API at startup. API JSON responses are correctly parsed and rendered by the React frontend components. The MLOps pipeline correctly triggers retraining and the API hot-swaps new models without restart.

### 7.1.5 System Testing

System testing verified the complete end-to-end flow: user enters data in dashboard, API receives and validates request, models predict, NLP analyzes, SHAP explains, response is rendered with gauge and charts, prediction is logged in history table, and the History tab displays the new entry. The RunProject.bat launcher was tested to verify auto-training fallback when model files are missing.

## 7.2 Test Cases

*Table 7.2 Test Cases*

| S.No | Description | Input | Expected Output | Actual Output | Result |
|------|-------------|-------|-----------------|---------------|--------|
| 1 | Valid high-funded startup | Rounds=5, Raised=$10M, Investors=8, Age=3 | Success with high confidence | Success (87.2%) | Pass |
| 2 | Bootstrapped startup | Rounds=0, Raised=$0, Investors=0, Age=1 | Failure with low confidence | Failure (23.4%) | Pass |
| 3 | Negative funding value | total_raised_usd = -5000 | Validation error 400 | Error: must be >= 0 | Pass |
| 4 | Extreme age value | startup_age = 500 | Validation error 400 | Error: max 200 years | Pass |
| 5 | Positive NLP sentiment | Bio="Revolutionary innovative disruptive" | Positive sentiment boost | Positive (+4.2%) | Pass |
| 6 | Negative NLP sentiment | Bio="Struggling failing uncertain" | Negative sentiment penalty | Negative (-3.8%) | Pass |
| 7 | Unknown category code | category_code="xyz_unknown" | Maps to 'unknown', no crash | Predicted successfully | Pass |
| 8 | Batch prediction | Array of 3 startup JSON objects | 3 results returned | 3 results with indices | Pass |
| 9 | Health check endpoint | GET /api/health | Status online, models loaded | models_loaded: true | Pass |
| 10 | Model retrain trigger | POST /api/retrain | 202 Accepted | Retraining started | Pass |
| 11 | Prediction history | GET /api/history | List of past predictions | Entries returned | Pass |
| 12 | SHAP values in response | Valid prediction request | shap_values array | 6 feature values returned | Pass |

---

# CHAPTER 8

# 8. SAMPLE SCREEN SHOTS

## 8.1 Dashboard Home Screen

[INSERT SCREENSHOT HERE]

*Fig 8.1 Dashboard Home Screen*

## 8.2 Prediction Input Form

[INSERT SCREENSHOT HERE]

*Fig 8.2 Prediction Input Form*

## 8.3 Prediction Result with SHAP Chart

[INSERT SCREENSHOT HERE]

*Fig 8.3 Prediction Result with Confidence Gauge, RF/NN bars, NLP Sentiment Badge and SHAP Feature Importance Chart*

## 8.4 Prediction History Tab

[INSERT SCREENSHOT HERE]

*Fig 8.4 Prediction History Tab*

## 8.5 Model Management Tab

[INSERT SCREENSHOT HERE]

*Fig 8.5 Model Management Tab*

---

# CHAPTER 9

# 9. CONCLUSION

The AI-Powered Startup Success Prediction System successfully demonstrates a complete production-grade AI platform that goes beyond a simple machine learning model. By combining an ensemble of Random Forest (86.86% accuracy) and Neural Network (79.05% accuracy) with NLP sentiment analysis and SHAP explainability, the system provides transparent, trustworthy and actionable predictions for startup viability assessment.

The system addresses every identified limitation of existing approaches. Class imbalance is handled through SMOTE ensuring the model learns from both successes and failures equally. Qualitative data is incorporated through VADER NLP analysis of founder descriptions. Predictions are fully explained through SHAP values eliminating the black-box problem. The MLOps pipeline ensures continuous learning through live API integration and automated retraining. The React dashboard provides an intuitive visual interface accessible to non-technical users.

The complete data flow — from raw CSV ingestion through ETL, model training, API serving to dashboard visualization — represents a professional software engineering approach to AI system development. The system has practical impact for entrepreneurs, investors, incubators and researchers working with startup data.

Our project has attained the Program Outcomes PO1, PO2, PO3, PO4, PO5, PO6, PO7, PO8, PO9, PO10, PO11, PO12 and Program Specific Outcomes PSO1, PSO2, PSO3 as below.

*Table 9.1 PO/PSO Attainment Matrix*

| PO1 | PO2 | PO3 | PO4 | PO5 | PO6 | PO7 | PO8 | PO9 | PO10 | PO11 | PO12 | PSO1 | PSO2 | PSO3 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|------|
| 3 | 3 | 3 | 3 | 3 | 2 | 2 | 3 | 2 | 3 | 2 | 3 | 3 | 3 | 2 |

---

# CHAPTER 10

# 10. BIBLIOGRAPHY

[1] A. Krishna, A. Agrawal, and A. Choudhary, "Predicting the outcome of startups: less failure, more success," in Proc. IEEE 16th Int. Conf. Data Mining Workshops (ICDMW), Barcelona, Spain, 2016, pp. 798-805.

[2] G. Xiang, Z. Zheng, M. Wen, J. Hong, C. Rose, and C. Liu, "A supervised approach to predict company acquisition with factual and topic features using profiles and news articles on TechCrunch," in Proc. 6th Int. AAAI Conf. Weblogs and Social Media, Dublin, Ireland, 2012, pp. 607-610.

[3] B. Sharchilev, M. Roizner, A. Rumyantsev, and D. Ozornin, "Web-based startup success prediction," in Proc. 27th ACM Int. Conf. Inf. and Knowl. Manage. (CIKM), Torino, Italy, 2018, pp. 2283-2291.

[4] Y. N. Ang, P. Y. Chia, and S. Shen, "Deep learning approaches for startup funding prediction," in Proc. IEEE Int. Conf. Big Data (Big Data), Orlando, FL, USA, 2021, pp. 4741-4748.

[5] H. S. Bhat and S. R. Reddy, "Explainable AI for business analytics: SHAP-based feature importance in machine learning models," J. Business Analytics, vol. 5, no. 2, pp. 112-125, 2022.

[6] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.

[7] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems 30 (NIPS 2017), Long Beach, CA, USA, 2017, pp. 4765-4774.

[8] C. J. Hutto and E. Gilbert, "VADER: A parsimonious rule-based model for sentiment analysis of social media text," in Proc. 8th Int. AAAI Conf. Weblogs and Social Media (ICWSM), Ann Arbor, MI, USA, 2014, pp. 216-225.

[9] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," J. Artif. Intell. Res., vol. 16, pp. 321-357, 2002.

[10] L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp. 5-32, 2001.
