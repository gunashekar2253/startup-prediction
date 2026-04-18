# Startup Success Prediction System — UML Diagrams

---

## 1. Use Case Diagram

```mermaid
graph TB
    subgraph System["🚀 Startup Success Prediction System"]
        UC1["Predict Startup Success"]
        UC2["View Prediction History"]
        UC3["Manage ML Models"]
        UC4["Retrain Models"]
        UC5["Batch Predict Startups"]
        UC6["Check System Health"]
        UC7["Fetch Live API Data"]
        UC8["Auto-Retrain on Schedule"]
    end

    User["👤 User / Entrepreneur"]
    Investor["💼 Investor / VC"]
    Admin["⚙️ Admin / Data Scientist"]
    Scheduler["⏰ Automated Scheduler"]

    User --> UC1
    User --> UC2
    Investor --> UC1
    Investor --> UC5
    Admin --> UC3
    Admin --> UC4
    Admin --> UC6
    Scheduler --> UC7
    Scheduler --> UC8

    UC1 -.->|includes| UC6
    UC4 -.->|includes| UC3
    UC8 -.->|includes| UC7
```

---

## 2. System Architecture Diagram (Component Diagram)

```mermaid
graph TB
    subgraph Frontend["🖥️ Presentation Layer - React + Vite :5173"]
        PredictTab["Predict Tab<br/>Form + Results"]
        HistoryTab["History Tab<br/>Past Predictions"]
        ModelTab["Model Tab<br/>Management + Retrain"]
        HealthBadge["Health Badge<br/>Auto-polls /api/health"]
        GaugeComp["Confidence Gauge<br/>SVG Animation"]
        ShapComp["SHAP Chart<br/>Feature Bars"]
    end

    subgraph Backend["⚙️ Application Layer - Flask API :5000"]
        Router["API Router"]
        Validator["Input Validator"]
        PredEngine["Prediction Engine"]
        RFModel["Random Forest<br/>150 Trees | 86.86%"]
        NNModel["Neural Network<br/>128→64→32 | 79.05%"]
        NLPEngine["VADER NLP<br/>Sentiment Analyzer"]
        SHAPEngine["SHAP Explainer<br/>TreeExplainer"]
        HistLogger["History Logger"]
        RetrainMgr["Retrain Manager<br/>Background Thread"]
    end

    subgraph Data["💾 Data Layer - SQLite"]
        StartupsDB[("startups<br/>66,368 rows")]
        FundingDB[("funding_rounds<br/>114,984 rows")]
        FoundersDB[("founders<br/>229,968 rows")]
        HistoryDB[("prediction_history")]
        MLOpsDB[("mlops_log")]
    end

    subgraph MLOps["🔄 MLOps Layer - Background Process"]
        ClearbitAPI["Clearbit API"]
        NewsAPI["NewsAPI"]
        YahooAPI["Yahoo Finance"]
        Scheduler["12-Hour Scheduler"]
        ModelVersioner["Model Versioner"]
    end

    subgraph Models["📦 Model Artifacts"]
        RFFile["rf_model.pkl"]
        NNFile["nn_model.pkl"]
        ScalerFile["nn_scaler.pkl"]
        EncoderFile["le_category.pkl<br/>le_country.pkl"]
        MetaFile["model_metadata.json"]
        VersionsDir["model/versions/"]
    end

    Frontend -->|HTTP JSON| Router
    Router --> Validator
    Validator --> PredEngine
    PredEngine --> RFModel
    PredEngine --> NNModel
    PredEngine --> NLPEngine
    PredEngine --> SHAPEngine
    PredEngine --> HistLogger
    HistLogger --> HistoryDB
    Router --> RetrainMgr

    RFModel -.->|loads| RFFile
    NNModel -.->|loads| NNFile
    NNModel -.->|loads| ScalerFile
    PredEngine -.->|loads| EncoderFile

    Scheduler --> ClearbitAPI
    Scheduler --> NewsAPI
    Scheduler --> YahooAPI
    Scheduler --> MLOpsDB
    Scheduler --> ModelVersioner
    ModelVersioner --> VersionsDir
    Scheduler -->|triggers| RetrainMgr
```

---

## 3. Sequence Diagram — Single Prediction Flow

```mermaid
sequenceDiagram
    actor User
    participant React as React Dashboard
    participant Flask as Flask API
    participant Val as Input Validator
    participant RF as Random Forest
    participant NN as Neural Network
    participant NLP as VADER NLP
    participant SHAP as SHAP Explainer
    participant DB as SQLite DB

    User->>React: Enter startup metrics + Click "Predict"
    React->>Flask: POST /api/predict (JSON body)
    
    Flask->>Val: validate_inputs(data)
    
    alt Validation Fails
        Val-->>Flask: Return error list
        Flask-->>React: 400 {error: "validation failed", details: [...]}
        React-->>User: Display error message
    end

    Val-->>Flask: Return cleaned data
    
    Flask->>Flask: build_features(data)<br/>Encode category + country
    
    par Parallel Model Inference
        Flask->>RF: predict_proba(features)
        RF-->>Flask: rf_success_prob = 82.45%
    and
        Flask->>NN: scaler.transform → predict_proba
        NN-->>Flask: nn_success_prob = 76.30%
    end
    
    Flask->>Flask: ensemble_avg = (82.45 + 76.30) / 2 = 79.375%
    
    Flask->>NLP: polarity_scores(founder_bio)
    NLP-->>Flask: compound = +0.65 → boost = +6.5%
    
    Flask->>Flask: final_confidence = 79.375 + 6.5 = 85.875%
    
    Flask->>SHAP: shap_values(features)
    SHAP-->>Flask: feature contributions array
    
    Flask->>DB: INSERT INTO prediction_history
    DB-->>Flask: Logged successfully
    
    Flask-->>React: 200 JSON Response
    
    React->>React: Render gauge, bars, NLP badge, SHAP chart
    React-->>User: Display prediction results
```

---

## 4. Sequence Diagram — Model Retrain Flow

```mermaid
sequenceDiagram
    actor Admin
    participant React as React Dashboard
    participant Flask as Flask API
    participant Thread as Background Thread
    participant RF_Train as train_advanced_model.py
    participant NN_Train as train_nn_model.py
    participant DB as SQLite DB
    participant Models as Model Files (.pkl)

    Admin->>React: Click "Trigger Model Retrain"
    React->>Flask: POST /api/retrain
    
    alt Already Retraining
        Flask-->>React: 409 "Retraining already in progress"
    end
    
    Flask->>Thread: Start retrain_task() in daemon thread
    Flask-->>React: 202 "Retraining started in background"
    React-->>Admin: Show "Retraining..." message
    
    Thread->>Thread: retraining_in_progress = True
    
    Thread->>RF_Train: subprocess.run(train_advanced_model.py)
    RF_Train->>DB: SELECT startups, funding, founders
    RF_Train->>RF_Train: Feature engineering (6 features)
    RF_Train->>RF_Train: SMOTE balancing (66K → 120K)
    RF_Train->>RF_Train: Train RandomForest(150 trees)
    RF_Train->>Models: Save rf_model.pkl + encoders + metadata
    RF_Train-->>Thread: Training complete (86.86%)
    
    Thread->>NN_Train: subprocess.run(train_nn_model.py)
    NN_Train->>DB: SELECT from same tables
    NN_Train->>NN_Train: Load saved label encoders
    NN_Train->>NN_Train: SMOTE + StandardScaler
    NN_Train->>NN_Train: Train MLP(128→64→32)
    NN_Train->>Models: Save nn_model.pkl + scaler.pkl
    NN_Train-->>Thread: Training complete (79.05%)
    
    Thread->>Flask: load_models() — hot-reload into memory
    Thread->>Thread: retraining_in_progress = False
    
    Note over React,Flask: Health badge auto-polls every 10s
    React->>Flask: GET /api/health
    Flask-->>React: {retraining_in_progress: false, accuracy: 86.86}
    React-->>Admin: Badge updates to "Online | Acc: 86.86%"
```

---

## 5. Sequence Diagram — MLOps Pipeline Flow

```mermaid
sequenceDiagram
    participant Sched as 12-Hour Scheduler
    participant MLOps as mlops_pipeline.py
    participant Clearbit as Clearbit API
    participant News as NewsAPI
    participant Yahoo as Yahoo Finance
    participant DB as SQLite DB
    participant Version as Model Versioner
    participant Train as Training Scripts

    Sched->>MLOps: Trigger job_continuous_learning()
    
    MLOps->>Clearbit: GET /v2/companies/find?domain=stripe.com
    alt API Key Missing
        Clearbit-->>MLOps: Skipped (no key)
    else Key Present
        Clearbit-->>MLOps: {name, sector, location}
    end
    
    MLOps->>News: GET /v2/everything?q=Anthropic AND funding
    alt API Key Missing
        News-->>MLOps: Skipped (no key)
    else Key Present
        News-->>MLOps: {articles: [...]}
    end
    
    MLOps->>Yahoo: GET /v8/finance/chart/UBER
    Yahoo-->>MLOps: {price: $XX.XX, status: trading}
    
    MLOps->>DB: INSERT INTO mlops_log (all fetched data)
    DB-->>MLOps: Logged
    
    MLOps->>Version: save_model_version()
    Version->>Version: Copy .pkl files to model/versions/<br/>with timestamp prefix
    
    MLOps->>Train: subprocess: train_advanced_model.py
    Train-->>MLOps: RF retrained
    MLOps->>Train: subprocess: train_nn_model.py
    Train-->>MLOps: NN retrained
    
    MLOps-->>Sched: Job complete. Next run in 12 hours.
```

---

## 6. Activity Diagram — Complete Prediction Workflow

```mermaid
flowchart TD
    Start([User Opens Dashboard]) --> Health{Backend Online?}
    Health -->|No| ShowOffline[Display Offline Badge]
    Health -->|Yes| ShowOnline[Display Online Badge + Accuracy]
    ShowOnline --> EnterData[User Enters Startup Metrics]
    
    EnterData --> FillForm["Fill: Funding Rounds, Raised USD,<br/>Investors, Age, Category, Country, Bio"]
    FillForm --> ClickPredict([Click 'Predict Success'])
    
    ClickPredict --> Validate{Input Valid?}
    Validate -->|No| ShowError[Display Validation Errors]
    ShowError --> FillForm
    
    Validate -->|Yes| BuildFeatures["Build Feature Matrix (6 features)<br/>Encode category + country"]
    
    BuildFeatures --> RunRF["Random Forest predict_proba()"]
    BuildFeatures --> RunNN["Scale features → NN predict_proba()"]
    
    RunRF --> GetRFProb["RF Success Prob: 82.45%"]
    RunNN --> GetNNProb["NN Success Prob: 76.30%"]
    
    GetRFProb --> Ensemble["Ensemble Average<br/>(82.45 + 76.30) / 2 = 79.375%"]
    GetNNProb --> Ensemble
    
    Ensemble --> CheckBio{Founder Bio<br/>Provided?}
    CheckBio -->|No| SkipNLP["NLP Score = 0<br/>Sentiment = Neutral"]
    CheckBio -->|Yes| RunNLP["VADER Sentiment Analysis<br/>compound = +0.65"]
    RunNLP --> CalcBoost["NLP Boost = compound × 10<br/>= +6.5%"]
    
    CalcBoost --> FinalScore["Final = 79.375 + 6.5 = 85.875%"]
    SkipNLP --> FinalScore2["Final = 79.375%"]
    
    FinalScore --> Classify{Final ≥ 50%?}
    FinalScore2 --> Classify
    
    Classify -->|Yes| Success["✅ Prediction: SUCCESS"]
    Classify -->|No| Failure["❌ Prediction: FAILURE"]
    
    Success --> RunSHAP["SHAP: Calculate feature contributions"]
    Failure --> RunSHAP
    
    RunSHAP --> LogToDB["Log prediction to SQLite<br/>prediction_history table"]
    
    LogToDB --> RenderResults["Render: Gauge + Bars +<br/>NLP Badge + SHAP Chart"]
    
    RenderResults --> Done([User Views Results])
```

---

## 7. Activity Diagram — ETL Pipeline

```mermaid
flowchart TD
    Start([Run etl_pipeline.py]) --> Connect["Connect to SQLite DB"]
    
    Connect --> CheckBig{"big_startup_secsees<br/>_dataset.csv exists?"}
    
    CheckBig -->|Yes| LoadBig["Read big dataset<br/>(66,368 rows)"]
    LoadBig --> BuildStartups["Build 'startups' table<br/>Map: permalink → company_id<br/>Extract founded_year from date"]
    BuildStartups --> BuildFunding["Build 'funding_rounds' table<br/>Use np.repeat() to expand<br/>N rows per company"]
    BuildFunding --> BuildInvestors["Build 'founders' table<br/>Estimate 2 investors per round"]
    BuildInvestors --> Done([3/3 Tables Loaded ✅])
    
    CheckBig -->|No| Fallback["Fallback to Crunchbase CSVs"]
    
    Fallback --> CheckComp{"companies.csv<br/>exists?"}
    CheckComp -->|Yes| LoadComp["Load startups table"]
    CheckComp -->|No| WarnComp["⚠️ WARNING: Missing"]
    
    LoadComp --> CheckRounds{"rounds.csv<br/>exists?"}
    WarnComp --> CheckRounds
    CheckRounds -->|Yes| LoadRounds["Load funding_rounds table"]
    CheckRounds -->|No| WarnRounds["⚠️ WARNING: Missing"]
    
    LoadRounds --> CheckInv{"investments.csv<br/>exists?"}
    WarnRounds --> CheckInv
    CheckInv -->|Yes| LoadInv["Load founders table"]
    CheckInv -->|No| WarnInv["⚠️ WARNING: Missing"]
    
    LoadInv --> Done2(["N/3 Tables Loaded"])
    WarnInv --> Done2
```

---

## 8. Class Diagram — Backend Structure

```mermaid
classDiagram
    class FlaskApp {
        +secret_key: str
        +CORS: enabled
        +port: 5000
        +route("/api/health") health_check()
        +route("/api/predict") predict_api()
        +route("/api/predict/batch") predict_batch()
        +route("/api/history") get_history()
        +route("/api/retrain") retrain_model()
        +route("/api/model/versions") list_model_versions()
    }

    class PredictionEngine {
        -rf_model: RandomForestClassifier
        -nn_model: MLPClassifier
        -nn_scaler: StandardScaler
        -le_category: LabelEncoder
        -le_country: LabelEncoder
        -explainer: TreeExplainer
        -feature_names: list
        +load_models()
        +build_features(data) DataFrame
        +run_prediction(data) dict
    }

    class InputValidator {
        +validate_inputs(data) tuple
        -check_funding_rounds(val) error
        -check_raised_usd(val) error
        -check_investors(val) error
        -check_startup_age(val) error
    }

    class NLPAnalyzer {
        -analyzer: SentimentIntensityAnalyzer
        +polarity_scores(text) dict
        +get_compound_score(bio) float
    }

    class SHAPExplainer {
        -explainer: TreeExplainer
        +shap_values(features) array
        +get_top_feature(values) str
    }

    class HistoryLogger {
        -DB_PATH: str
        +init_history_db()
        +log_prediction(data, result)
        +get_history(limit) list
    }

    class RetrainManager {
        -retraining_in_progress: bool
        +retrain_task()
        +load_models()
    }

    FlaskApp --> PredictionEngine : uses
    FlaskApp --> InputValidator : uses
    FlaskApp --> HistoryLogger : uses
    FlaskApp --> RetrainManager : uses
    PredictionEngine --> NLPAnalyzer : uses
    PredictionEngine --> SHAPExplainer : uses
```

---

## 9. Class Diagram — ML Training Pipeline

```mermaid
classDiagram
    class TrainAdvancedModel {
        -DB_PATH: str
        -MODEL_DIR: str
        +get_training_data() DataFrame
        +feature_engineering(df) tuple
        +train_model(X, y) tuple
        +generate_shap_explanations(model, X)
        +save_artifacts(model, features, acc)
    }

    class TrainNNModel {
        -DB_PATH: str
        -MODEL_DIR: str
        +get_training_data() tuple
        +train_neural_network(X, y) tuple
    }

    class SMOTE {
        -random_state: 42
        +fit_resample(X, y) tuple
    }

    class RandomForestClassifier {
        -n_estimators: 150
        -max_depth: 15
        -class_weight: balanced
        +fit(X_train, y_train)
        +predict_proba(X) array
    }

    class MLPClassifier {
        -hidden_layer_sizes: 128,64,32
        -max_iter: 500
        -early_stopping: True
        +fit(X_train, y_train)
        +predict_proba(X) array
    }

    class LabelEncoder {
        -classes_: array
        +fit_transform(values) array
        +transform(values) array
    }

    class StandardScaler {
        -mean_: array
        -scale_: array
        +fit_transform(X) array
        +transform(X) array
    }

    TrainAdvancedModel --> SMOTE : balances data
    TrainAdvancedModel --> RandomForestClassifier : trains
    TrainAdvancedModel --> LabelEncoder : encodes categories
    TrainNNModel --> SMOTE : balances data
    TrainNNModel --> MLPClassifier : trains
    TrainNNModel --> StandardScaler : normalizes
    TrainNNModel --> LabelEncoder : reuses encoders
```

---

## 10. Deployment Diagram

```mermaid
graph TB
    subgraph UserDevice["👤 User's Browser"]
        Browser["Web Browser<br/>Chrome / Firefox / Edge"]
    end

    subgraph Server["💻 Local Machine / Server"]
        subgraph FrontendServer["Frontend Server :5173"]
            Vite["Vite Dev Server"]
            ReactApp["React Application<br/>App.jsx + App.css"]
        end

        subgraph BackendServer["Backend Server :5000"]
            Flask["Flask Application<br/>Main.py"]
            ModelFiles["Model Artifacts<br/>model/*.pkl"]
            EnvFile[".env<br/>API Keys"]
        end

        subgraph Database["Database"]
            SQLite[("SQLite<br/>startup_data.db")]
        end

        subgraph BGProcess["Background Process"]
            MLOpsPipe["mlops_pipeline.py<br/>Runs every 12 hours"]
        end
    end

    subgraph ExternalAPIs["🌐 External APIs"]
        Clearbit["Clearbit API<br/>Company Data"]
        NewsAPI["NewsAPI<br/>Funding News"]
        YahooFin["Yahoo Finance<br/>IPO Status"]
    end

    Browser -->|"HTTP :5173"| Vite
    Vite --> ReactApp
    ReactApp -->|"REST API :5000"| Flask
    Flask --> ModelFiles
    Flask --> SQLite
    Flask --> EnvFile
    MLOpsPipe --> SQLite
    MLOpsPipe --> Clearbit
    MLOpsPipe --> NewsAPI
    MLOpsPipe --> YahooFin
    MLOpsPipe -->|"Triggers Retrain"| Flask
```

---

## 11. Data Flow Diagram (DFD Level 0)

```mermaid
flowchart LR
    subgraph External["External Sources"]
        CSV["📁 CSV Datasets"]
        API1["🌐 Clearbit"]
        API2["🌐 NewsAPI"]
        API3["🌐 Yahoo Finance"]
    end

    subgraph System["Startup Prediction System"]
        ETL["ETL Pipeline"]
        DB[("SQLite DB")]
        Train["Training Pipeline"]
        Models["ML Models (.pkl)"]
        API["Flask REST API"]
        FE["React Dashboard"]
    end

    subgraph Users["Users"]
        User["👤 Entrepreneur"]
        Investor["💼 Investor"]
    end

    CSV -->|"Raw Data"| ETL
    ETL -->|"Cleaned Data"| DB
    DB -->|"Training Data"| Train
    Train -->|"Trained Models"| Models
    Models -->|"Loaded at Startup"| API

    API1 -->|"Company Info"| DB
    API2 -->|"Funding News"| DB
    API3 -->|"IPO Status"| DB

    User -->|"Startup Metrics"| FE
    Investor -->|"Batch Data"| FE
    FE -->|"JSON Request"| API
    API -->|"Prediction + SHAP"| FE
    FE -->|"Visual Results"| User
    FE -->|"Batch Results"| Investor
    API -->|"Log Prediction"| DB
```

---

## 12. State Diagram — Model Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Untrained: First Setup

    Untrained --> Training: Run train_advanced_model.py<br/>+ train_nn_model.py
    
    Training --> Trained: Models saved as .pkl
    Training --> Failed: Training error
    Failed --> Training: Fix data & retry
    
    Trained --> Serving: Main.py loads models
    
    Serving --> Predicting: /api/predict called
    Predicting --> Serving: Response returned
    
    Serving --> Retraining: /api/retrain or<br/>MLOps 12-hour trigger
    Retraining --> Versioned: Old models backed up<br/>to model/versions/
    Versioned --> Training
    
    Serving --> Stale: Model accuracy<br/>degrades over time
    Stale --> Retraining: Manual or auto retrain

    note right of Serving
        Models are hot-reloaded
        after retraining completes.
        No server restart needed.
    end note
```

---

> **Note:** All diagrams above use Mermaid syntax. They can be rendered in:
> - GitHub / GitLab (native support)
> - VS Code (with Mermaid extension)
> - Any Mermaid live editor: [mermaid.live](https://mermaid.live)
