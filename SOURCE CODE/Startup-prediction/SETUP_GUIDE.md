# 🚀 Startup Success Predictor — Setup & Run Guide

> **Repository:** [https://github.com/gunashekar2253/startup-prediction](https://github.com/gunashekar2253/startup-prediction)

Follow the steps below to download, configure, and run this project on your local machine.

---

## 📋 Prerequisites

Make sure you have the following installed before proceeding:

| Tool       | Minimum Version | Download Link                          |
|------------|-----------------|----------------------------------------|
| **Python** | 3.9+            | [python.org/downloads](https://www.python.org/downloads/) |
| **Node.js**| 18+             | [nodejs.org](https://nodejs.org/)      |
| **npm**    | 9+              | *(comes bundled with Node.js)*         |
| **Git**    | any             | [git-scm.com](https://git-scm.com/)   |

> **Tip:** Run `python --version`, `node --version`, and `git --version` in your terminal to verify.

---

## Step 1 — Clone the Repository

Open a terminal (Command Prompt / PowerShell / Git Bash) and run:

```bash
git clone https://github.com/gunashekar2253/startup-prediction.git
```

Then navigate into the project folder:

```bash
cd startup-prediction/SOURCE CODE/Startup-prediction
```

> 📁 This is the **root working directory** for all remaining steps.

---

## Step 2 — Set Up Python Virtual Environment (Recommended)

Create and activate a virtual environment to isolate dependencies:

**Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

---

## Step 3 — Install Python Dependencies

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This installs all required Python packages: Flask, scikit-learn, pandas, SHAP, etc.

---

## Step 4 — Configure Environment Variables (API Keys)

This project uses external APIs. You need to set up a `.env` file:

1. **Copy the example file:**
   ```bash
   copy .env.example .env
   ```
   *(On macOS/Linux use `cp .env.example .env`)*

2. **Open `.env` in any text editor** and fill in your API keys:

   ```env
   # Get free key at: https://newsapi.org/register
   NEWS_API_KEY=your_actual_newsapi_key

   # Get free key at: https://dashboard.clearbit.com/signup
   CLEARBIT_API_KEY=your_actual_clearbit_key

   # Any random string for Flask session security
   SECRET_KEY=any_random_secret_string
   ```

> ⚠️ **Important:** The `.env` file is listed in `.gitignore` — it will NOT be pushed to GitHub. Each person must create their own.

> **Note:** Yahoo Finance API does **not** require a key.

---

## Step 5 — Train the ML Models

The machine learning models (`.pkl` files) are **not** included in the repository (they are in `.gitignore`). You must train them locally:

```bash
python train_advanced_model.py
python train_nn_model.py
```

After successful training, you will see model files created inside the `model/` folder:
- `model/startup_success_rf_model.pkl` (Random Forest)
- `model/startup_success_nn_model.pkl` (Neural Network)

---

## Step 6 — Install Frontend Dependencies

Navigate to the `frontend` folder and install Node.js packages:

```bash
cd frontend
npm install
cd ..
```

---

## Step 7 — Run the Project

You have **two options** to start the application:

### Option A: Use the One-Click Launcher (Windows Only)

Simply double-click `RunProject.bat` or run:

```bash
RunProject.bat
```

This will automatically:
1. Check if models exist (trains them if missing)
2. Start the Flask backend server
3. Start the React frontend dev server

### Option B: Start Manually (Any OS)

**Terminal 1 — Start the Backend:**
```bash
python Main.py
```
The Flask API server starts at: **http://127.0.0.1:5000**

**Terminal 2 — Start the Frontend:**
```bash
cd frontend
npm run dev
```
The React app starts at: **http://localhost:5173**

---

## Step 8 — Open the Application

Once both servers are running, open your browser and go to:

🌐 **http://localhost:5173**

You should see the Startup Success Predictor dashboard.

---

## 📂 Project Structure (Quick Reference)

```
Startup-prediction/
├── Dataset/                  # CSV datasets (Crunchbase, startup data)
├── data_pipeline/            # ETL pipeline & database logic
├── frontend/                 # React + Vite frontend application
│   ├── src/                  # React source code
│   ├── package.json          # Node.js dependencies
│   └── vite.config.js        # Vite configuration
├── model/                    # Trained model files (generated locally)
├── .env.example              # Template for environment variables
├── .gitignore                # Files excluded from Git
├── Main.py                   # Flask backend API server
├── etl_pipeline.py           # ETL data processing pipeline
├── mlops_pipeline.py         # MLOps automation pipeline
├── setup_db.py               # Database setup script
├── train_advanced_model.py   # Random Forest model training
├── train_nn_model.py         # Neural Network model training
├── requirements.txt          # Python dependencies
├── RunProject.bat            # One-click launcher (Windows)
└── StartupPrediction.ipynb   # Jupyter Notebook (EDA & experiments)
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` not recognized | Add Python to your system PATH, or use `python3` instead |
| `npm` not recognized | Install Node.js from [nodejs.org](https://nodejs.org/) |
| Model training fails | Ensure the `Dataset/` folder has all CSV files and dependencies are installed |
| Frontend shows CORS error | Make sure the Flask backend is running on port 5000 before opening the frontend |
| `pip install` fails | Try upgrading pip: `python -m pip install --upgrade pip` |
| Port 5000 already in use | Close any other app using port 5000, or change the port in `Main.py` |

---

## 📝 Quick Start Summary

```bash
# 1. Clone
git clone https://github.com/gunashekar2253/startup-prediction.git
cd startup-prediction/SOURCE\ CODE/Startup-prediction

# 2. Python setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. Environment variables
copy .env.example .env         # then edit .env with your API keys

# 4. Train models
python train_advanced_model.py
python train_nn_model.py

# 5. Frontend setup
cd frontend && npm install && cd ..

# 6. Run
RunProject.bat                 # Windows one-click
# OR manually: python Main.py  (terminal 1)  +  cd frontend && npm run dev  (terminal 2)
```

---

**Happy Coding! 🎉**
