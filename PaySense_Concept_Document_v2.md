# PaySense: Gen-Z Financial Copilot
### An AI-Driven System for Financial Health Forecasting and BNPL Debt Prevention

**Version:** 4.0 — June 2026  
**Student:** Nur Farah Binti Ahmad Nazri (22007916)  
**Supervisor:** Dr. Nazleeni Samiha Bt Haron  
**Institution:** Universiti Teknologi PETRONAS — Bachelor of Computer Science (Hons)  
**Area:** Computer Science (Data Analytics)  
**Grand Challenge:** Sustainable Living  
**Research Theme:** Inclusive and Equitable Development

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | May 2025 | Initial concept document |
| 2.0 | June 2025 | Major revision: confirmed ML model candidates (ARIMA, Prophet, LSTM for forecasting; Random Forest, XGBoost, LightGBM for classification), updated tech stack (Next.js, Supabase, FastAPI), added detailed model evaluation plan, two-semester Gantt chart, open-source references, and architecture details |
| 3.0 | June 2026 | Literature review integration: expanded from 5 to 30 papers across 7 domains (2021–2026), added full Literature Review Matrix (Section 10.2), 5 Research Gaps (Section 10.3), updated Kaggle dataset to ramyapintchy Personal Finance Data (2020–2024), expanded references to all 30 papers |
| 4.0 | June 2026 | ML training complete: Engine 1 winner (ARIMA, RMSE = 7,300.30) and Engine 2 winner (XGBoost, F1 = 0.8718, ROC-AUC = 0.8751, Recall = 0.8500) documented with full 3-model empirical comparison tables (Section 8.3) |

---

## 1. Research Question

> To what extent can an AI-driven predictive system, combining time-series forecasting and BNPL risk classification, effectively prevent debt accumulation from overlapping BNPL instalments among Malaysian university students?

---

## 2. Research Objectives

1. Identify and evaluate 3 candidate ML models for **time-series cash flow forecasting** (ARIMA, Prophet, LSTM) and select the best-performing model based on empirical evaluation metrics.
2. Identify and evaluate 3 candidate ML models for **BNPL risk classification** (Random Forest, XGBoost, LightGBM) and select the best-performing model based on empirical evaluation metrics.
3. Develop a **Financial Health Forecasting Engine** that estimates the user's projected bank balance for the next 3 rolling months based on historical spending habits.
4. Implement an **AI-Driven BNPL Risk Classifier** that evaluates proposed purchases and dynamically calculates a risk score to flag potential debt traps.
5. Design and deploy an **interactive web application (PaySense)** that visualises future financial health metrics, provides proactive alerts, and delivers explainable AI (XAI) insights using SHAP.

---

## 3. Problem Statement

Traditional personal finance apps are reactive digital ledgers — they log historical spending but don't forecast future cash flows. This is a critical gap for Malaysian university students who heavily rely on BNPL services.

**Key statistics:**
- Malaysian BNPL market has reached **RM9.3 billion** in spending
- Gen-Z students are among the heaviest adopters due to limited income and no access to traditional credit
- Despite high self-reported financial literacy (M = 4.07/5), the correlation with actual debt management behaviour is very weak (r = 0.060) — the **Knowledge-Action Gap**
- Dominant predictor of BNPL usage: **Perceived Behavioural Control** (Beta = 0.694) — students overestimate their repayment ability

### Core Problems

| # | Problem |
|---|---------|
| 1 | Existing finance apps are reactive ledgers, not predictive tools |
| 2 | BNPL instalment stacking creates invisible future cash flow crises |
| 3 | Students cannot stress-test new purchases against future projected balances |
| 4 | No AI-driven system combines time-series forecasting AND BNPL risk classification for Malaysian students |

### Gap in Existing Solutions

| Platform Type | Example | Limitation |
|---------------|---------|------------|
| AI Financial Platform | ANIKA by IBPO, RinggitWise | Chatbot prompts only, not predictive modelling |
| Banking / Investment App | Moomoo MY, UOB Mighty Insights | Siloed within one bank, no BNPL integration |
| BNPL Provider App | SPayLater, Atome, GrabPayLater | Has calculators but no future balance forecasting |
| **PaySense (This Project)** | — | **Combines both forecasting AND BNPL risk classification — this gap does not exist in the market** |

---

## 4. Machine Learning Model Evaluation Plan

Both AI engines follow a **compare-3-pick-1** approach. All candidate models are trained, tested, and compared on the same dataset. The best-performing model per engine is selected and integrated into the final system.

### 4.1 Engine 1 — Cash Flow Forecasting (Time-Series)

Models are trained on historical transaction data to project the user's bank balance month-by-month for the next 3 months.

| Model | Type | Strengths | Weaknesses | Reference |
|-------|------|-----------|------------|-----------|
| **ARIMA** | Statistical Time-Series | Computationally efficient, interpretable, well-established for short-horizon forecasting | Assumes linearity; struggles with irregular spending patterns and missing dates | Box & Jenkins (1970); Weytjens et al. (2021) |
| **Prophet** | Additive Time-Series (Facebook) | Handles seasonality natively (monthly allowance cycles); robust to missing dates; minimal tuning | Higher error on short-term daily fluctuations | Taylor & Letham (2018); Yadav (2022) |
| **LSTM** | Deep Learning (Recurrent NN) | Highest accuracy for complex, long-term patterns; captures non-linear spending behaviour | Requires more data and compute; less interpretable | Weytjens et al. (2021); Yadav (2022) |

**Evaluation Metrics:**

| Metric | Description |
|--------|-------------|
| MAE (Mean Absolute Error) | Average absolute difference between predicted and actual balance — lower is better |
| RMSE (Root Mean Square Error) | Penalises large prediction errors more heavily — lower is better |
| MAPE (Mean Absolute Percentage Error) | Percentage-based error, useful for comparing across scales |

> **Selection criterion:** Lowest RMSE on the test set. If RMSE values are close, MAE and MAPE are used as tiebreakers.

---

### 4.2 Engine 2 — BNPL Risk Classification

Ensemble classification models trained on synthetic BNPL scenario data to classify a proposed purchase as **Safe**, **At Risk**, or **Danger** given the user's forecasted future balance.

| Model | Type | Strengths | Weaknesses | Reference |
|-------|------|-----------|------------|-----------|
| **Random Forest** | Ensemble (Bagging) | Robust to overfitting; handles imbalanced data; high recall for risky cases | Lower overall accuracy vs. boosting models | Mittal et al. (2018); XAI paper (2025) |
| **XGBoost** | Ensemble (Gradient Boosting) | Highest accuracy and AUC-ROC in credit risk literature; handles imbalanced datasets with SMOTE; industry standard for fintech | More hyperparameters to tune | Multiple credit risk studies (2022–2024) |
| **LightGBM** | Ensemble (Gradient Boosting) | Fastest training speed; best precision; efficient on large datasets; native SHAP support | Can overfit on small datasets | XAI credit risk paper (2025); Default prediction studies |

**Evaluation Metrics:**

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| Accuracy | Overall % of correct predictions | Baseline performance measure |
| Precision | Of all predicted 'Risky', how many were actually risky | Avoids false alarms — important for user trust |
| Recall | Of all actual 'Risky' cases, how many were caught | **Critical** — missing a risky case is more harmful than a false alarm |
| F1-Score | Harmonic mean of Precision and Recall | Balanced measure for imbalanced datasets |
| ROC-AUC | Area under the Receiver Operating Characteristic curve | Overall discriminatory power of the classifier |

> **Selection criterion:** Highest F1-Score and ROC-AUC on validation set. **Recall is prioritised over Precision.** SMOTE applied to address class imbalance before training.

---

### 4.3 XAI Layer — Explainability

Both engines use **SHAP (SHapley Additive exPlanations)** for explainability. All three Engine 2 candidate models have native SHAP support via the `shap` Python library.

SHAP values are translated into plain-language explanations, e.g.:
> *"Your risk is HIGH because your projected balance drops below RM0 on 15 July due to 3 overlapping BNPL payments."*

---

## 5. Data Strategy

### 5.1 Dataset Sources

| Source | Purpose | Description |
|--------|---------|-------------|
| Kaggle — Personal Finance Data (ramyapintchy) | Engine 1 Training (Forecasting) | Synthetic personal finance transaction records (Jan 2020–Dec 2024) with columns: Date, Transaction Description, Category, Amount, Type (Income/Expense). Download: [kaggle.com/datasets/ramyapintchy/personal-finance-data](https://www.kaggle.com/datasets/ramyapintchy/personal-finance-data) |
| Personal Real Transaction Data | Engine 1 Demo Data | 4+ months of real personal transaction records covering income, categorised expenses, and BNPL instalments |
| Synthetic BNPL Scenarios (Generated) | Engine 2 Training (Classification) | Algorithmically generated BNPL purchase scenarios with labelled outcomes (Safe / At Risk / Danger) |
| User-Inputted Data (Live) | Production Runtime | Transactions entered manually or imported via CSV / PDF bank statement upload |

### 5.2 Dataset Requirements

| Requirement | Why It Matters |
|-------------|----------------|
| Time-series structure with date column | Required for ARIMA, Prophet, and LSTM to learn temporal patterns |
| Minimum 6 months of transaction history | Sufficient data for train/test split and for LSTM to learn long-term dependencies |
| Seasonality present (monthly income cycles) | Validates Prophet's seasonal decomposition capability |
| Handles missing dates / irregular transactions | Real student spending is irregular; Prophet and LSTM handle this better than ARIMA |
| Class imbalance in BNPL risk labels | Expect 80–90% 'Safe' cases; SMOTE will be applied before classification training |
| Malaysian Ringgit (RM) denomination or convertible | Directly relevant to target user demographic |

### 5.3 Transaction Data Fields

| Field | Type | Description |
|-------|------|-------------|
| Date | Date | Date of transaction (YYYY-MM-DD) |
| Amount (RM) | Float | Transaction value in Malaysian Ringgit |
| Type | Categorical | Income or Expense |
| Category | Categorical | Food, Transport, Shopping, Bills, BNPL Instalment, etc. |
| Description | Text | Optional free-text note |
| Bank / Account | Text | Which bank account or e-wallet |
| BNPL Flag | Boolean | Indicates if this transaction is a BNPL instalment payment |
| BNPL Plan ID | String | Links the transaction to a specific BNPL plan if flagged |

### 5.4 BNPL Plan Data Fields

| Field | Type | Description |
|-------|------|-------------|
| Item Name | Text | Name of purchased item |
| BNPL Provider | Categorical | e.g. Atome, Shopee PayLater, GrabPayLater |
| Total Price (RM) | Float | Full purchase value |
| Interest Rate (%) | Float | 0% if none, or applicable rate |
| Total Payable (RM) | Float (Calculated) | Total price + interest |
| Number of Instalments | Integer | e.g. 3, 6, 12 |
| Instalment Amount (RM) | Float (Calculated) | Total payable ÷ number of instalments |
| First Payment Date | Date | Start date of repayment schedule |
| Instalment Due Dates | Date Array | Auto-calculated schedule of all payment dates |
| Status | Categorical | Active / Completed / Overdue |
| Risk Score at Creation | Integer (0–100) | Score generated when the plan was created via Risk Checker |

---

## 6. System Architecture

### 6.1 Sequential AI Architecture

The two AI engines work **sequentially** — this is the core technical novelty of PaySense. The Forecasting Engine's output becomes the evaluation context for the Risk Classifier.

| Step | Action | Engine |
|------|--------|--------|
| 1 | Forecasting Engine runs first. Takes user's historical transaction data (income, spending patterns, existing active BNPL instalments) and generates a projected future balance curve for the next 3 months. | Engine 1 |
| 2 | User proposes a new BNPL purchase: item name, total price, BNPL provider, number of instalments, first payment date, interest rate. | User Input |
| 3 | Risk Classifier uses the forecasted balance curve as its evaluation context. Instead of asking "can you afford this today?", it overlays the new instalment schedule onto the projected balance curve: "will your future balance absorb each payment at every instalment date?" | Engine 2 |
| 4 | System outputs: risk score (0–100), health label (Safe / At Risk / Danger), visual impact overlay on the balance chart, and SHAP-based XAI explanation. | Output |

### 6.2 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Frontend Framework | **Next.js** (React-based) | Optimal for Vercel deployment; server-side rendering; best ecosystem for dynamic dashboards |
| Styling | **Tailwind CSS** | Utility-first; rapid development; responsive design out of the box |
| Backend / API | **Python with FastAPI** | Modern, async-capable; industry standard for ML model serving; auto-generates API docs |
| Forecasting Model | **ARIMA / Prophet / LSTM** (best selected) | Python-native via statsmodels, prophet, and tensorflow/keras |
| Risk Classifier | **Random Forest / XGBoost / LightGBM** (best selected) | scikit-learn / xgboost / lightgbm Python libraries |
| XAI Library | **SHAP** | Native support for all three candidate classifiers |
| Database | **Supabase (PostgreSQL)** | Open-source; relational structure suits transactions + BNPL plan linkage; free tier available |
| Data Visualisation | **Chart.js or Recharts** (via Next.js) | Lightweight, responsive charts embedded in the web app |
| File Parsing — CSV | **pandas** (Python) | Industry-standard data manipulation library |
| File Parsing — PDF | **pdfplumber** (Python) | Handles Malaysian bank statement PDF formats (Maybank, CIMB, RHB, etc.) |
| Deployment — Frontend | **Vercel** | Native Next.js deployment platform; zero-config; free tier |
| Deployment — Backend | **Render or Railway** | Free-tier Python/FastAPI hosting; supports background workers for model inference |
| Development Tools | VS Code, Jupyter Notebook, Git/GitHub | Model prototyping in Jupyter; version control on GitHub |
| Model Training Environment | **Google Colab (Pro)** | GPU-accelerated LSTM training |

### 6.3 Three-Tier Architecture

```
[Frontend — Next.js on Vercel]
  ↕ REST API calls
[Backend — FastAPI on Render/Railway]
  - Hosts trained ML models (.pkl / .h5)
  - Endpoints: balance forecasting, BNPL risk scoring, SHAP explanations, transaction CRUD, PDF/CSV parsing
  ↕ Supabase Python client
[Database — Supabase (PostgreSQL)]
  - User transaction records
  - BNPL plan records
  - Risk score history
```

### 6.4 Open-Source References

| Repository | URL | Relevance |
|------------|-----|-----------|
| DominicRoyStang / finance-predictor | github.com/DominicRoyStang/finance-predictor | Personal finance ML prediction using transaction CSV — model selection approach reference |
| sergepaulc / Machine-Learning-for-Finance | github.com/sergepaulc/Machine-Learning-for-Finance | scikit-learn and PyTorch ML models for financial applications |
| naveenventuri / Cash-Flow-prediction-master | github.com/naveenventuri/Cash-Flow-prediction-master | SVR, Random Forest, DNN, ensemble models for cash flow |
| Facebook Prophet (official) | github.com/facebook/prophet | Official Prophet library |
| SHAP (official) | github.com/slundberg/shap | Official SHAP library — integrates with all three classifier candidates |
| Supabase Python client | github.com/supabase/supabase-py | Official Supabase Python client for FastAPI backend |
| XGBoost (official) | github.com/dmlc/xgboost | Official XGBoost library |
| LightGBM (official) | github.com/microsoft/LightGBM | Official Microsoft LightGBM library |

---

## 7. Application Features & Screens

PaySense is a fully deployed responsive web application accessible on desktop and mobile browsers via Vercel.

### 7.1 Screen Summary

| Screen | Key Features |
|--------|-------------|
| **Main Dashboard** | Current balance, Financial Health Speedometer (0–100), 6-month bar chart (3 historical + 3 projected), active BNPL list, upcoming payment alerts, Check New BNPL button |
| **Transactions Screen** | Full transaction history, manual entry form, CSV import, PDF bank statement import, category breakdown, balance sync |
| **BNPL Plans Manager** | List of all active/completed BNPL plans, manual plan creation, plan detail view with instalment timeline |
| **BNPL Risk Checker** | Purchase input form, risk score output (0–100), balance impact overlay chart, SHAP explanation panel, top risk factors, recommendation, confirm & save to plans |

---

### 7.2 Main Dashboard

| Feature | Description |
|---------|-------------|
| Current Balance Display | Prominently displays the user's current bank balance. Can be manually corrected at any time via Balance Sync. |
| Income & Expense Summary | Mini cards showing total income and total expenses for the current month. |
| Financial Health Speedometer | Speedometer-style gauge, score 0–100. Labels: **Safe** (green, 70–100), **At Risk** (yellow, 40–69), **Danger** (red, 0–39). Computed by ML risk classifier combined with rule-based thresholds. |
| 6-Month Bar Chart | 3 months of historical spending (actual) + 3 months of projected spending (AI Engine 1). Income and expense bars shown side by side per month. |
| Active BNPL Instalments List | All ongoing BNPL commitments: provider name, item, remaining instalments, amount per instalment, next due date. |
| Upcoming Payment Alerts | Proactive alerts highlighting BNPL payments due within 7 days, flagging if projected balance may be insufficient. |
| Check New BNPL Button | CTA button navigating to the BNPL Risk Checker screen. |

---

### 7.3 Transaction Management

| Feature | Description |
|---------|-------------|
| Manual Transaction Entry | Fields: Amount, Type (Income/Expense), Category, Description, Date, Bank/Account Name, BNPL flag toggle. |
| CSV Bank Statement Import | Upload CSV; system parses and maps transactions automatically with a preview screen for user confirmation. |
| PDF Bank Statement Import | Upload PDF from Malaysian banks (Maybank, CIMB, RHB, etc.). Parsed via pdfplumber. Best-effort parsing — user confirmation required. |
| Balance Sync | User can manually override and correct current balance at any time. |
| Transaction History | Full searchable and filterable list of all transactions by month, category, or type. |
| Category Breakdown | Visual breakdown of spending by category with percentage bars. |

---

### 7.4 BNPL Plans Manager

| Feature | Description |
|---------|-------------|
| Manual BNPL Plan Creation | User inputs: Item Name, BNPL Provider, Total Price, Interest Rate (%), Number of Instalments, First Payment Date. System auto-calculates each instalment amount and due date schedule. |
| Auto-Creation from Risk Checker | If user confirms a purchase after risk checking, system automatically creates a new BNPL plan record. |
| Plan Detail View | Full instalment timeline, payment schedule, current status (Active / Completed / Overdue), and the risk score generated at creation. |
| Plan Status Tracking | Plans are marked Overdue if a payment date passes without the user logging the payment. |

---

### 7.5 BNPL Risk Checker

The **core feature** of PaySense. The two AI engines work sequentially here.

| Feature | Description |
|---------|-------------|
| Purchase Input Form | User enters: Item Name, Total Price (RM), BNPL Provider, Number of Instalments, First Payment Date, Interest Rate (%). |
| Risk Score Output | Prominent score 0–100 with colour-coded label: **Safe** (green), **At Risk** (yellow), **Danger** (red). |
| Balance Impact Overlay | 6-month balance chart re-renders with proposed BNPL instalments overlaid as a new line, visually showing where the balance would drop. |
| SHAP XAI Explanation Panel | Plain-language explanation of the top 3 factors driving the risk score. Example: *"Your risk is HIGH because: (1) You already have 2 active BNPL plans. (2) Your projected balance drops below RM50 on 10 August. (3) Your instalment-to-income ratio exceeds 30%."* |
| Recommendation | System-generated recommendation: e.g. "Safe to proceed", "Consider delaying by 3 weeks", or "Avoid this purchase — high deficit risk in Month 2". |
| Confirm & Save to Plans | If user proceeds, a single tap saves the purchase as a new active BNPL plan in the BNPL Plans Manager. |

---

### 7.6 Forecasting Engine (Backend)

Powers the 6-month bar chart and the risk evaluation context. Not a standalone screen.

| Feature | Description |
|---------|-------------|
| Time-Series Forecasting | Uses the selected best model (ARIMA / Prophet / LSTM) to project month-by-month income and expense trends for the next 3 months. |
| Future Balance Curve | Computes a week-by-week projected bank balance for the next 3 months, accounting for known recurring expenses and active BNPL instalment schedules. |
| Instalment Stress Testing | When a new BNPL purchase is proposed, the engine simulates the updated balance curve with new instalments included, identifying any dates where balance drops below a safe threshold (RM0 or a configurable buffer). |

---

## 8. AI Model Design (Detailed)

### 8.1 Engine 1 — Financial Health Forecasting

| Attribute | Detail |
|-----------|--------|
| Candidate Models | ARIMA, Prophet, LSTM (all three trained and evaluated — best selected) |
| Input Features | Historical monthly income, historical monthly expenses by category, active BNPL instalment schedule, current balance, date features (month, day of week, is_month_end) |
| Output | Projected month-by-month balance for next 3 months; week-by-week balance curve for stress testing |
| Training Data | Kaggle personal finance dataset + personal 4-month transaction history (augmented with synthetic data if needed) |
| Train/Test Split | 80% training, 20% testing (time-series aware split — no random shuffling) |
| Evaluation Metrics | MAE, RMSE, MAPE |
| Selection Criterion | Lowest RMSE on test set |
| Training Tool | Google Colab Pro (Python / Jupyter Notebook) |
| Serialisation | Saved as `.pkl` (ARIMA/Prophet) or `.h5` (LSTM) and loaded by FastAPI at inference time |

### 8.2 Engine 2 — BNPL Risk Classifier

| Attribute | Detail |
|-----------|--------|
| Candidate Models | Random Forest, XGBoost, LightGBM (all three trained and evaluated — best selected) |
| Input Features | Proposed instalment amount (RM), number of existing active BNPLs, total monthly BNPL commitment, forecasted balance at each instalment date, income stability score, remaining balance buffer, instalment-to-income ratio, days until next payday |
| Output | Risk score (0–100) and risk label (Safe / At Risk / Danger) |
| XAI Layer | SHAP — feature importance values mapped to plain-language explanations for end-user display |
| Training Data | Synthetic BNPL scenarios with labelled outcomes (Safe / At Risk / Danger) |
| Class Imbalance Handling | SMOTE applied before training to balance classes |
| Train/Test Split | 70% training, 15% validation, 15% test |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| Selection Criterion | Highest F1-Score and ROC-AUC; Recall prioritised |
| Hyperparameter Tuning | Grid Search or Optuna for XGBoost and LightGBM |
| Serialisation | Saved as `.pkl` and loaded by FastAPI at inference time |

### 8.3 ML Model Evaluation Results

Both engines have been trained and evaluated on Google Colab (GPU T4). Results below reflect empirical performance on held-out test sets using the Kaggle Personal Finance Dataset (Engine 1) and 2,000 synthetic BNPL profiles (Engine 2).

#### Engine 1 — Cash Flow Forecasting Results

**Dataset:** Kaggle Personal Finance Data (Jan 2020 – Dec 2024) | **Split:** 80% train / 20% test (time-series aware, no shuffling)

| Rank | Model | MAE | RMSE | MAPE | Selected |
|------|-------|----:|-----:|-----:|:--------:|
| 1 | **ARIMA** | 5,350.51 | **7,300.30** | 257.89% | ✓ WINNER |
| 2 | Prophet | 6,518.97 | 7,607.86 | 274.01% | |
| 3 | LSTM | 7,911.61 | 9,923.00 | 516.27% | |

> **Winner: ARIMA** — selected by lowest RMSE (7,300.30). ARIMA outperformed Prophet by 4.0% on RMSE and LSTM by 26.4% on RMSE.
>
> **Note on MAPE:** All three models show high MAPE (>100%) because the target variable — monthly net cashflow — includes months near zero or negative, causing percentage-based errors to be mathematically inflated. RMSE was used as the primary selection metric as planned, since it is robust to this issue.

#### Engine 2 — BNPL Risk Classification Results

**Dataset:** 2,000 synthetic BNPL profiles (Gen-Z Malaysian students) | **Split:** 70% train / 15% val / 15% test (stratified) | **SMOTE applied on train set only**

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | F1+AUC Avg | Selected |
|------|-------|:--------:|:---------:|:------:|:--:|:-------:|:----------:|:--------:|
| 1 | **XGBoost** | 0.8167 | 0.8947 | **0.8500** | **0.8718** | 0.8751 | **0.8734** | ✓ WINNER |
| 2 | LightGBM | 0.8067 | **0.9091** | 0.8182 | 0.8612 | **0.8831** | 0.8722 | |
| 3 | Random Forest | 0.8100 | 0.9015 | 0.8318 | 0.8652 | 0.8763 | 0.8708 | |

> **Winner: XGBoost** — selected by highest average of F1 (0.8718) and ROC-AUC (0.8751), giving a combined score of 0.8734. XGBoost also achieved the highest Recall (0.8500), which is the prioritised metric — missing a high-risk case is more harmful than a false alarm.
>
> **Note on LightGBM:** LightGBM achieved the highest Precision (0.9091) and highest ROC-AUC (0.8831), but its lower Recall (0.8182) placed it second under the Recall-prioritised selection criterion. In a BNPL risk context, a false negative (missed high-risk case) carries greater financial harm than a false positive (unnecessary warning), justifying Recall as the tiebreaker.

#### Summary of Selected Models

| Engine | Task | Winner | Primary Metric | Value |
|--------|------|--------|---------------|-------|
| Engine 1 | Cash Flow Forecasting | **ARIMA** | RMSE (lowest) | 7,300.30 |
| Engine 2 | BNPL Risk Classification | **XGBoost** | F1 + ROC-AUC (highest avg) | 0.8734 |

Both winner models are serialised and saved to `PaySense/models/` in Google Drive:
- `engine1_winner.pkl` — ARIMA forecasting model (pmdarima)
- `engine2_winner.pkl` — XGBoost classification model

---

## 9. Project Scope

### In Scope

- Fully deployed responsive web application (Next.js on Vercel)
- Two AI engines: time-series forecasting (best of ARIMA / Prophet / LSTM) and BNPL risk classification (best of Random Forest / XGBoost / LightGBM)
- Empirical model comparison and selection based on evaluation metrics
- XAI explanations for risk classifier output using SHAP
- Manual transaction entry and bank statement import (CSV and PDF — Malaysian bank formats)
- BNPL Plans Manager with manual and auto-creation flows
- Financial Health Speedometer (0–100 score with Safe / At Risk / Danger labels)
- Supabase (PostgreSQL) database for persistent data storage
- FastAPI Python backend for ML model serving and data operations
- Full literature review matrix comparing at least 20 papers on BNPL risk, credit scoring, and cash flow forecasting

### Out of Scope (Future Work)

- User authentication and multi-user account management
- Direct API integration with Malaysian banks or e-wallets for automated transaction sync
- Push notifications (mobile app-style alerts)
- Investment tracking or stock portfolio features
- Peer comparison or social features
- Full BNPL provider API integration (e.g. Atome, SPayLater)

---

## 10. Academic & Research Alignment

### 10.1 Literature Review Summary

A total of **30 papers (2021–2026)** have been reviewed and categorised across 7 domains. Sources: Google Scholar, IEEE Xplore, ScienceDirect, Springer, Taylor & Francis.

**Distribution:** 21 Journal Papers | 3 Conference Papers | 3 Working Papers/Preprints | 3 Review Papers

| Domain | Count |
|--------|-------|
| BNPL Consumer Behavior | 8 |
| BNPL Regulation & Market | 3 |
| Financial Literacy | 4 |
| Time-Series Forecasting | 5 |
| Credit Risk Classification | 5 |
| Explainable AI (XAI) | 3 |
| Personal Finance Apps | 2 |
| **Total** | **30** |

---

### 10.2 Literature Review Matrix

#### A — BNPL Consumer Behavior (8 papers)

| # | Author(s) & Year | Key Findings | Research Gap | PaySense Relevance |
|---|-----------------|-------------|-------------|-------------------|
| 1 | Osman et al. (2024) | Attitude and Perceived Behavioral Control (PBC) significantly drive BNPL adoption among Malaysian Gen Z; social influence promotes impulsive spending | Cross-sectional; no predictive modelling of financial outcomes | Validates target demographic; confirms knowledge-action gap |
| 2 | Tan et al. (2026) | High self-reported financial literacy (M = 4.07/5) but very weak correlation with debt management (r = 0.060); PBC (Beta = 0.694) is dominant predictor of BNPL usage | No longitudinal tracking; no AI intervention proposed | **Core reference** — provides key statistics for the problem statement |
| 3 | Raj et al. (2024) | BNPL intensifies materialism; reduces 'pain of payment'; promotes overconsumption and debt traps | No quantification of future cash flow impact of BNPL stacking | Supports Engine 2's role in simulating cumulative financial impact before commitment |
| 4 | Aisjah, S. (2024) | Financial self-efficacy and parenting influence BNPL adoption; low self-efficacy = higher BNPL vulnerability | Indonesian context; no predictive modelling | Confirms university students are a high-risk group for BNPL-related debt |
| 5 | Kumar et al. (2024) | BNPL drives 10% increase in basket sizes; reduces price sensitivity; effects strongest for consumers who made smaller purchases | No tracking of individual debt accumulation over time | Empirical evidence that BNPL increases spending — justifies forecasting need |
| 6 | Blue et al. (2023) | Many users experience debt cycling — using one BNPL to pay another; BNPL normalises debt and reduces awareness of cumulative obligations | Qualitative; no quantitative financial impact measurement | Confirms 'instalment stacking' problem central to Engine 1's forecasting approach |
| 7 | Relja et al. (2023) | Users underestimate BNPL risk due to 'interest-free' framing; trust and perceived usefulness drive adoption | UK-centric; no forecasting component | Supports need for AI risk classifier that objectively evaluates purchase risk |
| 8 | Cook et al. (2023) | BNPL providers frame debt as a neutral 'payment method', reducing risk perception among youth | No technological interventions proposed | Explains why explicit dashboard visualisation of future financial impact is essential |

#### B — BNPL Regulation & Market (3 papers)

| # | Author(s) & Year | Key Findings | Research Gap | PaySense Relevance |
|---|-----------------|-------------|-------------|-------------------|
| 9 | Irawati et al. (2024) | Malaysia emphasises financial inclusion; BNPL poses over-indebtedness, credit risk, and data protection challenges across ASEAN | No consumer-facing tools proposed | Highlights regulatory gap justifying consumer protection tools like PaySense |
| 10 | Filotto et al. (2024) | BNPL displacing credit cards among younger demographics globally; significant scale reached with limited regulatory oversight | Macro-level analysis; no individual-level consumer tools | Confirms urgency and scale of BNPL growth |
| 11 | Cornelli et al. (2023) | BNPL growth correlated with higher e-commerce and lower credit card access; disproportionately affects younger, lower-income consumers | Aggregate country-level data; no individual debt trajectory analysis | Confirms target demographic selection for PaySense |

#### C — Financial Literacy (4 papers)

| # | Author(s) & Year | Key Findings | Research Gap | PaySense Relevance |
|---|-----------------|-------------|-------------|-------------------|
| 12 | Cappelli et al. (2024) | Financial literacy alone insufficient to change behaviour; digital tools and behavioural interventions more effective than knowledge-only approaches | No specific technological solutions tested | Directly supports AI-driven copilot rationale over education-only interventions |
| 13 | Surwanti et al. (2024) | Gen Z shows lower financial discipline despite digital fluency; fintech adoption outpaces financial literacy development | Self-reported data; no predictive modelling | Validates need for predictive copilot bridging digital convenience and financial responsibility |
| 14 | Di Maggio et al. (2022) | BNPL users stack multiple commitments; financially fragile consumers most affected; stacking increases overall spending significantly | US-focused; no consumer forecasting tools | Foundational evidence for instalment stacking — directly motivates Engine 1 |
| 15 | Gerrans et al. (2022) | BNPL users with lower financial literacy significantly more likely to experience financial stress; minimal affordability checks in BNPL | Australian context; no AI or predictive modelling component | Directly justifies Engine 2's affordability stress-testing before purchase commitment |

#### D — Time-Series Forecasting (5 papers)

| # | Author(s) & Year | Key Findings | Research Gap | PaySense Relevance |
|---|-----------------|-------------|-------------|-------------------|
| 16 | Ning et al. (2022) | Prophet robust for automatic seasonality detection; LSTM outperforms for complex non-linear patterns; ARIMA competitive for stationary series; MAE & RMSE as primary metrics | Oil production context; performance may differ for personal finance with irregular patterns | **Primary reference** for Engine 1 model selection and evaluation metric choice |
| 17 | Arslan, S. (2022) | Hybrid Prophet-LSTM significantly outperforms standalone models; Prophet decomposes seasonality, LSTM handles non-linear residuals | Energy consumption context; computational complexity concern for real-time apps | Validates hybrid decomposition approach as potential Engine 1 enhancement |
| 18 | Khan & Ali (2023) | Prophet provides best balance of accuracy and simplicity for short-horizon financial forecasts; ARIMA effective for stationary data; deep learning offers marginal gain with short horizons | Macroeconomic indicators, not personal finance | Validates Prophet and ARIMA as candidate models; confirms 3-month horizon is best suited to Prophet |
| 19 | Wang & Li (2023) | Combined Prophet-LSTM achieves superior accuracy; Prophet's trend/seasonality decomposition provides interpretability; computationally efficient for real-time applications | Not validated on personal finance or spending data | Provides implementable architecture combining Prophet interpretability with LSTM accuracy |
| 20 | Ibrahim & Al-Shammari (2025) | ARIMA-Prophet hybrid achieves lowest error rates across multiple financial datasets; framework adaptable to different forecasting horizons | Not tested on personal finance data; tuning overhead increases with longer horizons | State-of-the-art hybrid framework applicable to Engine 1 as a potential enhancement |

#### E — Credit Risk Classification (5 papers)

| # | Author(s) & Year | Key Findings | Research Gap | PaySense Relevance |
|---|-----------------|-------------|-------------|-------------------|
| 21 | Zhu et al. (2023) | XGBoost achieves AUC > 0.85 for loan default prediction; SHAP reveals income-to-debt ratio and number of active accounts as key features; explainability does not reduce accuracy | Traditional loan default context; not BNPL-specific; no future balance forecasting as input | **Primary reference** for Engine 2 — validates XGBoost + SHAP integration |
| 22 | Chang et al. (2024) | Gradient Boosting achieves best F1-score for P2P lending; ratio-based feature engineering significantly improves credit risk classification performance | P2P lending context; no BNPL-specific features; no forecasting integration | Validates Gradient Boosting (XGBoost/LightGBM) for credit risk; ratio features transferable to Engine 2 |
| 23 | Naik, K. S. (2021) | Random Forest and XGBoost consistently outperform traditional classifiers for unsecured credit risk; repayment history and debt-to-income ratio are strongest predictors | No BNPL-specific features; no forecasting integration; no XAI layer | Benchmark for ML classifier comparison in consumer credit — directly applicable to Engine 2 selection |
| 24 | Sousa et al. (2022) | Random Forest achieves best accuracy; consumer spending patterns are stronger predictors than static demographic features | Brazilian banking context; no time-series forecasting input | Validates Random Forest as candidate; spending patterns as predictors align with Engine 2 features |
| 25 | Rahmani et al. (2025) | Gradient Boosting: accuracy = 0.89, F1 = 0.81; SMOTE significantly improves minority class (defaulter) recall; structured ML pipeline with feature engineering and tuning is essential | General lending data; not BNPL-specific; no consumer-facing risk scores | Complete reproducible ML pipeline design adaptable for Engine 2; validates SMOTE approach for class imbalance |

#### F — Explainable AI (XAI) (3 papers)

| # | Author(s) & Year | Key Findings | Research Gap | PaySense Relevance |
|---|-----------------|-------------|-------------|-------------------|
| 26 | Osei-Bonsu & Mensah (2024) | SHAP provides consistent global feature importance; LIME offers intuitive local explanations; combining both produces complementary global and local insights | Lender-focused; not consumer-facing | Directly informs XAI layer design for Engine 2's Risk Checker screen |
| 27 | Mujo et al. (2025) | SHAP values mapped to plain-language explanations improve user trust and reduce bias in automated credit decisions | Not applied to consumer-facing personal finance tools | Validates feasibility of mapping SHAP values to plain-language alerts for students |
| 28 | Azzutti et al. (2025) | SHAP and LIME most widely adopted model-agnostic XAI methods in finance; consumer-facing XAI in personal finance **explicitly identified as a research gap** | Review paper; does not implement new XAI approaches | **Positions PaySense as a novel contribution** — fills the identified consumer-facing XAI gap |

#### G — Personal Finance Apps (2 papers)

| # | Author(s) & Year | Key Findings | Research Gap | PaySense Relevance |
|---|-----------------|-------------|-------------|-------------------|
| 29 | Kamarudeen & Vijayalakshmi (2023) | ML-based expense categorisation reduces manual effort ~70%; students using the app showed improved budgeting; real-time feedback critical for engagement | Historical tracking only; no predictive or forecasting capability; no BNPL-specific risk handling | Validates ML-powered financial apps for students; PaySense extends this with prediction + BNPL risk classification |
| 30 | Georgieva et al. (2024) | Mobile-first design increases adoption; CSV/PDF import best balance of accuracy and usability; visual financial status reports improve financial awareness | Purely reactive — historical tracking only; no AI or predictive features; no BNPL or risk assessment | Represents current state-of-the-art; the limitations identified are precisely the gaps PaySense addresses |

---

### 10.3 Research Gaps Identified

Five critical research gaps have been identified through systematic review of the 30 papers:

| Gap | Description | How PaySense Addresses It |
|-----|-------------|--------------------------|
| **Gap 1 — No Predictive Consumer-Facing BNPL Tools** | All existing BNPL research documents behaviour and risk but no study proposes a consumer-facing predictive tool that simulates the financial impact of new BNPL commitments *before* purchase | Engine 1 + Engine 2 sequential architecture provides pre-purchase simulation and stress-testing |
| **Gap 2 — Forecasting Not Applied to Personal Finance** | Prophet and ARIMA have been applied to macroeconomic indicators, stock prices, and enterprise cash flows — not to individual student personal finance forecasting | Engine 1 targets individual student spending patterns with explicit BNPL instalment awareness in the balance projection |
| **Gap 3 — Credit Risk Models Not Adapted for BNPL** | Existing ML classifiers use institutional lending features; no classifier uses forecasted future balance as input or models BNPL-specific factors (instalment stacking, overlapping due dates, remaining buffer) | Engine 2 uses Engine 1's forecasted balance curve as a key feature; input features are explicitly designed around BNPL-specific risk signals |
| **Gap 4 — XAI Not Applied to Consumer-Facing Financial Tools** | SHAP/LIME extensively used in institutional credit scoring; consumer-facing XAI in personal finance explicitly identified as underexplored by Azzutti et al. (2025) | SHAP explanations translated to student-friendly plain-language alerts surfaced in the Risk Checker screen |
| **Gap 5 — No Integration of Forecasting + Classification** | No existing system feeds forecasted future balances into a BNPL risk classifier for real-time, context-aware purchase stress-testing | **Core novelty of PaySense** — sequential Engine 1 → Engine 2 architecture where the forecasted balance curve is the evaluation context for every risk decision |

> **Search terms:** `BNPL credit risk machine learning`, `credit scoring classification algorithms`, `consumer default prediction XGBoost`, `cash flow forecasting LSTM Prophet`, `peer-to-peer lending default prediction`, `personal finance time series ARIMA`, `explainable AI credit risk SHAP`, `BNPL Malaysia university students`, `financial literacy Gen Z`  
> **Sources:** Google Scholar, IEEE Xplore, ScienceDirect, Springer, Taylor & Francis

---

## 11. Project Timeline

### FYP 1 — Current Semester

| Week | Task / Milestone | Deliverable |
|------|-----------------|-------------|
| 1–2 | Topic confirmation, supervisor meeting, concept document v1 | Concept Document v1 |
| 3–4 | Literature review — 20 papers, build comparison matrix | Literature Matrix (draft) |
| 5–6 | Identify and confirm 3 candidate models per engine; identify Kaggle datasets | Model shortlist, Dataset sourced |
| 7 | Concept document v2 — finalised ML plan, tech stack, architecture | Concept Document v2 |
| 8 | Proposal Defence preparation — slides, script, architecture diagrams | Proposal Defence Slides |
| 9 | **Proposal Defence** | ✅ Proposal Defence — SUBMITTED |
| 10–11 | Interim Report writing — Introduction, Literature Review, Methodology chapters | Interim Report (draft) |
| 12 | Finalize and submit Interim Report | ✅ Interim Report — SUBMITTED |
| 13–14 | Begin data preparation — clean Kaggle dataset, generate synthetic BNPL scenarios | Prepared datasets |

### FYP 2 — Next Semester

| Week | Task / Milestone | Deliverable |
|------|-----------------|-------------|
| 1–3 | Train and evaluate 3 candidate forecasting models (ARIMA, Prophet, LSTM); compare MAE/RMSE/MAPE | Model evaluation report (Engine 1) |
| 4–6 | Train and evaluate 3 candidate classification models (RF, XGBoost, LightGBM) with SMOTE; compare F1/AUC | Model evaluation report (Engine 2) |
| 5–7 | Select best models; integrate SHAP; build FastAPI backend with all ML endpoints | FastAPI backend (complete) |
| 6–9 | Build Next.js frontend — Dashboard, Transactions, BNPL Manager, Risk Checker; connect to Supabase | Frontend (complete) |
| 8–10 | Full system integration — frontend + backend + database; CSV/PDF parsing | Integrated system |
| 10–11 | System testing, bug fixing, performance tuning; deploy to Vercel + Render | Deployed live application |
| 11–12 | Write Dissertation + Technical Paper (6 pages, IEEE format) | Dissertation draft, Technical Paper draft |
| 13 | Project Presentation preparation; final application demo | Presentation slides, Demo ready |
| 14 | **Project Presentation & Demo** | ✅ Project Presentation — SUBMITTED |
| 15 | Submit hard-bound dissertation copies and final revised technical paper | ✅ Hard-bound Dissertation — SUBMITTED |

---

## 12. References

### BNPL Consumer Behavior

1. I. Osman, N. A. M. Ariffin, M. F. N. B. Mohd Yuraimie, M. F. B. Ali, and M. A. F. B. Noor Akbar, "How Buy Now, Pay Later (BNPL) is Shaping Gen Z's Spending Spree in Malaysia," *Information Management and Business Review*, vol. 16, no. 3S(I)a, pp. 757–770, 2024.
2. W. S. Tan et al., "Buy Now Pay Later (BNPL) Usage and Its Impact on Debt Management Among Malaysian University Students," *International Journal of Accounting and Finance in Asia Pacific*, vol. 9, no. 1, pp. 90–103, 2026.
3. V. A. Raj, S. S. Jasrotia, and S. S. Rai, "Intensifying Materialism Through Buy-Now Pay-Later (BNPL): Examining the Dark Sides," *International Journal of Bank Marketing*, vol. 42, no. 1, pp. 94–112, 2024.
4. S. Aisjah, "Intention to Use Buy-Now-Pay-Later Payment System Among University Students: A Combination of Financial Parenting, Financial Self-Efficacy, and Social Media Intensity," *Cogent Social Sciences*, vol. 10, no. 1, 2306705, 2024.
5. A. Kumar, J. Salo, and R. Bezawada, "The Effects of Buy Now, Pay Later (BNPL) on Customers' Online Purchase Behavior," *Journal of Retailing*, Advance Online Publication, 2024.
6. L. Blue, L. Coglan, T. Pham, I. Lammer, R. Menner, and C. Lee, "Spend and Repeat! Young Adult's Experiences with Buy Now Pay Later Services," *Financial Planning Research Journal*, vol. 9, no. 1, pp. 1–19, 2023.
7. R. Relja, P. Ward, and A. Zhao, "Understanding the Psychological Determinants of Buy-Now-Pay-Later (BNPL) in the UK: A User Perspective," *International Journal of Bank Marketing*, vol. 42, no. 1, pp. 7–37, 2023.
8. J. Cook, K. Davies, D. Farrugia, S. Threadgold, J. Coffey, K. Senior, A. Haro, and B. Shannon, "Buy Now Pay Later Services as a Way to Pay: Credit Consumption and the Depoliticization of Debt," *Consumption Markets & Culture*, vol. 26, no. 4, pp. 245–257, 2023.

### BNPL Regulation & Market

9. L. Irawati, M. Z. Hamzah, and E. Sofilda, "Regulating Buy Now Pay Later (BNPL) in ASEAN: A Comparative Analysis on Regulatory Challenges and Opportunities," *International Journal of Economics, Management and Accounting*, vol. 1, no. 4, pp. 60–81, 2024.
10. U. Filotto, D. Salerno, G. Sampagnaro, and G. P. Stella, "Riding the Wave of Change: Buy Now, Pay Later as a Disruptive Threat to Payment Cards in the Global Market," *Research in International Business and Finance*, vol. 72, 102499, 2024.
11. G. Cornelli, L. Gambacorta, and L. Pancotto, "Buy Now, Pay Later: A Cross-Country Analysis," *BIS Quarterly Review*, Bank for International Settlements, 2023.

### Financial Literacy

12. T. Cappelli, A. P. Banks, and B. Gardner, "Understanding Money-Management Behaviour and Its Potential Determinants Among Undergraduate Students: A Scoping Review," *PLOS ONE*, vol. 19, no. 8, e0307137, 2024.
13. A. Surwanti, M. Maulidah, Wihandaru, R. Kusumawati, and F. Santi, "Financial Management Behaviour Z Generation," *E3S Web of Conferences*, vol. 571, 03003, 2024.
14. M. Di Maggio, E. Williams, and J. Katz, "Buy Now, Pay Later Credit: User Characteristics and Effects on Spending Patterns," NBER Working Paper No. 30508, National Bureau of Economic Research, 2022.
15. P. Gerrans, D. G. Baur, and S. Lavagna-Slater, "Fintech and Responsibility: Buy-Now-Pay-Later Arrangements," *Australian Journal of Management*, vol. 48, no. 4, pp. 772–793, 2022.

### Time-Series Forecasting

16. Y. Ning, H. Kazemi, and P. Tahmasebi, "A Comparative Machine Learning Study for Time Series Oil Production Forecasting: ARIMA, LSTM, and Prophet," *Computers & Geosciences*, vol. 164, 105126, 2022.
17. S. Arslan, "A Hybrid Forecasting Model Using LSTM and Prophet for Energy Consumption with Decomposition of Time Series Data," *PeerJ Computer Science*, vol. 8, e1001, 2022.
18. S. N. Khan and A. Ali, "Comparative Analysis of Advanced Time Series Forecasting Techniques: Evaluating the Accuracy of ARIMA, Prophet, and Deep Learning Models for Predicting Financial Indicators," *Advances in Deep Learning Techniques*, vol. 3, no. 1, pp. 52–98, 2023.
19. Z. Wang and H. Li, "Time-Series Prediction Research Based on Combined Prophet-LSTM Model," in *Proc. IEEE 2nd International Conference on Electrical Engineering, Big Data and Algorithms (EEBDA)*, 2023, pp. 1395–1399.
20. M. Z. Ibrahim and E. T. Al-Shammari, "A Hybrid Approach to Time Series Forecasting: Integrating ARIMA and Prophet for Improved Accuracy," *Ain Shams Engineering Journal*, Article in Press, 2025.

### Credit Risk Classification

21. X. Zhu, Q. Chu, X. Song, P. Hu, and I. L. Peng, "Explainable Prediction of Loan Default Based on Machine Learning Models," *Data Science and Management*, vol. 6, pp. 123–133, 2023.
22. A. H. Chang et al., "Performance Evaluation of Hybrid Machine Learning Algorithms for Online Lending Credit Risk Prediction," *Applied Artificial Intelligence*, vol. 38, no. 1, 2358661, 2024.
23. K. S. Naik, "Predicting Credit Risk for Unsecured Lending: A Machine Learning Approach," arXiv preprint, arXiv:2110.02206, 2021.
24. M. R. Sousa, J. Gama, and E. Brandão, "Machine Learning Predictivity Applied to Consumer Creditworthiness," *Computational Economics*, vol. 59, pp. 1523–1557, 2022.
25. R. Rahmani, M. Parola, and M. G. C. A. Cimino, "Data-Driven Loan Default Prediction: A Machine Learning Approach for Enhancing Business Process Management," *Systems (MDPI)*, vol. 13, no. 7, 581, 2025.

### Explainable AI (XAI)

26. A. Osei-Bonsu and F. Mensah, "Explainable AI for Credit Risk Assessment: A Data-Driven Approach to Transparent Lending Decisions," *Journal of Economics, Finance and Accounting Studies*, vol. 6, no. 5, pp. 01–12, 2024.
27. A. Mujo, S. Nikolla, E. Hoxha, and E. Pelivani, "Explainable AI in Credit Scoring: Improving Transparency in Loan Decisions," *Journal of Information Systems Engineering and Management*, vol. 10, no. 3, 4437, 2025.
28. A. Azzutti et al., "Model-Agnostic Explainable Artificial Intelligence Methods in Finance: A Systematic Review, Recent Developments, Limitations, Challenges and Future Directions," *Artificial Intelligence Review*, vol. 58, 270, 2025.

### Personal Finance Apps

29. M. Kamarudeen and K. Vijayalakshmi, "Machine Learning Based Financial Management Mobile Application to Enhance College Students' Financial Literacy," in *Proc. ICRES 2023 — International Conference on Research in Education and Science*, 2023, pp. 1237–1253.
30. D. Georgieva, G. Tsochev, and G. Angelov, "Personal Finance Management Application," *TEM Journal*, vol. 13, no. 3, pp. 2066–2075, 2024.

---

*PaySense: Gen-Z Financial Copilot • Nur Farah Binti Ahmad Nazri • 22007916 • Universiti Teknologi PETRONAS*
