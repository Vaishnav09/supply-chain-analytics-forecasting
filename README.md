# Supply Chain Analytics & Demand Forecasting

End-to-end analytics and ML pipeline on the DataCo Global Supply Chain dataset — covering data cleaning, exploratory analysis, XGBoost demand forecasting, a live Streamlit dashboard, and a Tableau analytics workbook.

## Live Demo

| Tool | Link |
|---|---|
| Streamlit App | https://supply-chain-analytics-forecasting-ryz8appvtlnhr8dpgsrtzyc.streamlit.app/ |
| Tableau Dashboard | https://public.tableau.com/shared/KFRK3DXZW?:display_count=n&:origin=viz_share_link |

---

## Dataset

| Attribute | Value |
|---|---|
| Source | DataCo Global Supply Chain (public) |
| Raw records | 180,519 rows × 53 columns |
| Date range | Jan 2015 – Jan 2018 |
| Clean window | Dec 2014 – Sep 2017 (144 weeks) |
| Markets | Pacific Asia, USCA, Europe, LATAM, Africa |
| Products | 118 products across 50 categories |
| Customer segments | Consumer, Corporate, Home Office |
| Late delivery rate | 54.8% |

> The dataset contains a synthetic artifact from Oct 2, 2017 onward where weekly demand locks to an alternating 479/480 pattern. All modeling uses data strictly before this cutoff.

---

## Project Structure

```
supply-chain-analytics/
├── app.py                          # Streamlit dashboard (deployed)
├── requirements.txt
├── data/
│   ├── raw/                        # Original DataCo CSV
│   └── processed/                  # Cleaned dataset + Tableau exports
├── models/
│   ├── xgb_demand.pkl              # Trained XGBoost model
│   ├── metadata.json               # Training metrics and config
│   └── training_series.csv         # 144-week weekly demand series
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_Demand_Forecasting.ipynb
└── src/
    ├── train.py                    # Model training script
    └── forecast.py                 # Inference and safety stock
```

---

## Notebooks

### 01 — Data Cleaning
- Removed PII columns (email, password, name, address)
- Parsed `order_date` and `shipping_date` to datetime
- Standardised column names to snake_case
- Engineered features: `shipping_delay`, `is_late`, `order_month`
- Output: `clean_supply_chain.csv` (180,519 rows × 48 columns, 0 nulls)

### 02 — Exploratory Data Analysis
- Monthly order volume and revenue trends (2015–2017)
- Late delivery rate: 54.8% overall, worst on Standard Class shipping
- Revenue breakdown by market: Europe leads at $10.87M
- Top categories by revenue: Fishing, Cleats, Camping & Hiking
- Profit ratio distribution by category and segment

### 03 — Demand Forecasting
- Aggregated order quantity to weekly time series (144 clean weeks)
- Detected and excluded synthetic data artifact (Oct 2017 onward)
- Feature engineering: lag features (1, 2, 4, 12, 52 weeks), rolling means (4w, 12w), week-of-year
- Walk-forward holdout evaluation on last 12 weeks

| Model | MAE | RMSE | SMAPE |
|---|---|---|---|
| Naive (last week) | — | — | ~6.9% |
| Rolling avg (4-week) | — | — | ~6.8% |
| Holt-Winters ETS | — | — | ~5.2% |
| **XGBoost (production)** | **86.26** | **111.02** | **3.47%** |

XGBoost delivers a **50% improvement in SMAPE** over the naive baseline.

---

## Streamlit App

Interactive demand planning dashboard with:
- 12-week recursive XGBoost forecast
- Adjustable service level (90–99%) for safety stock calculation
- Forecast accuracy tracking (On Track / Watch / Miss)
- Revenue, delivery, and market KPIs

```bash
streamlit run app.py
```

---

## Tableau Dashboard

Two-dashboard Tableau workbook published on Tableau Public:

**Supply Chain Overview** — KPIs, monthly revenue trend, revenue by market, late delivery rate, top 10 product categories

**Operations Deep Dive** — Late delivery breakdown by shipping mode and market, profit ratio by category, customer segment revenue

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Retrain model (optional)
python src/train.py

# Run dashboard
streamlit run app.py
```

---

## Stack

`pandas` · `numpy` · `xgboost` · `scikit-learn` · `scipy` · `statsmodels` · `plotly` · `streamlit` · `Tableau Public`
