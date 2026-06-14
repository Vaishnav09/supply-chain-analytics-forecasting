# Supply Chain Analytics

End-to-end analytics pipeline on the DataCo Global Supply Chain dataset — covering data cleaning, exploratory analysis, and weekly demand forecasting using Facebook Prophet.

## Dataset

| Attribute | Value |
|---|---|
| Source | DataCo Global Supply Chain (public) |
| Raw records | 180,519 rows × 53 columns |
| Date range | Jan 2015 – Jan 2018 |
| Markets | Pacific Asia, USCA, Europe, LATAM, Africa |
| Products | 118 products across 50 categories |
| Customer segments | Consumer, Corporate, Home Office |
| Avg order value | $203.77 |
| Late delivery rate | 54.8% |

---

## Project Structure

```
supply-chain-analytics/
├── data/
│   ├── raw/                    # Original DataCo CSV
│   └── processed/              # Cleaned dataset
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_Demand_Forecasting.ipynb
├── src/
│   └── preprocessing.py        # Reusable cleaning functions
└── requirements.txt
```

---

## Notebooks

### 01 — Data Cleaning
- Removed PII columns (email, password, name, address)
- Parsed `order_date` and `shipping_date` to datetime
- Standardised column names (snake_case)
- Engineered features: `shipping_delay`, `is_late`, `order_month`
- Output: `data/processed/clean_supply_chain.csv` (180,519 rows × 48 columns, 0 nulls)

### 02 — Exploratory Data Analysis
- Delivery status distribution and late delivery breakdown by market and shipping mode
- Revenue and profit trends over time
- Category and product-level sales analysis
- Shipping delay distribution across modes

### 03 — Demand Forecasting
- Aggregated order quantity to weekly time series (162 weeks)
- Trained Facebook Prophet on 150 weeks, evaluated on 12-week holdout
- Benchmarked against naive (last-week) and rolling 4-week average baselines

| Model | SMAPE |
|---|---|
| Naive (last week) | 6.88% |
| Rolling avg (4-week) | 6.78% |
| **Prophet (this model)** | **2.80%** |

**59% improvement over naive baseline**

Additional metrics on test set:
- MAE: 33.19 orders
- RMSE: 41.64 orders
- Avg weekly demand: 1,185 orders (model off by ~2.8% on average)

> **Note:** The DataCo dataset contains a synthetic data artifact from Oct 2017 onward where weekly order counts lock to an alternating 479/480 pattern. Evaluation metrics reflect this period. Re-evaluation on the 2015–Sep 2017 window (natural variation, std ~87 orders/week) is recommended for a more robust benchmark.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## Stack

`pandas` · `numpy` · `prophet` · `scikit-learn` · `plotly` · `matplotlib` · `seaborn`
