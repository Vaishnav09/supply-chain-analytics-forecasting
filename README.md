# Demand Forecasting for a Global Supply Chain

**The problem:** A global retailer ships 180,000+ orders across 5 markets. Over half arrive late. No one knows what demand looks like next week.

**What I built:** An end-to-end ML pipeline that cleans the data, identifies why late deliveries happen, and forecasts the next 12 weeks of demand — deployed as a live web app.

[Live App](https://supply-chain-analytics-forecasting-ryz8appvtlnhr8dpgsrtzyc.streamlit.app/) · [Tableau Dashboard](https://public.tableau.com/shared/KFRK3DXZW?:display_count=n&:origin=viz_share_link) · [Dataset: DataCo Global Supply Chain](https://data.mendeley.com/datasets/8gx2fvg2k6/5)

---

## The Dataset

DataCo Global's order history: 180,519 transactions from January 2015 to January 2018, spanning 5 markets (Europe, LATAM, Pacific Asia, USCA, Africa), 118 products, and 3 customer segments.

The first thing I noticed: **the data has a hidden artifact.** From October 2, 2017 onward, weekly order quantities lock to an alternating 479/480 pattern — a dead giveaway that the data was synthetically extended. Every model trained on this tail would learn a pattern that doesn't exist in the real world.

I cut the training window at September 25, 2017, leaving 144 weeks of genuine demand signal.

---

## What the Data Revealed

Before building any model, I spent time understanding the operational picture:

- **54.8% of orders arrive late** — not a rounding error, a structural problem
- **Standard Class shipping** accounts for the majority of late deliveries by volume, but **First Class** has a disproportionately high late rate relative to its order count
- **Europe leads in revenue** at $10.87M, nearly double Africa's $2.29M
- **Fishing and Cleats** are the highest-revenue categories, but **Golf Bags & Carts** has the highest profit ratio at ~19%
- **Consumer segment** drives $19.1M — nearly double Corporate's $11.17M

These aren't just numbers — they tell a story about where margin is being lost and which markets and shipping modes need operational attention.

---

## The Forecasting Problem

Weekly demand is noisy. It doesn't follow a clean seasonal pattern — the ACF shows significant lags at 1, 5, and 6 weeks, not the textbook lag-52 annual seasonality you'd expect. The ADF test confirmed the series is non-stationary.

I tested four approaches:

| Model | SMAPE | Notes |
|---|---|---|
| Naive (last week) | ~6.9% | Baseline — just repeat last week |
| Rolling 4-week average | ~6.8% | Slightly smoother, not much better |
| Holt-Winters ETS | ~5.2% | Handles trend and seasonality |
| **XGBoost** | **3.47%** | Recursive multi-step with lag features |

XGBoost won by a significant margin. The key was feature engineering: lag values at 1, 2, 4, 12, and 52 weeks, 4-week and 12-week rolling means, and week-of-year. These gave the model the context to learn both short-term momentum and longer-term patterns.

For production forecasting, I used recursive multi-step prediction — each week's forecast feeds into the lag features of the next week. This is how you'd actually deploy a forecasting model, not just evaluate it on a static holdout.

**Holdout results (last 12 weeks):**
- MAE: 86.26 units
- RMSE: 111.02 units
- SMAPE: 3.47%

The model is off by ~3.5% on average. For a demand planner deciding how much inventory to order, that's a useful signal.

---

## What I Built Around the Model

The model artifact alone isn't useful. I built two things on top of it:

**Streamlit App** — A live demand planning tool where you can adjust the service level (90–99%) and forecast horizon (4–12 weeks). Safety stock is calculated as `z × RMSE`, where z is the service level z-score. The app tracks forecast accuracy weekly with On Track / Watch / Miss labels so a planner can see if the model is drifting.

**Tableau Dashboard** — Two dashboards: one for supply chain operations (revenue trends, late delivery breakdown, category performance) and one for operations deep dive (late deliveries by market and shipping mode, profit ratios, segment revenue). Designed for business stakeholders who need insight without touching the model.

---

## Project Structure

```
├── app.py                      # Streamlit app (deployed)
├── src/
│   ├── train.py                # Training pipeline
│   └── forecast.py             # Recursive inference + safety stock
├── models/
│   ├── xgb_demand.pkl          # Trained model artifact
│   ├── metadata.json           # Metrics, config, last training date
│   └── training_series.csv     # 144-week demand series
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_Demand_Forecasting.ipynb
└── data/
    ├── raw/                    # Original DataCo CSV
    └── processed/              # Cleaned dataset + Tableau exports
```

---

## Reproducing the Results

```bash
git clone https://github.com/Vaishnav09/supply-chain-analytics-forecasting.git
cd supply-chain-analytics-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Retrain the model
python src/train.py

# Run the app locally
streamlit run app.py
```

---

## Stack

Python · XGBoost · scikit-learn · pandas · NumPy · SciPy · Plotly · Streamlit · Tableau
