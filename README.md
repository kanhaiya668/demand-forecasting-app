# 📈 Store Demand Forecasting System


🔗 **[Live Demo](https://demand-forecasting-app-xyzzz123.streamlit.app/)**

An end-to-end machine learning system that forecasts daily retail sales for 1,115 stores up to 6 weeks in advance, built on the Kaggle Rossmann Store Sales dataset. Includes feature engineering, a LightGBM forecasting model, and an interactive Streamlit dashboard for store-level, multi-day sales predictions.

## 🎯 Problem Statement

Retail store managers need to forecast daily sales in advance to plan staffing, inventory, and promotions. This project builds a demand forecasting pipeline that predicts future daily sales per store, accounting for promotions, holidays, seasonality, and competition.

## 📊 Dataset

- **Source:** [Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
- **Size:** 1,017,209 records across 1,115 stores (Jan 2013 – Jul 2015)
- **Features:** Store type, assortment, promotions, competition distance, holidays, and historical sales

## 🔧 Approach

1. **Data Cleaning:** Merged store metadata with sales data, handled missing values (competition dates, promo intervals)
2. **Feature Engineering:**
   - Date-based features (Year, Month, Day, WeekOfYear)
   - Competition age (`CompetitionOpen`) and active promo flag (`IsPromo2Active`)
   - Lag features (`Sales_Lag_1/7/14`) and rolling averages (`Sales_RollingMean_7/30`)
3. **Modeling:** LightGBM regression with time-based train-test split (no random split, to avoid lookahead bias)
4. **Evaluation:** RMSE, MAE, and MAPE on a held-out final 6-week test period

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 784.63 |
| MAE | 550.84 |
| MAPE | 8.32% |

**Top predictive features:** Sales_Lag_1, Sales_RollingMean_30, Promo, DayOfWeek

## 🖥️ App Features

- Select any store and forecast horizon (1–42 days)
- Toggle promo assumption for what-if scenarios
- Recursive multi-day forecasting (each day's prediction feeds into the next)
- Interactive chart, summary metrics, and downloadable CSV forecast

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** LightGBM, scikit-learn, pandas, numpy
- **App:** Streamlit
- **Visualization:** Matplotlib

## 🚀 Running Locally

```bash
git clone https://github.com/kanhaiya668/demand-forecasting-app.git
cd demand-forecasting-app
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure