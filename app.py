import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")


st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #FF4B4B;
        font-weight: 800;
        padding-bottom: 0px;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #FF4B4B;
    }
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #D63838;
        color: white;
    }
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #8B949E;
    }
    [data-testid="stMetricValue"] {
        color: #58D68D;
        font-weight: 700;
    }
    .stDataFrame {
        border: 1px solid #30363D;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load('model/demand_forecast_model.pkl')
    feature_cols = joblib.load('model/feature_cols.pkl')
    cat_features = joblib.load('model/cat_features.pkl')
    store_latest = pd.read_csv('model/store_latest_info.csv')
    return model, feature_cols, cat_features, store_latest

model, feature_cols, cat_features, store_latest = load_artifacts()

st.title("📈 Store Demand Forecasting System")
st.write("Select a store and forecast horizon to predict future daily sales.")

st.sidebar.header("Forecast Settings")

store_list = sorted(store_latest['Store'].unique())
selected_store = st.sidebar.selectbox("Select Store", store_list)

horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=1, max_value=42, value=14)

promo_assumption = st.sidebar.radio(
    "Assume Promo during forecast period?",
    options=["No Promo", "Promo Active"]
)

forecast_btn = st.sidebar.button("Generate Forecast")


def generate_forecast(store_id, horizon_days, promo_flag):
    store_info = store_latest[store_latest['Store'] == store_id].iloc[0].to_dict()

    last_date = pd.Timestamp('2015-07-31')
    lag_1 = store_info['Sales_Lag_1']
    lag_7_history = [store_info['Sales_Lag_7'], store_info['Sales_Lag_14']]
    rolling_7_history = [store_info['Sales_RollingMean_7']] * 7
    rolling_30_history = [store_info['Sales_RollingMean_30']] * 30

    results = []

    for i in range(1, horizon_days + 1):
        future_date = last_date + timedelta(days=i)

        row = {
            'Store': store_id,
            'DayOfWeek': future_date.dayofweek + 1,
            'Promo': 1 if promo_flag == "Promo Active" else 0,
            'StateHoliday': '0',
            'SchoolHoliday': 0,
            'StoreType': store_info['StoreType'],
            'Assortment': store_info['Assortment'],
            'CompetitionDistance': store_info['CompetitionDistance'],
            'Promo2': store_info['Promo2'],
            'Year': future_date.year,
            'Month': future_date.month,
            'Day': future_date.day,
            'WeekOfYear': future_date.isocalendar()[1],
            'CompetitionOpen': 100,
            'IsPromo2Active': 0,
            'Sales_Lag_1': lag_1,
            'Sales_Lag_7': lag_7_history[-7] if len(lag_7_history) >= 7 else lag_7_history[0],
            'Sales_Lag_14': lag_7_history[0],
            'Sales_RollingMean_7': np.mean(rolling_7_history[-7:]),
            'Sales_RollingMean_30': np.mean(rolling_30_history[-30:])
        }

        X_pred = pd.DataFrame([row])[feature_cols]
        for col in cat_features:
            X_pred[col] = X_pred[col].astype('category')

        pred_sales = model.predict(X_pred)[0]
        pred_sales = max(0, pred_sales)

        results.append({'Date': future_date, 'Predicted_Sales': pred_sales})

        lag_1 = pred_sales
        lag_7_history.append(pred_sales)
        rolling_7_history.append(pred_sales)
        rolling_30_history.append(pred_sales)

    return pd.DataFrame(results)


if forecast_btn:
    with st.spinner("Generating forecast..."):
        forecast_df = generate_forecast(selected_store, horizon, promo_assumption)

    st.success(f"Forecast generated for Store {selected_store} — next {horizon} days")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Forecast", f"€{forecast_df['Predicted_Sales'].mean():,.0f}")
    col2.metric("Total Forecasted Sales", f"€{forecast_df['Predicted_Sales'].sum():,.0f}")
    col3.metric("Peak Day", forecast_df.loc[forecast_df['Predicted_Sales'].idxmax(), 'Date'].strftime('%d %b'))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], marker='o', color='#FF4B4B')
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Sales (€)")
    ax.set_title(f"Store {selected_store} — {horizon}-Day Sales Forecast")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Detailed Forecast Table")
    display_df = forecast_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%d %b %Y (%a)')
    display_df['Predicted_Sales'] = display_df['Predicted_Sales'].round(0).astype(int)
    st.dataframe(display_df, use_container_width=True)

    if horizon > 14:
        st.info("📌 Note: Predictions further out (beyond ~2 weeks) carry more uncertainty since each day's forecast builds on the previous predicted day, not actual data.")

    csv = forecast_df.to_csv(index=False)
    st.download_button("Download Forecast as CSV", csv, f"forecast_store_{selected_store}.csv", "text/csv")