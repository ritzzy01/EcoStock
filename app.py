import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------------- CSS STYLING ---------------------------
st.set_page_config(page_title="EcoStock AI", layout="wide")
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            color: #eaeaea;
        }
        h1, h2, h3, h4 {
            color: #ffffff;
        }
        .stApp {
            background-color: #121212;
            padding: 1rem;
        }
        .block-container {
            padding: 2rem 1.5rem;
        }
        .suggestion-card {
            background-color: #1e1e1e;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        }
        .suggestion-card:hover {
            background-color: #252525;
        }
        .risk-high {
            color: #e76f51;
        }
        .risk-medium {
            color: #f4a261;
        }
        .risk-low {
            color: #2a9d8f;
        }
        .metric-label {
            font-size: 14px;
            color: #aaaaaa;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
        }
        .footer {
            text-align: center;
            margin-top: 4rem;
            color: #888888;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------- LOAD DATA ---------------------------
df = pd.read_csv("mock_inventory.csv")
df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
df['DaysToExpire'] = (df['ExpiryDate'] - datetime.today()).dt.days

# --------------------------- MODEL ---------------------------
X = df[['Category', 'StoreID', 'Weather', 'HolidayFlag']]
y = df['WeeklySales']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'StoreID', 'Weather'])],
    remainder='passthrough'
)

model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
model.fit(X, y)
df['PredictedDemand'] = model.predict(X).round(2)

# --------------------------- RISK LEVEL ---------------------------
conditions = [
    (df['PredictedDemand'] < 0.7 * df['StockQty']) & (df['DaysToExpire'] < 5),
    (df['PredictedDemand'] < 0.9 * df['StockQty']) | ((df['DaysToExpire'] >= 5) & (df['DaysToExpire'] < 8))
]
choices = ['HIGH', 'MEDIUM']
df['RiskLevel'] = np.select(conditions, choices, default='LOW')

# --------------------------- SIDEBAR FILTER ---------------------------
with st.sidebar:
    st.markdown("### 🔍 Filter Inventory")
    selected_category = st.multiselect("Select Category", options=df['Category'].unique(), default=df['Category'].unique())

filtered_df = df[df['Category'].isin(selected_category)] if selected_category else df
at_risk = filtered_df[filtered_df['RiskLevel'].isin(['HIGH', 'MEDIUM'])]

# --------------------------- HEADER ---------------------------
st.markdown("<h1>🌿 EcoStock AI</h1>", unsafe_allow_html=True)
st.markdown("##### Smart Inventory Optimization for Retail")
st.markdown("---")

# --------------------------- KPI METRICS ---------------------------
col1, col2, col3 = st.columns(3)
col1.markdown("<div class='metric-label'>📦 Total Products</div>", unsafe_allow_html=True)
col1.markdown(f"<div class='metric-value'>{len(filtered_df)}</div>", unsafe_allow_html=True)

col2.markdown("<div class='metric-label'>⚠️ High Risk</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-value'>{(filtered_df['RiskLevel'] == 'HIGH').sum()}</div>", unsafe_allow_html=True)

col3.markdown("<div class='metric-label'>🟡 Medium Risk</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-value'>{(filtered_df['RiskLevel'] == 'MEDIUM').sum()}</div>", unsafe_allow_html=True)

# --------------------------- INVENTORY TABLE ---------------------------
st.markdown("### 📦 Inventory Overview")
st.dataframe(filtered_df[['Product', 'Category', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']],
             use_container_width=True, height=350)

# --------------------------- RISK TABLE ---------------------------
st.markdown("### 🚨 At-Risk Inventory")
if not at_risk.empty:
    st.dataframe(at_risk[['Product', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']],
                 use_container_width=True)
else:
    st.success("🎉 No high or medium risk items currently.")

# --------------------------- SUGGESTIONS ---------------------------
st.markdown("### ✅ Actionable Suggestions")
if not at_risk.empty:
    for _, row in at_risk.iterrows():
        risk_class = {
            'HIGH': 'risk-high',
            'MEDIUM': 'risk-medium',
            'LOW': 'risk-low'
        }.get(row['RiskLevel'], 'risk-low')
        
        st.markdown(
            f"""
            <div class='suggestion-card'>
                <div style='font-size: 18px; font-weight: bold;'>{row['Product']}</div>
                <div class='{risk_class}'>Risk Level: {row['RiskLevel']}</div>
                <div style='margin-top: 5px;'>
                    Category: <b>{row['Category']}</b> | Store: <b>{row['StoreID']}</b><br>
                    Predicted Demand: <b>{row['PredictedDemand']}</b> | Stock: <b>{row['StockQty']}</b> | Expiry in: <b>{row['DaysToExpire']} days</b>
                </div>
                <div style='margin-top: 8px; color: #2a9d8f;'>
                    💡 Suggest: Consider <b>discounting</b>, <b>bundling</b>, or <b>adjusting reorder volume</b>.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
else:
    st.info("No actionable suggestions to show.")

# --------------------------- CHARTS ---------------------------
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📊 Weekly Sales vs Predicted Demand")
    st.bar_chart(filtered_df[['Product', 'WeeklySales', 'PredictedDemand']].set_index('Product'))

with col5:
    st.markdown("### 📈 Risk Distribution")
    risk_counts = filtered_df['RiskLevel'].value_counts()
    st.bar_chart(risk_counts)

# --------------------------- FOOTER ---------------------------
st.markdown("<div class='footer'>Built by <b>Ritika & Nikhil</b> for Walmart Sparkathon 2025 💡<br>GitHub: <a href='https://github.com/Nikhil020Yadav/EcoStock' style='color: #888;' target='_blank'>EcoStock</a></div>", unsafe_allow_html=True)
