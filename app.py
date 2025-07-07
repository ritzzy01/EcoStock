import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load CSV
df = pd.read_csv("mock_inventory.csv")

# Preprocessing
df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
df['DaysToExpire'] = (df['ExpiryDate'] - datetime.today()).dt.days

# Machine Learning model: Predict Weekly Demand
X = df[['Category', 'StoreID', 'Weather', 'HolidayFlag']]
y = df['WeeklySales']

categorical_features = ['Category', 'StoreID', 'Weather']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X, y)
df['PredictedDemand'] = model.predict(X).round(2)

# Risk Levels
conditions = [
    (df['PredictedDemand'] < 0.7 * df['StockQty']) & (df['DaysToExpire'] < 5),
    (df['PredictedDemand'] < 0.9 * df['StockQty']) | ((df['DaysToExpire'] >= 5) & (df['DaysToExpire'] < 8))
]
choices = ['HIGH ⚠️', 'MEDIUM 🟡']
df['RiskLevel'] = np.select(conditions, choices, default='LOW ✅')

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="EcoStock AI", layout="wide")
st.title("🌿 EcoStock AI – Smart Inventory Optimization")

st.markdown("Optimize your retail inventory by reducing waste and restocking smartly using machine learning insights.")

# Sidebar: Only Category Filter (Store filter removed)
with st.sidebar:
    st.header("🔍 Filters")
    selected_category = st.multiselect("Filter by Category", options=df['Category'].unique(), default=df['Category'].unique())

# Apply category filter
filtered_df = df[df["Category"].isin(selected_category)] if selected_category else df

# Inventory Overview
st.subheader("📦 Current Inventory Summary")
st.dataframe(filtered_df[['Product', 'Category', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']],
             use_container_width=True, height=350)

# Risk Analysis
st.subheader("🚨 Risk Analysis")
at_risk = filtered_df[filtered_df['RiskLevel'].isin(['HIGH ⚠️', 'MEDIUM 🟡'])]
st.dataframe(at_risk[['Product', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']],
             use_container_width=True)

# Actionable Suggestions
st.subheader("✅ Actionable Suggestions")
for _, row in at_risk.iterrows():
    st.markdown(
        f"- **{row['Product']}** ({row['Category']}, Store {row['StoreID']}): _Predicted demand {row['PredictedDemand']}_ "
        f"is lower than stock **{row['StockQty']}**, expiring in **{row['DaysToExpire']} days**. "
        f"Consider **discounting**, **bundling**, or **reducing reorder volume**."
    )

# Charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Predicted Demand vs Weekly Sales")
    bar_data = filtered_df[['Product', 'WeeklySales', 'PredictedDemand']].set_index('Product')
    st.bar_chart(bar_data)

with col2:
    st.subheader("📈 Expiry Risk Distribution")
    st.bar_chart(filtered_df['RiskLevel'].value_counts())

# Footer
st.markdown("---")
st.caption("Built by Ritika & Nikhil for Walmart Sparkathon 2025 💡 | GitHub: [EcoStock](https://github.com/Nikhil020Yadav/EcoStock)")
