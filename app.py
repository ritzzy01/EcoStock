import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load inventory CSV
df = pd.read_csv("mock_inventory.csv")

st.set_page_config(page_title="AI-Powered Inventory", layout="wide")
st.title("🧠 AI-Powered Inventory Waste Reduction")

# Preprocessing
df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
df['DaysToExpire'] = (df['ExpiryDate'] - datetime.today()).dt.days

# Define features and target
X = df[['Category', 'StoreID', 'Weather', 'HolidayFlag']]
y = df['WeeklySales']

# ML Pipeline: Encode categorical and fit regression
categorical_features = ['Category', 'StoreID', 'Weather']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X, y)

# Predict demand
df['PredictedDemand'] = model.predict(X).round(2)

# 🟡 Risk Level Calculation: HIGH, MEDIUM, LOW
conditions = [
    # High Risk: Expiring soon AND significantly overstocked
    (df['PredictedDemand'] < 0.7 * df['StockQty']) & (df['DaysToExpire'] < 5),

    # Medium Risk: Moderate overstock OR expires in 5–7 days
    (df['PredictedDemand'] < 0.9 * df['StockQty']) | ((df['DaysToExpire'] >= 5) & (df['DaysToExpire'] < 8))
]

choices = ['HIGH', 'MEDIUM']
df['RiskLevel'] = np.select(conditions, choices, default='LOW')


# 📦 Inventory Table
st.subheader("📦 Inventory Overview")
st.dataframe(df[['Product', 'Category', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']])

# 🚨 At-Risk Items (HIGH + MEDIUM)
st.subheader("🚨 At-Risk Products")
at_risk = df[df['RiskLevel'].isin(['HIGH', 'MEDIUM'])]
st.dataframe(at_risk[['Product', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']])


# ✅ Suggested Actions
st.subheader("✅ Suggested Actions")
for _, row in at_risk.iterrows():
    st.markdown(f"- **{row['Product']}**: Predicted demand is {row['PredictedDemand']} but stock is {row['StockQty']} and expires in {row['DaysToExpire']} days. _Consider markdowns, bundling or reduced reorders._")

# 📊 Visualization
st.subheader("📊 Weekly Sales vs Predicted Demand")
chart_data = df[['Product', 'WeeklySales', 'PredictedDemand']].set_index('Product')
st.bar_chart(chart_data)

# Footer
st.caption("Built for Walmart Sparkathon 2025 | Team: Ritika & Nikhil 💡")
