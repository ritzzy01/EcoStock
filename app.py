import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------------- PAGE CONFIG & STYLE ---------------------------
st.set_page_config(page_title="EcoStock AI", layout="wide")
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            color: #eaeaea;
        }
        .stApp {
            background-color: #101010;
        }
        h1, h2, h3 {
            color: #ffffff;
            margin-bottom: 0.3rem;
        }
        .suggestion-card {
            background-color: #1a1a1a;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4);
            transition: transform 0.2s ease;
        }
        .suggestion-card:hover {
            transform: translateY(-2px);
        }
        .risk-high {
            color: #ff6b6b;
            font-weight: bold;
        }
        .risk-medium {
            color: #ffb347;
            font-weight: bold;
        }
        .risk-low {
            color: #4cd137;
            font-weight: bold;
        }
        .metric-label {
            font-size: 14px;
            color: #cccccc;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 600;
            color: #ffffff;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #888888;
            font-size: 14px;
        }
        .sidebar-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: #f4f4f4;
        }
        .sidebar-section {
            border-bottom: 1px solid #333;
            padding-bottom: 1rem;
            margin-bottom: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------- FUNCTION DEFINITIONS ---------------------------
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("mock_inventory.csv")
    df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
    df['DaysToExpire'] = (df['ExpiryDate'] - datetime.today()).dt.days
    return df

def train_model(df):
    X = df[['Category', 'StoreID', 'Weather', 'HolidayFlag']]
    y = df['WeeklySales']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'StoreID', 'Weather'])],
        remainder='passthrough'
    )
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model.fit(X, y)
    return model

def apply_predictions(df, model):
    X = df[['Category', 'StoreID', 'Weather', 'HolidayFlag']]
    df['PredictedDemand'] = model.predict(X).round(2)
    return df

def classify_risk(df):
    conditions = [
        (df['PredictedDemand'] < 0.7 * df['StockQty']) & (df['DaysToExpire'] < 5),
        (df['PredictedDemand'] < 0.9 * df['StockQty']) | ((df['DaysToExpire'] >= 5) & (df['DaysToExpire'] < 8))
    ]
    choices = ['HIGH','MEDIUM']
    df['RiskLevel'] = np.select(conditions, choices, default='LOW')
    return df

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-section'><div class='sidebar-title'>📥 Upload or Add Inventory</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><div class='sidebar-title'>✍️ Add New Product</div>", unsafe_allow_html=True)
    with st.form("inventory_form", clear_on_submit=True):
        product = st.text_input("Product Name")
        category = st.selectbox("Category", ["Dairy", "Bakery", "Beverages", "Fruits", "Packaged", "Snacks", "Condiments"])
        stock_qty = st.number_input("Stock Quantity", min_value=0, step=1)
        weekly_sales = st.number_input("Weekly Sales", min_value=0.0, step=1.0)
        expiry_date = st.date_input("Expiry Date", min_value=datetime.today())
        store_id = st.selectbox("Store ID", ["S01", "S02", "S03"])
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Hot"])
        holiday_flag = st.selectbox("Holiday Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        submitted = st.form_submit_button("➕ Add Item")

    if submitted:
        new_entry = {
            "Product": product,
            "Category": category,
            "StockQty": stock_qty,
            "WeeklySales": weekly_sales,
            "ExpiryDate": expiry_date,
            "StoreID": store_id,
            "Weather": weather,
            "HolidayFlag": holiday_flag
        }
        CSV_FILE="mock_inventory.csv"
        try:
            df_existing = pd.read_csv(CSV_FILE)
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=new_entry.keys())
        df_updated = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
        df_updated.to_csv(CSV_FILE, index=False)
        st.success(f"✅ Product '{product}' added successfully!")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><div class='sidebar-title'>🔍 Filter Inventory</div>", unsafe_allow_html=True)
    selected_category = st.multiselect("Select Category", options=[], default=[])
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- MAIN APP ---------------------------
st.markdown("<h1>🌿 EcoStock AI</h1>", unsafe_allow_html=True)
st.markdown("##### Smart Inventory Optimization for Retail")
st.markdown("---")

# --------------------------- LOAD + PROCESS DATA ---------------------------
df = load_data(uploaded_file)
if st.session_state.get("manual_data"):
    manual_df = pd.DataFrame(st.session_state.manual_data)
    manual_df['ExpiryDate'] = pd.to_datetime(manual_df['ExpiryDate'])
    manual_df['DaysToExpire'] = (manual_df['ExpiryDate'] - datetime.today()).dt.days
    df = pd.concat([df, manual_df], ignore_index=True)
model = train_model(df)
df = apply_predictions(df, model)
df = classify_risk(df)

# Apply selected filter
if not selected_category:
    selected_category = df['Category'].unique().tolist()
filtered_df = df[df['Category'].isin(selected_category)].reset_index(drop=True)
at_risk = filtered_df[filtered_df['RiskLevel'].isin(['HIGH', 'MEDIUM'])].reset_index(drop=True)
at_risk = at_risk.sort_values(by=['RiskLevel', 'DaysToExpire'])

# --------------------------- KPIs ---------------------------
col1, col2, col3 = st.columns(3)
col1.markdown("<div class='metric-label'>📦 Total Products</div>", unsafe_allow_html=True)
col1.markdown(f"<div class='metric-value'>{len(filtered_df)}</div>", unsafe_allow_html=True)

col2.markdown("<div class='metric-label'>⚠️ High Risk</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-value'>{(filtered_df['RiskLevel'] == 'HIGH').sum()}</div>", unsafe_allow_html=True)

col3.markdown("<div class='metric-label'>🟡 Medium Risk</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-value'>{(filtered_df['RiskLevel'] == 'MEDIUM').sum()}</div>", unsafe_allow_html=True)

# --------------------------- TABLES ---------------------------
st.markdown("### 📦 Inventory Overview")
st.dataframe(filtered_df[['Product', 'Category', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']], use_container_width=True, height=350)

st.markdown("### 🚨 At-Risk Inventory")
high_risk_items = at_risk[at_risk['RiskLevel'] == 'HIGH'].reset_index(drop=True)
if not high_risk_items.empty:
    st.dataframe(high_risk_items[['Product', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']], use_container_width=True)
else:
    st.success("🎉 No high-risk items currently.")

# --------------------------- SUGGESTIONS ---------------------------
st.markdown("### ✅ Actionable Suggestions")
if not at_risk.empty:
    for i in range(0, len(at_risk), 3):
        row_data = at_risk.iloc[i:i+3]
        cols = st.columns(3)
        for col, (_, row) in zip(cols, row_data.iterrows()):
            risk_class = {
                'HIGH': 'risk-high',
                'MEDIUM': 'risk-medium',
                'LOW': 'risk-low'
            }.get(row['RiskLevel'], 'risk-low')
            col.markdown(f"""
                <div class='suggestion-card'>
                    <div style='font-size: 18px; font-weight: bold;'>{row['Product']}</div>
                    <div class='{risk_class}' style='margin-top: 4px;'>● {row['RiskLevel']} Risk</div>
                    <div style='margin-top: 8px; font-size: 14px; line-height: 1.6'>
                        <b>Category:</b> {row['Category']}<br>
                        <b>Store:</b> {row['StoreID']}<br>
                        <b>Demand:</b> {row['PredictedDemand']}<br>
                        <b>Stock:</b> {row['StockQty']}<br>
                        <b>Expires in:</b> {row['DaysToExpire']} day(s)
                    </div>
                    <div style='margin-top: 10px; color: #1abc9c; font-weight: 500; font-size: 13px;'>
                        💡 Suggestion: {
                            "🔥 Act fast — apply aggressive discounting!" if row['RiskLevel'] == 'HIGH' else
                            "🎯 Bundle or lightly promote to clear inventory." if row['RiskLevel'] == 'MEDIUM' else
                            "✅ Inventory in good shape — no action needed."
                        }
                    </div>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("No actionable suggestions to show.")

# --------------------------- CHARTS ---------------------------
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📊 Weekly Sales vs Predicted Demand")
    st.bar_chart(filtered_df[['Product', 'WeeklySales', 'PredictedDemand']].set_index('Product'))

with col5:
    st.markdown("### 📈 Risk Distribution")
    st.bar_chart(filtered_df['RiskLevel'].value_counts())

# --------------------------- FOOTER ---------------------------
st.markdown("<div class='footer'>Built by <b>Ritika & Nikhil</b> for Walmart Sparkathon 2025 💡<br>GitHub: <a href='https://github.com/Nikhil020Yadav/EcoStock' style='color: #888;' target='_blank'>EcoStock</a></div>", unsafe_allow_html=True)
