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
        .stApp { background-color: #121212; }
        h1, h2, h3 { color: #ffffff; }
        .suggestion-card {
            background-color: #1e1e1e;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        }
        .risk-high { color: #e76f51; }
        .risk-medium { color: #f4a261; }
        .risk-low { color: #2a9d8f; }
        .metric-label { font-size: 14px; color: #aaaaaa; }
        .metric-value { font-size: 24px; font-weight: bold; color: #ffffff; }
        .footer {
            text-align: center;
            margin-top: 4rem;
            color: #888888;
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
    st.markdown("### 📥 Upload or Add Inventory")

    # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    



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

# --------------------------- USER INPUT ---------------------------
with st.sidebar:
    st.markdown("### ✍️ Add New Product")

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
        expiry_date_str = expiry_date.strftime('%Y-%m-%d')  
        new_entry = {
            "Product": product,
            "Category": category,
            "StockQty": stock_qty,
            "WeeklySales": weekly_sales,
            "ExpiryDate": expiry_date,  # Ensure it's datetime object
            "StoreID": store_id,
            "Weather": weather,
            "HolidayFlag": holiday_flag
        }
        CSV_FILE="mock_inventory.csv"
        try:
            df_existing = pd.read_csv(CSV_FILE)
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=new_entry.keys())

        # Append new row
        df_updated = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)

        # Save back to CSV without time component
        df_updated.to_csv(CSV_FILE, index=False)
        
        st.success(f"✅ Product '{product}' added successfully!")
        st.rerun()
    
# --------------------------- CATEGORY FILTER ---------------------------
with st.sidebar:
    st.markdown("### 🔍 Filter Inventory")
    selected_category = None  # To be set after loading
    selected_category = st.multiselect("Select Category", options=df['Category'].unique(), default=df['Category'].unique())

filtered_df = df[df['Category'].isin(selected_category)].reset_index(drop=True) if selected_category else df
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
st.dataframe(filtered_df[['Product', 'Category', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']],
             use_container_width=True, height=350)


st.markdown("### 🚨 At-Risk Inventory")
high_risk_items = at_risk[at_risk['RiskLevel'] == 'HIGH'].reset_index(drop=True)

if not high_risk_items.empty:
    st.dataframe(high_risk_items[['Product', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']],
                 use_container_width=True)
else:
    st.success("🎉 No high-risk items currently.")


# --------------------------- SUGGESTIONS ---------------------------
st.markdown("### ✅ Actionable Suggestions")
if not at_risk.empty:
    # Loop through items in chunks of 3
    for i in range(0, len(at_risk), 3):
        row_data = at_risk.iloc[i:i+3]
        cols = st.columns(3)  # 3 cards per row

        for col, (_, row) in zip(cols, row_data.iterrows()):
            risk_class = {
                'HIGH': 'risk-high',
                'MEDIUM': 'risk-medium',
                'LOW': 'risk-low'
            }.get(row['RiskLevel'], 'risk-low')

            col.markdown(
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
    st.bar_chart(filtered_df['RiskLevel'].value_counts())

# --------------------------- FOOTER ---------------------------
st.markdown("<div class='footer'>Built by <b>Ritika & Nikhil</b> for Walmart Sparkathon 2025 💡<br>GitHub: <a href='https://github.com/Nikhil020Yadav/EcoStock' style='color: #888;' target='_blank'>EcoStock</a></div>", unsafe_allow_html=True)
