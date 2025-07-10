import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import os

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
            color: #cccccc;
            margin-bottom: 0.3rem;
            margin-top: 0.5rem;
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
with st.sidebar.expander("### 📥 Upload or Add Inventory"):
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
with st.sidebar.expander("### ✍️ Add New Product"):

    with st.form("upload_form", clear_on_submit=True):
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
with st.sidebar.expander("### 🔍 Filter Inventory"):
    selected_category = None  # To be set after loading
    selected_category = st.multiselect("Select Category", options=df['Category'].unique(), default=df['Category'].unique())

filtered_df = df[df['Category'].isin(selected_category)].reset_index(drop=True) if selected_category else df
at_risk = filtered_df[filtered_df['RiskLevel'].isin(['HIGH', 'MEDIUM'])].reset_index(drop=True)
at_risk = at_risk.sort_values(by=['RiskLevel', 'DaysToExpire'])


# --------------------------- KPIs ---------------------------
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div class='metric-value'>📦 Total - {len(filtered_df)}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-value'>⚠️ High Risk - {(filtered_df['RiskLevel'] == 'HIGH').sum()}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-value'>🟡 Medium Risk - {(filtered_df['RiskLevel'] == 'MEDIUM').sum()}</div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-value'>🟢 Low Risk - {(filtered_df['RiskLevel'] == 'LOW').sum()}</div>", unsafe_allow_html=True)

# --------------------------- CHARTS ---------------------------

col4, col5 = st.columns(2)

with col4:
    st.markdown("### 📊 Weekly Sales vs Predicted Demand")
    sales_fig = px.bar(
        filtered_df,
        x="Product",
        y=["WeeklySales", "PredictedDemand"],
        barmode="group",
        title="Weekly Sales vs Predicted Demand",
        color_discrete_map={"WeeklySales": "#636EFA", "PredictedDemand": "#EF553B"}
    )
    st.plotly_chart(sales_fig, use_container_width=True)

with col5:
    st.markdown("### 📈 Risk Distribution")
    risk_count = filtered_df['RiskLevel'].value_counts().reset_index()
    risk_count.columns = ['RiskLevel', 'Count']
    risk_fig = px.bar(
        risk_count,
        x="RiskLevel",
        y="Count",
        color="RiskLevel",
        title="Risk Level Distribution",
        color_discrete_map={
            "HIGH": "#e76f51",
            "MEDIUM": "#f4a261",
            "LOW": "#2a9d8f"
        }
    )
    st.plotly_chart(risk_fig, use_container_width=True)

# --------------------------- TABLES ---------------------------

def format_days_to_expire(days):
    if days <= 0:
        return "<span style='color:#ff4d4d; font-weight:bold;'>EXPIRED</span>"
    else:
        return f"{days} day(s)"

# Create a copy for rendering with HTML formatting
display_df = filtered_df.copy()
display_df['DaysToExpire'] = display_df['DaysToExpire'].apply(format_days_to_expire)

# st.markdown("### 📦 Inventory Overview")
# st.dataframe(filtered_df[['Product', 'Category', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']],
#              use_container_width=True, height=350)
st.markdown("### 📦 Inventory Overview")
st.write(display_df[['Product', 'Category', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']].to_html(escape=False, index=False), unsafe_allow_html=True)


if "show_delete" not in st.session_state:
    st.session_state.show_delete = False
if "selected_delete_idx" not in st.session_state:
    st.session_state.selected_delete_idx = None
_, col_export, col_del = st.columns([8, 2, 1])
with col_export:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📤 Export CSV",
        data=csv,
        file_name="filtered_inventory.csv",
        mime='text/csv',
        key="export_button"
    )

with col_del:
    # st.session_state.show_delete = st.button("🗑️", key="delete_button", help="Delete a product")
    if st.button("🗑️", key="delete_button", help="Delete a product"):
        st.session_state.show_delete = True  # Persist delete mode

if st.session_state.show_delete:
    st.markdown("### ❌ Delete Inventory Item")

    # Create readable labels and safe index reference
    filtered_df = filtered_df.reset_index(drop=True)
    label_df = filtered_df.copy()
    label_df['Label'] = label_df.apply(
        lambda x: f"{x['Product']} ({x['Category']}) - {x['StoreID']} | Exp: {x['ExpiryDate'].date()}",
        axis=1
    )

    # Show selectbox with index as key
    st.session_state.selected_idx = st.selectbox(
        "Select an item to delete:",
        options=label_df.index,
        format_func=lambda i: label_df.loc[i, 'Label']
    )

    if st.button("Confirm Delete"):
        try:
            full_df = pd.read_csv("mock_inventory.csv")
            full_df['ExpiryDate'] = pd.to_datetime(full_df['ExpiryDate'])

            row = filtered_df.loc[st.session_state.selected_idx]
            exp = pd.to_datetime(row['ExpiryDate']).normalize()

            condition = (
                (full_df['Product'] == row['Product']) &
                (full_df['Category'] == row['Category']) &
                (full_df['StoreID'] == row['StoreID']) &
                (full_df['ExpiryDate'].dt.normalize() == exp)
            )

            updated_df = full_df[~condition]
            updated_df.to_csv("mock_inventory.csv", index=False)

            st.success(f"✅ Deleted: {row['Product']} from store {row['StoreID']}")
            st.session_state.show_delete = False
            st.session_state.selected_delete_idx = None   
            st.rerun()

        except Exception as e:
            st.error(f"Error deleting entry: {e}")


st.markdown("### 🚨 At-Risk Inventory")
high_risk_items = at_risk[at_risk['RiskLevel'] == 'HIGH'].reset_index(drop=True)
high_risk_items_display = high_risk_items.copy()
high_risk_items_display['DaysToExpire'] = high_risk_items_display['DaysToExpire'].apply(format_days_to_expire)
if not high_risk_items.empty:
    # st.dataframe(high_risk_items[['Product', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']], use_container_width=True)
    st.write(high_risk_items_display[['Product', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire', 'RiskLevel']].to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.success("🎉 No high-risk items currently.")

# --------------------------- SUGGESTIONS ---------------------------
# st.markdown("### ✅ Actionable Suggestions")
# if not at_risk.empty:
#     for i in range(0, len(at_risk), 3):
#         row_data = at_risk.iloc[i:i+3]
#         cols = st.columns(3)
#         for col, (_, row) in zip(cols, row_data.iterrows()):
#             risk_class = {
#                 'HIGH': 'risk-high',
#                 'MEDIUM': 'risk-medium',
#                 'LOW': 'risk-low'
#             }.get(row['RiskLevel'], 'risk-low')
#             col.markdown(f"""
#                 <div class='suggestion-card'>
#                     <div style='font-size: 18px; font-weight: bold;'>{row['Product']}</div>
#                     <div class='{risk_class}' style='margin-top: 4px;'>● {row['RiskLevel']} Risk</div>
#                     <div style='margin-top: 8px; font-size: 14px; line-height: 1.6'>
#                         <b>Category:</b> {row['Category']}<br>
#                         <b>Store:</b> {row['StoreID']}<br>
#                         <b>Demand:</b> {row['PredictedDemand']}<br>
#                         <b>Stock:</b> {row['StockQty']}<br>
#                         {f"<span style='color:#ff4d4d; font-weight:bold;'>EXPIRED</span><br>" if row['DaysToExpire'] <= 0 else f"<b>Expires in:</b> {row['DaysToExpire']} day(s)<br>"}
#                     </div>
#                    <div style='margin-top: 10px; color: #1abc9c; font-weight: 500; font-size: 13px;'>
#                     💡 Suggestion: {
#                     "🗑️ Discard expired product immediately!" if row['DaysToExpire'] <= 0 else
#                     "🔥 Act fast — apply aggressive discounting!" if row['RiskLevel'] == 'HIGH' else
#                     "🎯 Bundle or lightly promote to clear inventory." if row['RiskLevel'] == 'MEDIUM' else
#                     "✅ Inventory in good shape — no action needed."
#                     }
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
# else:
#     st.info("No actionable suggestions to show.")

# --------------------------- SMART ACTION CENTER ---------------------------
st.markdown("### 🧠 Smart Action Center")

# Filter expired items
expired_df = df[df['DaysToExpire'] <= 0].reset_index(drop=True)

if not expired_df.empty:
    st.warning("⚠️ You have expired items in inventory.")
    
    # Show expired items
    st.dataframe(expired_df[['Product', 'Category', 'StoreID', 'StockQty', 'WeeklySales', 'PredictedDemand', 'DaysToExpire']], use_container_width=True)

    # Allow download
    csv_expired = expired_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Expired Items",
        data=csv_expired,
        file_name="expired_inventory.csv",
        mime='text/csv',
        key="expired_download"
    )

    # Discard button
    if st.button("🗑️ Discard All Expired Items", key="discard_expired"):
        try:
            full_df = pd.read_csv("mock_inventory.csv")
            full_df['ExpiryDate'] = pd.to_datetime(full_df['ExpiryDate'])

            # Save backup
            full_df.to_csv("mock_inventory_backup.csv", index=False)

            # Remove expired rows
            updated_df = full_df[(full_df['ExpiryDate'] - datetime.today()).dt.days > 0]
            updated_df.to_csv("mock_inventory.csv", index=False)

            st.success("✅ All expired items discarded successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to discard expired items: {e}")
else:
    st.success("🎉 No expired items detected!")

# Undo discard
if "mock_inventory_backup.csv" in os.listdir():
    if st.button("↩️ Undo Last Discard", key="undo_expired"):
        try:
            backup_df = pd.read_csv("mock_inventory_backup.csv")
            backup_df.to_csv("mock_inventory.csv", index=False)
            st.success("✅ Undo successful — inventory restored.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to restore backup: {e}")
# High-risk discount suggestion
high_risk = at_risk[at_risk['RiskLevel'] == 'HIGH']
if not high_risk.empty:
    st.markdown("#### 🔥 High Risk: Discount Suggested")
    st.write(high_risk[['Product', 'Category', 'StoreID', 'PredictedDemand', 'StockQty']])
    st.info("Consider discounting these products by **25–40%** to accelerate clearance.")

# Medium-risk bundling
medium_risk = at_risk[at_risk['RiskLevel'] == 'MEDIUM']
if not medium_risk.empty:
    st.markdown("#### ⚠️ Medium Risk: Bundling Recommended")
    st.write(medium_risk[['Product', 'Category', 'StoreID', 'PredictedDemand', 'StockQty']])
    st.info("Consider offering **bundles or combo deals** to clear moderate risk inventory.")

# --------------------------- FOOTER ---------------------------
st.markdown("<div class='footer'>Built by <b>Ritika & Nikhil</b> for Walmart Sparkathon 2025 💡<br>GitHub: <a href='https://github.com/Nikhil020Yadav/EcoStock' style='color: #888;' target='_blank'>EcoStock</a></div>", unsafe_allow_html=True)


