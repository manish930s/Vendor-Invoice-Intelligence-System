import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Premium UI
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
        color: white;
    }

    /* Card Styling */
    .info-card {
        background: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.5rem;
    }
    .info-card h3, .info-card p {
        color: var(--text-color) !important;
    }
    
    /* Big Prediction Results */
    .result-metric {
        background: var(--secondary-background-color);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-top: 5px solid #3b82f6;
        animation: fadeIn 0.5s ease-in;
    }
    .result-metric.danger {
        border-top: 5px solid #ef4444;
    }
    .result-metric.success {
        border-top: 5px solid #10b981;
    }
    .result-value {
        font-size: 3rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 1rem 0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
        background-color: #3b82f6;
        color: white;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>📦 Vendor Invoice Intelligence</h1>
    <p>AI-Driven Freight Cost Prediction & Invoice Risk Flagging</p>
    This internal analytics portal leverages machine learning to<br>
- Forecast freight costs accurately<br>
- Detect risky or abnormal vendor invoices<br>
- Reduce financial leakage and manual workload<br>
</div>
""", unsafe_allow_html=True)

# st.markdown("""
# # 📦 Vendor Invoice Intelligence Portal
# ### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

# This internal analytics portal leverages machine learning to
# - **Forecast freight costs accurately**
# - **Detect risky or abnormal vendor invoices**
# - **Reduce financial leakage and manual workload**
# """)
# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3143/3143460.png", width=60)
    st.markdown("### **Navigation**")
    selected_model = st.radio(
        "Choose Module",
        [
            "🚛 Freight Cost Prediction",
            "📋 Invoice Manual Approval Flag"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    ### 💡 **Business Impact**
    <div style='background: var(--secondary-background-color); color: var(--text-color); padding: 1rem; border-radius: 8px; font-size: 0.9rem; border: 1px solid rgba(128, 128, 128, 0.2);'>
    <b>• Improved cost forecasting</b><br>
    <b>• Reduced invoice fraud</b><br>
    <b>• Faster finance operations</b>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Module 1: Freight Cost Prediction
# ---------------------------------------------------------
if selected_model == "🚛 Freight Cost Prediction":
    
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top:0;">🎯 Objective</h3>
        <p style="margin-bottom:0; color:#475569;">Predict the freight cost for vendor invoices using <b>Quantity</b> and <b>Invoice Dollars</b> to support accurate budgeting, forecasting, and vendor negotiation.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        with st.form("freight_form", clear_on_submit=False):
            st.markdown("#### 📝 Input Invoice Details")
            col1, col2 = st.columns(2)
            
            with col1:
                quantity = st.number_input(
                    "📦 Quantity",
                    min_value=1,
                    value=1200,
                    help="Total number of units in the shipment"
                )
                
            with col2:
                dollars = st.number_input(
                    "💰 Invoice Dollars ($)",
                    min_value=1.0,
                    value=18500.0,
                    help="Total monetary value of the invoice"
                )

            st.write("")
            submit_freight = st.form_submit_button("Predict Freight Cost 🚀")

    if submit_freight:
        with st.spinner("Analyzing shipping matrix and historical rates..."):
            input_data = {
                "Dollars": [dollars]
            }
            predictions = predict_freight_cost(input_data)['Predicted_Freight']
            
            st.markdown(f"""
            <div class="result-metric success">
                <span style="color:#64748b; font-size:1.1rem; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Predicted Freight Cost</span>
                <div class="result-value">${predictions[0]:,.2f}</div>
                <span style="color:#10b981; font-weight:500;">prediction generated successfully based on ${dollars:,.2f} invoice value.</span>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------
# Module 2: Invoice Flag Prediction
# -----------------------------------------------------------
else:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top:0;">🎯 Objective</h3>
        <p style="margin-bottom:0; color:#475569;">Predict whether a vendor invoice should be <b>flagged for manual approval</b> based on abnormal cost, freight, or delivery discrepancy patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        with st.form("invoice_flag_form"):
            st.markdown("#### 📝 Input Discrepancy Factors")
            col1, col2, col3 = st.columns(3)

            with col1:
                invoice_quantity = st.number_input(
                    "Invoice Quantity",
                    min_value=1,
                    value=50,
                    help="Quantity claimed on the vendor invoice"
                )
                freight = st.number_input(
                    "Freight Cost ($)",
                    min_value=0.0,
                    value=15.00
                )
                
            with col2:
                invoice_dollars = st.number_input(
                    "Invoice Dollars ($)",
                    min_value=1.0,
                    value=1500.00,
                    help="Amount billed by the vendor"
                )
                total_item_quantity = st.number_input(
                    "Purchased Quantity",
                    min_value=1,
                    value=50,
                    help="Internal quantity officially purchased"
                )

            with col3:
                total_item_dollars = st.number_input(
                  "Purchased Dollars ($)",
                  min_value=1.0,
                  value=1500.00,
                  help="Internal expected cost of items"
                )

            st.write("")
            submit_flag = st.form_submit_button("Evaluate Risk Status 🛡️")

    if submit_flag:
        with st.spinner("Running discrepancy analysis..."):
            input_data = {
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars]
            }

            flag_prediction = predict_invoice_flag(input_data)['flag_invoice']
            is_flagged = bool(flag_prediction[0])

            if is_flagged:
                st.markdown("""
                <div class="result-metric danger">
                    <span style="color:#ef4444; font-size:3rem;">🚨</span><br>
                    <span style="color:#64748b; font-size:1.1rem; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Status</span>
                    <div style="font-size: 2rem; font-weight: 800; color: #ef4444; margin: 1rem 0;">Requires Manual Approval</div>
                    <span style="color:#ef4444; font-weight:500;">High discrepancy detected between invoice and purchase orders.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-metric success">
                    <span style="color:#10b981; font-size:3rem;">✅</span><br>
                    <span style="color:#64748b; font-size:1.1rem; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Status</span>
                    <div style="font-size: 2rem; font-weight: 800; color: #10b981; margin: 1rem 0;">Safe for Auto-Approval</div>
                    <span style="color:#10b981; font-weight:500;">No significant anomalies detected. Standard automated processing authorized.</span>
                </div>
                """, unsafe_allow_html=True)
