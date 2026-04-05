import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Nykaa Dashboard", layout="wide")

# ------------------ LOAD DATA ------------------
customers = pd.read_csv("nykaa_customers.csv")
transactions = pd.read_csv("nykaa_transactions.csv")

# ------------------ TITLE ------------------
st.markdown("# 💄 Nykaa Smart CX Dashboard")
st.markdown("### Customer Segmentation + Experience Intelligence System")

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.header("🔍 Filters")

city = st.sidebar.selectbox("City Tier", ["All"] + list(customers["city_tier"].unique()))
loyalty = st.sidebar.selectbox("Loyalty Tier", ["All"] + list(customers["loyalty_tier"].unique()))

filtered = customers.copy()

if city != "All":
    filtered = filtered[filtered["city_tier"] == city]

if loyalty != "All":
    filtered = filtered[filtered["loyalty_tier"] == loyalty]

# ------------------ RFM CALCULATION ------------------
rfm = transactions.groupby("customer_id").agg({
    "days_since_last_purchase": "min",
    "order_id": "count",
    "order_value": "sum"
}).reset_index()

rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]

# Scoring
rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])
rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])

rfm["RFM_Score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

# Merge with customers
filtered = filtered.merge(rfm, on="customer_id", how="left")

# ------------------ NAVIGATION ------------------
menu = st.sidebar.radio("Navigation", [
    "Overview",
    "Problem Analysis",
    "Customer Segmentation",
    "RFM Segmentation",
    "Predictive Analytics",
    "Trust Score",
    "Churn Simulator",
    "Customer Lookup",
    "Actions",
    "Business Impact"
])

# ------------------ OVERVIEW ------------------
if menu == "Overview":
    st.header("📊 Overview")

    st.markdown("## 📌 Executive Summary")

    high_risk = len(filtered[filtered["trust_score"] < 60])
    avg_trust = round(filtered["trust_score"].mean(), 2)

    st.info(f"""
    - Total Customers: {len(filtered)}
    - High Risk Customers: {high_risk}
    - Average Trust Score: {avg_trust}

    👉 Customer experience issues directly impact trust and churn risk.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Customers", len(filtered))
    col2.metric("💰 Avg Order Value", round(filtered["avg_order_value"].mean(), 2))
    col3.metric("⭐ Avg Trust Score", avg_trust)
    col4.metric("⚠️ High Risk", high_risk)

    fig = px.histogram(filtered, x="avg_order_value", title="Order Value Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top 5 Customers")
    top_customers = filtered.sort_values(by="avg_order_value", ascending=False).head(5)
    st.dataframe(top_customers)

# ------------------ PROBLEM ANALYSIS ------------------
elif menu == "Problem Analysis":
    st.header("⚠️ Problem Analysis")

    filtered_transactions = transactions[
        transactions["customer_id"].isin(filtered["customer_id"])
    ]

    st.subheader("🚚 Delivery Failures")
    fig = px.histogram(filtered_transactions, x="delivery_status", color="delivery_status")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💸 Refund Issues")
    fig2 = px.histogram(filtered_transactions, x="refund_status", color="refund_status")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ CUSTOMER SEGMENTATION ------------------
elif menu == "Customer Segmentation":
    st.header("👥 K-Means Segmentation")

    features = filtered[["avg_order_value", "return_rate", "trust_score"]]

    kmeans = KMeans(n_clusters=4, random_state=42)
    filtered["cluster"] = kmeans.fit_predict(features)

    fig = px.scatter(filtered, x="avg_order_value", y="trust_score",
                     color=filtered["cluster"].astype(str))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ RFM SEGMENTATION ------------------
elif menu == "RFM Segmentation":
    st.header("📊 RFM Segmentation")

    fig = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="Monetary")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(rfm.head(20))

# ------------------ PREDICTIVE ------------------
elif menu == "Predictive Analytics":
    st.header("🔮 Churn Prediction")

    filtered["churn_risk"] = filtered["return_rate"].apply(
        lambda x: "High" if x > 0.3 else "Medium" if x > 0.15 else "Low"
    )

    fig = px.histogram(filtered, x="churn_risk", color="churn_risk")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ TRUST SCORE ------------------
elif menu == "Trust Score":
    st.header("⭐ Trust Score")

    fig = px.histogram(filtered, x="trust_score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ CHURN SIMULATOR ------------------
elif menu == "Churn Simulator":
    st.header("🧠 Churn Simulator")

    return_rate = st.slider("Return Rate", 0.0, 0.5, 0.2)
    complaints = st.slider("Complaint Count", 0, 5, 1)

    score = 100 - (return_rate * 100) - (complaints * 5)
    risk = "High" if score < 50 else "Medium" if score < 70 else "Low"

    st.metric("Trust Score", round(score, 2))
    st.write(f"Risk Level: {risk}")

# ------------------ CUSTOMER LOOKUP ------------------
elif menu == "Customer Lookup":
    st.header("🔍 Customer Lookup")

    cust_id = st.text_input("Enter Customer ID")

    if cust_id:
        result = customers[customers["customer_id"] == cust_id]

        if not result.empty:
            st.dataframe(result)
        else:
            st.write("Not found")

# ------------------ ACTIONS ------------------
elif menu == "Actions":
    st.header("🎯 Recommended Actions")

    def get_action(row):
        if row["trust_score"] < 50:
            return "🚨 Immediate retention action"
        elif row["trust_score"] < 70:
            return "⚠️ Offer discount"
        else:
            return "✅ Maintain loyalty"

    filtered["action"] = filtered.apply(get_action, axis=1)

    st.dataframe(filtered[["customer_id", "trust_score", "action"]].head(20))

# ------------------ BUSINESS IMPACT ------------------
elif menu == "Business Impact":
    st.header("💰 Business Impact")

    fig = px.bar(filtered.head(50), x="customer_id", y="avg_order_value")
    st.plotly_chart(fig, use_container_width=True)
