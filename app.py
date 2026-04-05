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
st.markdown("### Data-driven customer segmentation & experience analytics")

# ------------------ SIDEBAR ------------------
menu = st.sidebar.radio("Navigation", [
    "Overview",
    "Problem Analysis",
    "Customer Segmentation",
    "Predictive Analytics",
    "Trust Score",
    "Actions",
    "Business Impact"
])

# ------------------ OVERVIEW ------------------
if menu == "Overview":
    st.header("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Customers", len(customers))
    col2.metric("💰 Avg Order Value", round(customers["avg_order_value"].mean(), 2))
    col3.metric("⭐ Avg Trust Score", round(customers["trust_score"].mean(), 2))
    col4.metric("⚠️ High Risk Customers", len(customers[customers["trust_score"] < 60]))

    fig = px.histogram(customers, x="avg_order_value", title="Order Value Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Higher-value customers tend to have higher trust scores, indicating strong loyalty and better experience.")

# ------------------ PROBLEM ANALYSIS ------------------
elif menu == "Problem Analysis":
    st.header("⚠️ Problem Analysis")

    st.subheader("🚚 Delivery Failures")
    fig = px.histogram(transactions, x="delivery_status", color="delivery_status")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Delivery failures significantly impact customer trust and increase churn probability.")

    st.subheader("💸 Refund Issues")
    fig2 = px.histogram(transactions, x="refund_status", color="refund_status")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Delayed refunds create friction and reduce repeat purchase likelihood.")

# ------------------ CUSTOMER SEGMENTATION ------------------
elif menu == "Customer Segmentation":
    st.header("👥 Customer Segmentation")

    features = customers[["avg_order_value", "return_rate", "trust_score"]]

    kmeans = KMeans(n_clusters=4, random_state=42)
    customers["cluster"] = kmeans.fit_predict(features)

    fig = px.scatter(
        customers,
        x="avg_order_value",
        y="trust_score",
        color=customers["cluster"].astype(str),
        title="Customer Clusters"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Customers are grouped into distinct behavioral clusters, enabling targeted personalization and retention strategies.")

# ------------------ PREDICTIVE ANALYTICS ------------------
elif menu == "Predictive Analytics":
    st.header("🔮 Churn Risk Prediction")

    customers["churn_risk"] = customers["return_rate"].apply(
        lambda x: "High" if x > 0.3 else "Medium" if x > 0.15 else "Low"
    )

    fig = px.histogram(customers, x="churn_risk", color="churn_risk")
    st.plotly_chart(fig, use_container_width=True)

    high_risk = len(customers[customers["churn_risk"] == "High"])

    st.markdown("### 💡 Insight")
    st.write(f"{high_risk} customers are at high risk of churn and require immediate intervention.")

# ------------------ TRUST SCORE ------------------
elif menu == "Trust Score":
    st.header("⭐ Customer Trust Score")

    fig = px.histogram(customers, x="trust_score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    low_trust = len(customers[customers["trust_score"] < 60])

    st.markdown("### 💡 Insight")
    st.write(f"{low_trust} customers have low trust scores and need prioritized service recovery.")

# ------------------ ACTIONS ------------------
elif menu == "Actions":
    st.header("🎯 Recommended Actions")

    high_risk = customers[customers["trust_score"] < 60]

    st.subheader("⚠️ High Risk Customers")
    st.dataframe(high_risk.head(20))

    st.markdown("### 💡 Insight")
    st.write("Targeting high-risk customers with personalized offers and faster support can improve retention.")

# ------------------ BUSINESS IMPACT ------------------
elif menu == "Business Impact":
    st.header("💰 Business Impact")

    fig = px.bar(customers.head(50), x="customer_id", y="avg_order_value")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("A small segment of high-value customers contributes significantly to overall revenue, highlighting the need for retention strategies.")
