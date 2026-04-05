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

# ------------------ NAVIGATION ------------------
menu = st.sidebar.radio("Navigation", [
    "Overview",
    "Problem Analysis",
    "Customer Segmentation",
    "Predictive Analytics",
    "Trust Score",
    "Churn Simulator",
    "Customer Lookup",
    "Business Impact"
])

# ------------------ OVERVIEW ------------------
if menu == "Overview":
    st.header("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Customers", len(filtered))
    col2.metric("💰 Avg Order Value", round(filtered["avg_order_value"].mean(), 2))
    col3.metric("⭐ Avg Trust Score", round(filtered["trust_score"].mean(), 2))
    col4.metric("⚠️ High Risk", len(filtered[filtered["trust_score"] < 60]))

    fig = px.histogram(filtered, x="avg_order_value", title="Order Value Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Higher-value customers tend to have higher trust scores, indicating stronger loyalty.")

# ------------------ PROBLEM ANALYSIS ------------------
elif menu == "Problem Analysis":
    st.header("⚠️ Problem Analysis")

    st.subheader("🚚 Delivery Failures")
    fig = px.histogram(transactions, x="delivery_status", color="delivery_status")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Delivery failures significantly increase churn risk.")

    st.subheader("💸 Refund Issues")
    fig2 = px.histogram(transactions, x="refund_status", color="refund_status")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Refund delays reduce customer trust and repeat purchases.")

# ------------------ CUSTOMER SEGMENTATION ------------------
elif menu == "Customer Segmentation":
    st.header("👥 Customer Segmentation")

    features = filtered[["avg_order_value", "return_rate", "trust_score"]]

    kmeans = KMeans(n_clusters=4, random_state=42)
    filtered["cluster"] = kmeans.fit_predict(features)

    fig = px.scatter(
        filtered,
        x="avg_order_value",
        y="trust_score",
        color=filtered["cluster"].astype(str),
        title="Customer Clusters"
    )

    st.plotly_chart(fig, use_container_width=True)

    cluster_select = st.selectbox("Select Cluster", filtered["cluster"].unique())
    st.dataframe(filtered[filtered["cluster"] == cluster_select].head(20))

    st.markdown("### 💡 Insight")
    st.write("Distinct customer clusters enable personalized marketing and service strategies.")

# ------------------ PREDICTIVE ANALYTICS ------------------
elif menu == "Predictive Analytics":
    st.header("🔮 Churn Prediction")

    filtered["churn_risk"] = filtered["return_rate"].apply(
        lambda x: "High" if x > 0.3 else "Medium" if x > 0.15 else "Low"
    )

    fig = px.histogram(filtered, x="churn_risk", color="churn_risk")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Customers with higher return rates are more likely to churn.")

# ------------------ TRUST SCORE ------------------
elif menu == "Trust Score":
    st.header("⭐ Trust Score Analysis")

    fig = px.histogram(filtered, x="trust_score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("Low trust customers require immediate intervention.")

# ------------------ CHURN SIMULATOR ------------------
elif menu == "Churn Simulator":
    st.header("🧠 Churn Risk Simulator")

    return_rate = st.slider("Return Rate", 0.0, 0.5, 0.2)

    risk = "High" if return_rate > 0.3 else "Medium" if return_rate > 0.15 else "Low"

    st.write(f"### Predicted Churn Risk: {risk}")

    st.markdown("### 💡 Insight")
    st.write("Higher return rates increase dissatisfaction and churn probability.")

# ------------------ CUSTOMER LOOKUP ------------------
elif menu == "Customer Lookup":
    st.header("🔍 Customer Search")

    cust_id = st.text_input("Enter Customer ID (e.g., C100)")

    if cust_id:
        result = customers[customers["customer_id"] == cust_id]

        if not result.empty:
            st.dataframe(result)
        else:
            st.write("Customer not found")

# ------------------ BUSINESS IMPACT ------------------
elif menu == "Business Impact":
    st.header("💰 Business Impact")

    fig = px.bar(filtered.head(50), x="customer_id", y="avg_order_value")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Insight")
    st.write("A small group of high-value customers drives majority revenue.")
