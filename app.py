import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Nykaa Dashboard", layout="wide")

customers = pd.read_csv("nykaa_customers.csv")
transactions = pd.read_csv("nykaa_transactions.csv")

st.title("💄 Nykaa Customer Experience Dashboard")

menu = st.sidebar.radio("Navigation", [
    "Overview","Problem Analysis","Customer Segmentation",
    "Predictive Analytics","Trust Score","Actions","Business Impact"
])

if menu == "Overview":
    st.header("📊 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(customers))
    col2.metric("Avg Order Value", round(customers["avg_order_value"].mean(), 2))
    col3.metric("Avg Trust Score", round(customers["trust_score"].mean(), 2))
    fig = px.histogram(customers, x="avg_order_value")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Problem Analysis":
    st.header("⚠️ Problem Analysis")
    fig = px.histogram(transactions, x="delivery_status")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.histogram(transactions, x="refund_status")
    st.plotly_chart(fig2, use_container_width=True)

elif menu == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    features = customers[["avg_order_value","return_rate","trust_score"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    customers["cluster"] = kmeans.fit_predict(features)
    fig = px.scatter(customers, x="avg_order_value", y="trust_score",
                     color=customers["cluster"].astype(str))
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Predictive Analytics":
    st.header("🔮 Churn Risk")
    customers["churn_risk"] = customers["return_rate"].apply(
        lambda x: "High" if x > 0.3 else "Medium" if x > 0.15 else "Low"
    )
    fig = px.histogram(customers, x="churn_risk", color="churn_risk")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Trust Score":
    st.header("⭐ Trust Score Analysis")
    fig = px.histogram(customers, x="trust_score", nbins=10)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Actions":
    st.header("🎯 Recommended Actions")
    high_risk = customers[customers["trust_score"] < 60]
    st.dataframe(high_risk)

elif menu == "Business Impact":
    st.header("💰 Business Impact")
    fig = px.bar(customers, x="customer_id", y="avg_order_value")
    st.plotly_chart(fig, use_container_width=True)
