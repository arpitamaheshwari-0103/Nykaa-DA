import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Nykaa Dashboard", layout="wide")

# ------------------ LOAD DATA ------------------
customers = pd.read_csv("nykaa_customers.csv")
transactions = pd.read_csv("nykaa_transactions.csv")

customers["loyalty_tier"] = customers["loyalty_tier"].fillna("No Membership")

# ------------------ TITLE ------------------
st.markdown("# 💄 Nykaa Smart CX Dashboard")
st.markdown("### Customer Segmentation + Experience Intelligence System")

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔍 Filters")

city = st.sidebar.selectbox("City Tier", ["All"] + sorted(customers["city_tier"].unique()))
loyalty = st.sidebar.selectbox("Loyalty Tier", ["All"] + sorted(customers["loyalty_tier"].unique()))

filtered = customers.copy()

if city != "All":
    filtered = filtered[filtered["city_tier"] == city]

if loyalty != "All":
    filtered = filtered[filtered["loyalty_tier"] == loyalty]

# ------------------ RFM ------------------
rfm = transactions.groupby("customer_id").agg({
    "days_since_last_purchase": "min",
    "order_id": "count",
    "order_value": "sum"
}).reset_index()

rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]
filtered = filtered.merge(rfm, on="customer_id", how="left")

# ------------------ NAV ------------------
menu = st.sidebar.radio("Navigation", [
    "Overview", "Problem Analysis", "Customer Segmentation",
    "RFM Segmentation", "Predictive Analytics",
    "Trust Score", "Churn Simulator",
    "Customer Lookup", "Actions",
    "Business Impact", "Final Recommendations"
])

# ------------------ OVERVIEW ------------------
if menu == "Overview":
    st.header("📊 Overview")

    st.info("👉 Customer experience directly impacts trust and retention.")

    st.plotly_chart(
        px.histogram(filtered, x="avg_order_value",
                     color_discrete_sequence=["#636EFA"]),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("High-value customers show higher trust, while low trust users indicate churn risk.")

# ------------------ PROBLEM ------------------
elif menu == "Problem Analysis":
    st.header("⚠️ Problem Analysis")
    st.warning("⚠️ Delivery failures and refund delays drive churn.")

    tx = transactions[transactions["customer_id"].isin(filtered["customer_id"])]

    st.subheader("🚚 Delivery Status")
    st.plotly_chart(
        px.histogram(tx, x="delivery_status",
                     color="delivery_status",
                     color_discrete_map={"Delivered": "green", "Failed": "red"}),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("Failed deliveries directly increase dissatisfaction and churn.")

    st.subheader("💸 Refund Status")
    st.plotly_chart(
        px.histogram(tx, x="refund_status",
                     color="refund_status"),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("Refund delays reduce trust and repeat purchases.")

# ------------------ SEGMENTATION ------------------
elif menu == "Customer Segmentation":
    st.header("👥 Segmentation")
    st.info("📊 Customers grouped by behavior for targeting.")

    kmeans = KMeans(n_clusters=4, random_state=42)
    filtered["cluster"] = kmeans.fit_predict(filtered[["avg_order_value", "return_rate", "trust_score"]])

    st.plotly_chart(
        px.scatter(filtered, x="avg_order_value", y="trust_score",
                   color=filtered["cluster"].astype(str)),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("Clusters enable personalized marketing and retention strategies.")

# ------------------ RFM ------------------
elif menu == "RFM Segmentation":
    st.header("📊 RFM")
    st.info("📊 Identifies loyal vs at-risk customers.")

    st.plotly_chart(
        px.scatter(rfm, x="Recency", y="Monetary",
                   size="Frequency", color="Monetary"),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("Low recency & high spend = loyal. High recency = at risk.")

# ------------------ PREDICTIVE ------------------
elif menu == "Predictive Analytics":
    st.header("🔮 Churn Prediction")
    st.warning("⚠️ High return rates signal churn risk.")

    filtered["churn"] = filtered["return_rate"].apply(
        lambda x: "High" if x > 0.3 else "Medium" if x > 0.15 else "Low"
    )

    st.plotly_chart(
        px.histogram(filtered, x="churn", color="churn",
                     color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("Return-heavy customers are most likely to churn.")

# ------------------ TRUST ------------------
elif menu == "Trust Score":
    st.header("⭐ Trust Score")

    st.plotly_chart(
        px.histogram(filtered, x="trust_score",
                     color_discrete_sequence=["#00CC96"]),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("Low trust users require immediate attention.")

# ------------------ SIMULATOR ------------------
elif menu == "Churn Simulator":
    st.header("🧠 Churn Simulator")

    r = st.slider("Return Rate", 0.0, 0.5, 0.2)
    c = st.slider("Complaints", 0, 5, 1)

    score = 100 - (r * 100) - (c * 5)

    if score < 50:
        st.error(f"🚨 High Risk | Score: {round(score,2)}")
    elif score < 70:
        st.warning(f"⚠️ Medium Risk | Score: {round(score,2)}")
    else:
        st.success(f"✅ Low Risk | Score: {round(score,2)}")

    st.markdown("### 💡 Insight")
    st.write("More returns + complaints = lower trust & higher churn.")

# ------------------ LOOKUP ------------------
elif menu == "Customer Lookup":
    st.header("🔍 Customer Lookup")

    cid = st.text_input("Enter ID")

    if cid:
        res = customers[customers["customer_id"] == cid]
        st.dataframe(res)

        st.markdown("### 💡 Insight")
        st.write("Customer-level view helps targeted action.")

# ------------------ ACTION ------------------
elif menu == "Actions":
    st.header("🎯 Actions")

    def act(x):
        if x < 50:
            return "🚨 Immediate action"
        elif x < 70:
            return "⚠️ Offer discount"
        else:
            return "✅ Maintain loyalty"

    filtered["Action"] = filtered["trust_score"].apply(act)

    st.dataframe(filtered[["customer_id", "trust_score", "Action"]].head(20))

    st.markdown("### 💡 Insight")
    st.write("Different segments need different strategies.")

# ------------------ BUSINESS ------------------
elif menu == "Business Impact":
    st.header("💰 Business Impact")

    st.plotly_chart(
        px.bar(filtered.head(50), x="customer_id", y="avg_order_value",
               color_discrete_sequence=["#636EFA"]),
        use_container_width=True
    )

    st.markdown("### 💡 Insight")
    st.write("High-value customers drive most revenue.")

# ------------------ FINAL ------------------
elif menu == "Final Recommendations":
    st.header("📌 Final Recommendations")

    st.success("""
    ✔ Improve delivery reliability  
    ✔ Speed up refunds  
    ✔ Target high-risk customers  
    ✔ Reward loyal users  
    ✔ Personalize marketing  
    """)

    st.markdown("### 💡 Final Insight")
    st.write("Better customer experience = higher retention + revenue.")
