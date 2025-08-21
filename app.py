import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans

# --- Streamlit setup ---
st.set_page_config(page_title="Seasonal Customer Segmentation", layout="wide")
st.title("🛍️ Seasonal Customer Segmentation (DBSCAN & K-Means)")

# --- Upload CSV/Excel ---
uploaded_file = st.file_uploader("📤 Upload Online Retail Dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --- Data Cleaning ---
    st.subheader("🧹 Data Cleaning")
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    st.write(f"✅ Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Feature Engineering: RFM + Seasonality + Quarter ---
    st.subheader("🔧 Feature Engineering: RFM & Seasonality")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Aggregate RFM per customer
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': 'max',
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()
    rfm['Monetary'] = rfm['Quantity'] * rfm['UnitPrice']
    rfm['Recency'] = (snapshot_date - rfm['InvoiceDate']).dt.days
    rfm['Frequency'] = rfm['InvoiceNo']

    # Quarter & Season mapping
    df['Quarter'] = df['InvoiceDate'].dt.quarter
    month_to_season = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                       9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['Season'] = df['InvoiceDate'].dt.month.map(month_to_season)

    st.write("✅ RFM and seasonal features created")
    st.dataframe(rfm.head())

    # --- Prepare features for clustering ---
    st.subheader("📊 Prepare features for clustering")
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm[features].values
    X_scaled = StandardScaler().fit_transform(X)

    # --- Choose Algorithm ---
    algorithm = st.radio("Choose clustering algorithm:", ["DBSCAN", "K-Means"])

    if algorithm == "K-Means":
        st.subheader("📈 K-Means Clustering & Elbow Method")
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X_scaled)
            wcss.append(km.inertia_)
        fig_elbow, ax_elbow = plt.subplots()
        sns.lineplot(x=range(1, 11), y=wcss, marker='o', ax=ax_elbow)
        ax_elbow.set_xlabel("Number of Clusters (K)")
        ax_elbow.set_ylabel("WCSS")
        ax_elbow.set_title("Elbow Method")
        st.pyplot(fig_elbow)

        k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)
        model = KMeans(n_clusters=k, random_state=42)
        rfm['Cluster'] = model.fit_predict(X_scaled)
        st.success("✅ K-Means clustering done!")
    else:
        st.subheader("📌 DBSCAN Clustering Parameters")
        eps = st.slider("Epsilon (radius for neighborhood)", 0.1, 5.0, 0.5)
        min_samples = st.slider("Minimum Samples per Cluster", 1, 20, 3)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        rfm['Cluster'] = model.fit_predict(X_scaled)
        st.success("✅ DBSCAN clustering done!")

    # --- Merge cluster back to main dataset for seasonal/quarter analysis ---
    df = df.merge(rfm[['CustomerID', 'Cluster']], on='CustomerID', how='left')

    # --- Auto-generate Cluster Interpretation ---
    st.subheader("💡 Auto-generated Cluster Business Insights")
    cluster_summary = rfm.groupby('Cluster')[features].mean().reset_index()
    cluster_descriptions = {}

    mean_rec = rfm['Recency'].mean()
    mean_freq = rfm['Frequency'].mean()
    mean_mon = rfm['Monetary'].mean()

    # Add CustomerType column to rfm
    rfm['CustomerType'] = ''

    for idx, row in cluster_summary.iterrows():
        cluster_id = int(row['Cluster'])
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']

        # Customer Type Logic
        if recency < mean_rec and monetary > mean_mon:
            customer_type = "VIP"
            suggestion = "Offer premium products, loyalty rewards; focus on winter campaigns."
        elif recency > mean_rec and monetary > mean_mon:
            customer_type = "Impulsive"
            suggestion = "Send limited-time offers, flash sales; monitor seasonal spikes."
        elif recency < mean_rec and monetary < mean_mon:
            customer_type = "Careful"
            suggestion = "Promote budget-friendly bundles; highlight value deals."
        else:
            customer_type = "Thrifty"
            suggestion = "Encourage repeat purchases with discounts; seasonal promotions."

        cluster_descriptions[cluster_id] = {
            "CustomerType": customer_type,
            "Recency": round(recency, 1),
            "Frequency": round(frequency, 1),
            "Monetary": round(monetary, 1),
            "BusinessSuggestion": suggestion
        }

        rfm.loc[rfm['Cluster'] == cluster_id, 'CustomerType'] = customer_type

    # Display Cluster Insights
    for cluster_id, info in cluster_descriptions.items():
        st.markdown(f"### Cluster {cluster_id} → {info['CustomerType']} Customers")
        st.write(f"- **Average Recency (days):** {info['Recency']}")
        st.write(f"- **Average Frequency:** {info['Frequency']}")
        st.write(f"- **Average Monetary:** ${info['Monetary']}")
        st.write(f"- **Business Suggestion:** {info['BusinessSuggestion']}")
        st.write("---")

    # --- RFM Cluster Visualization ---
    st.subheader("🎨 RFM Cluster Visualization")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    palette = 'viridis' if algorithm == 'DBSCAN' else 'Set2'
    sns.scatterplot(
        data=rfm,
        x='Recency',
        y='Monetary',
        hue='Cluster',
        palette=palette,
        s=100,
        ax=ax1
    )
    ax1.set_title(f"{algorithm} Clusters (Recency vs Monetary)")
    st.pyplot(fig1)

    # --- Seasonal / Quarter × RFM Visualization ---
    st.subheader("📊 Seasonal / Quarter × RFM Visualization")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=df,
        x='Quantity',
        y='UnitPrice',
        hue='Cluster',
        style='Quarter',
        palette=palette,
        s=80,
        ax=ax2
    )
    ax2.set_title(f"{algorithm} Clusters Across Quarters")
    ax2.set_xlabel("Quantity Purchased")
    ax2.set_ylabel("Unit Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig2)

    # --- Business Dashboard Summary ---
    st.subheader("📊 Business Dashboard Summary")

    # Number of customers per cluster / type
    st.markdown("### 1️⃣ Number of Customers per Cluster / Customer Type")
    cluster_counts = rfm.groupby(['Cluster', 'CustomerType']).size().reset_index(name='Count')
    st.dataframe(cluster_counts)

    # Total monetary value per cluster
    st.markdown("### 2️⃣ Total Monetary Value per Cluster")
    monetary_summary = df.groupby('Cluster').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).reset_index(
        name='TotalMonetary')
    st.dataframe(monetary_summary)

    # Seasonal distribution per cluster
    st.markdown("### 3️⃣ Seasonal Distribution per Cluster")
    season_summary = df.groupby(['Cluster', 'Season']).size().reset_index(name='CustomerCount')
    st.dataframe(season_summary)

    # Quarterly cluster heatmap
    st.markdown("### 4️⃣ Quarterly Cluster Activity Heatmap")
    quarter_cluster = df.groupby(['Cluster', 'Quarter']).size().unstack(fill_value=0)
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
    sns.heatmap(quarter_cluster, annot=True, fmt='d', cmap='YlGnBu', ax=ax_heat)
    ax_heat.set_title("Number of Customers per Cluster per Quarter")
    st.pyplot(fig_heat)

    # --- NEW: Customer list per cluster ---
    st.subheader("📋 Customer List per Cluster")
    for cluster_id in sorted(rfm['Cluster'].unique()):
        st.markdown(f"#### Cluster {cluster_id}")
        cluster_table = rfm[rfm['Cluster'] == cluster_id][['CustomerID', 'Recency', 'Frequency', 'Monetary', 'CustomerType']]
        st.dataframe(cluster_table)

    # --- Highlight K-Means Limitation ---
    if algorithm == "K-Means":
        st.warning(
            "⚠️ K-Means may misclassify customers with irregular seasonal buying patterns or outliers. "
            "Notice clusters are spherical and cannot detect noise. DBSCAN handles irregular shapes and seasonal patterns better."
        )
