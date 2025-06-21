import streamlit as st
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

# Title & Description
st.set_page_config(page_title="Urban Mobility Optimizer", layout="wide")
st.title("🌍 AI for Sustainable Cities – Urban Mobility Optimizer")
st.markdown("""
This tool uses **K-Means Clustering** to identify high-demand zones from transportation data, helping cities optimize public transport routes.
Supports UN SDG 11: Sustainable Cities and Communities.
""")

# Sidebar
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with 'latitude' and 'longitude' columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show raw data
    st.subheader("📊 Raw Data")
    st.write(df.head())

    # Select lat/lon columns
    cols = df.columns.tolist()
    lat_col = st.sidebar.selectbox("Select Latitude Column", cols)
    lon_col = st.sidebar.selectbox("Select Longitude Column", cols)

    if lat_col and lon_col:
        data = df[[lat_col, lon_col]].dropna()

        # Cluster settings
        st.sidebar.header("Clustering Options")
        n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)

        if st.sidebar.button("Run Clustering"):
            # Preprocess
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(data_scaled)

            # Add cluster labels
            df['cluster'] = clusters

            # Display clustered data
            st.subheader("🧮 Clustered Data")
            st.write(df.head())

            # Map visualization
            st.subheader("🗺️ Cluster Map")
            m = folium.Map(location=[data[lat_col].mean(), data[lon_col].mean()], zoom_start=12)

            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'beige', 'darkblue', 'pink']

            for idx, row in df.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=2,
                    color=colors[row['cluster']],
                    fill=True,
                    fill_color=colors[row['cluster']]
                ).add_to(m)

            st_folium(m, width=1000, height=600)

            # Download clustered data
            st.sidebar.download_button(
                label="Download Clustered Data",
                data=df.to_csv(index=False),
                file_name='clustered_data.csv',
                mime='text/csv'
            )

else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io)  | 🎯 Supports UN SDG 11")