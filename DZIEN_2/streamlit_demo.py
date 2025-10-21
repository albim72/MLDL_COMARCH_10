# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

st.set_page_config(page_title="Interaktywny K-Means", page_icon="🌸", layout="centered")

st.title("🌸 Interaktywny K-Means Clustering Demo")
st.markdown("Eksperymentuj z liczbą klastrów i zobacz, jak zmienia się podział danych!")

# Ustawienia
n_samples = st.slider("Liczba punktów", 100, 1000, 300, 50)
n_clusters = st.slider("Liczba klastrów (K)", 2, 10, 3)
random_state = st.slider("Ziarno losowości", 0, 100, 42)

# Generowanie danych
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=1.0, random_state=random_state)

# Model K-Means
model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
labels = model.fit_predict(X)
centroids = model.cluster_centers_

# Wykres
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroidy')
ax.legend()
ax.set_title("Wizualizacja klastrów K-Means")
st.pyplot(fig)

# Tabela centroidów
st.subheader("Centroidy klastrów")
st.dataframe(pd.DataFrame(centroids, columns=["x", "y"]).round(3))

# Dodatkowa interakcja
st.markdown("---")
if st.checkbox("Pokaż dane w tabeli"):
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["klaster"] = labels
    st.dataframe(df.head(10))
