import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="Dashboard Klasterisasi Banjir DKI Jakarta", layout="wide")

FITUR = [
    "ketinggian_air_cm",
    "jumlah_terdampak_rt",
    "jumlah_terdampak_kk",
    "jumlah_terdampak_jiwa",
    "jumlah_pengungsi_tertinggi",
]

@st.cache_data
def load_data():
    return pd.read_csv("data_banjir_clustered.csv"), pd.read_csv("agregasi_kelurahan.csv")

@st.cache_resource
def load_model():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler

def make_label_map(kmeans, scaler):
    centers_scaled = kmeans.cluster_centers_
    centers_asli = pd.DataFrame(scaler.inverse_transform(centers_scaled), columns=FITUR)
    berat = int(centers_asli["jumlah_terdampak_jiwa"].idxmax())
    lm = {c: "Banjir Ringan" for c in range(kmeans.n_clusters)}
    lm[berat] = "Banjir Berat"
    return lm, centers_scaled, centers_asli

df, agg = load_data()
kmeans, scaler = load_model()
lmap, centers_scaled, centers_asli = make_label_map(kmeans, scaler)

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu", ["📊 Overview", "🗺️ Analisis Wilayah", "🔍 Hasil Klasterisasi"])

if menu == "📊 Overview":
    st.title("Dashboard Klasterisasi Banjir DKI Jakarta")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Kategori/Cluster")
        fig, ax = plt.subplots()
        if "kategori_banjir" in df.columns:
            df["kategori_banjir"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("Kategori")
        elif "cluster" in df.columns:
            df["cluster"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("Cluster")
        ax.set_ylabel("Jumlah Kejadian")
        st.pyplot(fig)

    with c2:
        st.subheader("Scatter Plot Klaster")
        fig, ax = plt.subplots()
        hue = "kategori_banjir" if "kategori_banjir" in df.columns else ("cluster" if "cluster" in df.columns else None)
        sns.scatterplot(data=df, x="ketinggian_air_cm", y="jumlah_terdampak_jiwa", hue=hue, ax=ax)
        ax.set_xlabel("Ketinggian Air (cm)")
        ax.set_ylabel("Jumlah Jiwa Terdampak")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📈 Perbandingan Karakteristik Klaster")
    cat = "kategori_banjir" if "kategori_banjir" in df.columns else ("cluster" if "cluster" in df.columns else None)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=cat, y="ketinggian_air_cm", ax=ax)
        ax.set_xlabel("Kategori/Cluster")
        ax.set_ylabel("Ketinggian Air (cm)")
        st.pyplot(fig)

    with c4:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=cat, y="jumlah_terdampak_jiwa", ax=ax)
        ax.set_xlabel("Kategori/Cluster")
        ax.set_ylabel("Jumlah Jiwa")
        st.pyplot(fig)

    st.markdown("### 🔥 Ringkasan Rata-rata Variabel per Klaster")
    cluster_mean = df.groupby(cat)[FITUR].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(cluster_mean, annot=True, fmt=".1f", ax=ax)
    st.pyplot(fig)

elif menu == "🗺️ Analisis Wilayah":
    st.title("Analisis Banjir per Kelurahan")
    st.markdown("---")
    top_n = st.slider("Tampilkan Top N Kelurahan", 5, 20, 10)
    top = agg.sort_values("persentase_banjir_berat", ascending=False).head(top_n)
    st.subheader("Tabel Agregasi Kelurahan")
    st.dataframe(top, use_container_width=True)
    st.subheader("Persentase Banjir Berat")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["kelurahan"], top["persentase_banjir_berat"])
    ax.set_xlabel("Persentase (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

else:
    st.title("Hasil Klasterisasi Data Banjir (Dataset)")
    st.markdown("---")

    X = df[FITUR].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_scaled = scaler.transform(X)

    if "cluster" not in df.columns:
        df["cluster"] = kmeans.predict(X_scaled)

    df["kategori_banjir"] = df["cluster"].map(lmap)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(centers_scaled)
    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    st.subheader("PCA Cluster Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, sub in df.groupby("kategori_banjir"):
        idx = sub.index.to_numpy()
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], s=35, alpha=0.7, label=name)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=260, marker="X", c="black", label="Cluster Centers")
    ax.set_xlabel(f"PC1 ({pc1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% variance)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Ringkasan Cluster")
    ringkas = df.groupby(["cluster", "kategori_banjir"])[FITUR].mean(numeric_only=True).reset_index()
    ukuran = df.groupby(["cluster", "kategori_banjir"]).size().reset_index(name="jumlah_data")
    ringkas = ukuran.merge(ringkas, on=["cluster", "kategori_banjir"]).sort_values("cluster")
    st.dataframe(ringkas, use_container_width=True)

    st.subheader("Detailed Cluster Analysis")
    d_all = np.linalg.norm(X_scaled[:, None, :] - centers_scaled[None, :, :], axis=2)

    for c in range(kmeans.n_clusters):
        nama = lmap.get(c, f"Cluster {c}")
        n = int((df["cluster"] == c).sum())
        with st.expander(f"● Cluster {c} - {nama} ({n} data)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Centroid (skala asli)**")
                st.dataframe(centers_asli.iloc[[c]].T.rename(columns={c: "nilai"}), use_container_width=True)
            with col2:
                st.markdown("**Mean data pada cluster (skala asli)**")
                st.dataframe(X[df["cluster"] == c].mean().to_frame("mean"), use_container_width=True)

            idx_cluster = np.where(df["cluster"].values == c)[0]
            if len(idx_cluster) == 0:
                st.info("Tidak ada data.")
            else:
                dist_c = d_all[idx_cluster, c]
                pick = idx_cluster[np.argsort(dist_c)[:10]]
                cols_show = [col for col in ["kelurahan", "kecamatan", "tanggal", "waktu", "lokasi"] if col in df.columns]
                cols_show += FITUR + ["cluster", "kategori_banjir"]
                st.markdown("**Contoh 10 data paling representatif**")
                st.dataframe(df.iloc[pick][cols_show], use_container_width=True)

    st.markdown("---")
    st.subheader("Download Results")
    st.download_button(
        "Download Clustered Data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="data_banjir_hasil_cluster.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Aplikasi Klasterisasi Banjir - K-Means | Data Publik DKI Jakarta")
