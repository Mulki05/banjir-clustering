import io, re, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Klasterisasi Banjir DKI", layout="wide")

FEATURES = [
    "ketinggian_air_cm",
    "jumlah_terdampak_rt",
    "jumlah_terdampak_kk",
    "jumlah_terdampak_jiwa",
    "jumlah_pengungsi_tertinggi",
]
LOG_FEATURES = [
    "jumlah_terdampak_rt",
    "jumlah_terdampak_kk",
    "jumlah_terdampak_jiwa",
    "jumlah_pengungsi_tertinggi",
]

def _cm(x) -> int:
    nums = re.findall(r"\d+", str(x)) if pd.notna(x) else []
    return int(max(map(int, nums))) if nums else 0

@st.cache_data
def get_df(path="banjir.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ketinggian_air_cm"] = df["ketinggian_air"].apply(_cm)
    for c in FEATURES:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    return df

def make_X(df: pd.DataFrame, use_log: bool) -> np.ndarray:
    X = df[FEATURES].to_numpy(float)
    if use_log:
        cols = [FEATURES.index(c) for c in LOG_FEATURES if c in FEATURES]
        X[:, cols] = np.log1p(X[:, cols])
    return X

def label_map(kmeans: KMeans, scaler: StandardScaler) -> tuple[dict[int, str], pd.DataFrame]:
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
    score = (
        np.log1p(centers["jumlah_terdampak_jiwa"]) * 1.0
        + np.log1p(centers["jumlah_terdampak_kk"]) * 0.7
        + np.log1p(centers["jumlah_pengungsi_tertinggi"]) * 0.5
        + np.log1p(centers["ketinggian_air_cm"]) * 0.2
        + np.log1p(centers["jumlah_terdampak_rt"]) * 0.2
    ).to_numpy()

    thr = float(np.quantile(score, 2 / 3))
    heavy = set(map(int, np.where(score >= thr)[0])) or {int(np.argmax(score))}
    m = {i: ("Banjir Berat" if i in heavy else "Banjir Ringan") for i in range(kmeans.n_clusters)}

    centers["severity_score"], centers["label"] = score, centers.index.map(m)
    return m, centers.sort_values("severity_score", ascending=False)

def agg_kelurahan(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("kelurahan", dropna=False)
    out = pd.DataFrame({
        "total_kejadian": g.size(),
        "jumlah_banjir_berat": g.apply(lambda x: (x["kategori_banjir"] == "Banjir Berat").sum()),
    }).reset_index()
    out["persentase_banjir_berat"] = np.where(
        out["total_kejadian"] > 0,
        out["jumlah_banjir_berat"] / out["total_kejadian"] * 100,
        0.0
    )
    return out.sort_values("persentase_banjir_berat", ascending=False)

# ============ Sidebar ============
st.sidebar.title("Navigasi")
k = st.sidebar.slider("Jumlah cluster", 2, 10, 4)
use_log = st.sidebar.checkbox("Gunakan log1p", True)
warn_min = st.sidebar.number_input("Peringatan cluster kecil (< data)", 1, 50, 5)
menu = st.sidebar.radio("Menu", ["Overview", "Analisis Wilayah", "Hasil Klasterisasi"])

# ============ Data + Clustering (DINAMIS SAJA) ============
df0 = get_df()
X = make_X(df0, use_log)

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(Xs)

df = df0.copy()
df["cluster"] = kmeans.labels_
lm, centers = label_map(kmeans, scaler)
df["kategori_banjir"] = df["cluster"].map(lm)

sizes = df["cluster"].value_counts().sort_index()
if (sizes < int(warn_min)).any():
    st.info("Ada cluster kecil. Coba turunkan k atau aktifkan log1p agar lebih stabil.")

# ============ Pages ============
st.title("Dashboard Klasterisasi Banjir DKI Jakarta")
st.caption(f"k aktif: {k} | log1p: {use_log}")

if menu == "Overview":
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        df["kategori_banjir"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="ketinggian_air_cm",
            y="jumlah_terdampak_jiwa",
            hue="kategori_banjir",
            ax=ax,
            s=35
        )
        ax.set_xlabel("Ketinggian (cm)")
        ax.set_ylabel("Jiwa")
        st.pyplot(fig)

    st.subheader("Ukuran Cluster")
    fig, ax = plt.subplots()
    ax.bar(sizes.index.astype(str), sizes.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Data")
    st.pyplot(fig)

elif menu == "Analisis Wilayah":
    top_n = st.slider("Top N Kelurahan", 5, 30, 10)
    top = agg_kelurahan(df).head(top_n)
    st.dataframe(top, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["kelurahan"].astype(str), top["persentase_banjir_berat"])
    ax.set_xlabel("Persentase Banjir Berat (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

else:
    st.subheader("Ringkasan Cluster")
    summ = (
        df.groupby(["cluster", "kategori_banjir"])
        .agg(jumlah_data=("cluster", "size"), **{c: (c, "mean") for c in FEATURES})
        .reset_index()
    )
    st.dataframe(
        summ.sort_values(["kategori_banjir", "jumlah_data"], ascending=[True, False]).round(2),
        use_container_width=True
    )

    st.subheader("Centroid & Severity")
    st.dataframe(centers.round(2), use_container_width=True)

    st.subheader("PCA (ringkas)")
    Z = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df["kategori_banjir"], ax=ax, s=30, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.subheader("Download")
    st.download_button(
        "Clustered CSV",
        df.to_csv(index=False).encode("utf-8"),
        f"data_cluster_k{k}.csv",
        "text/csv"
    )

    buf = io.BytesIO()
    pickle.dump(kmeans, buf)
    st.download_button(
        "Model PKL",
        buf.getvalue(),
        f"kmeans_k{k}.pkl",
        "application/octet-stream"
    )

    buf = io.BytesIO()
    pickle.dump(scaler, buf)
    st.download_button(
        "Scaler PKL",
        buf.getvalue(),
        "scaler.pkl",
        "application/octet-stream"
    )
