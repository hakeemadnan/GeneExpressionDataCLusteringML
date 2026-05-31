import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from google import genai

st.set_page_config(
    page_title="Gene Expression Clustering",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1a1a2e; font-size: 2rem; font-weight: 700; }
    h2 { color: #16213e; font-size: 1.3rem; font-weight: 600; }
    h3 { color: #0f3460; font-size: 1.1rem; font-weight: 600; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover { opacity: 0.88; transform: translateY(-1px); }
    .ai-box {
        background: #1e1e2e;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        border-top: 3px solid #4f8ef7;
        line-height: 1.75;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = ["#4f8ef7", "#22c48a", "#f5a623", "#e8485a", "#9b59b6", "#1abc9c"]
BORDER_COLORS  = ["#2a6dd9", "#16a06a", "#c47f0a", "#c0273a", "#7d3c98", "#148f77"]

def load_and_preprocess(file):
    df = pd.read_csv(file)
    df = df.set_index(df.columns[0])
    df.index.name = "gene_name"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.apply(lambda x: x.fillna(x.mean()), axis=1)
    return df

def run_clustering(df, linkage_method, n_clusters):
    scaler = StandardScaler()
    df_T = df.T
    scaled = scaler.fit_transform(df_T)
    df_scaled = pd.DataFrame(scaled, index=df_T.index, columns=df_T.columns)

    linked = linkage(scaled, method=linkage_method)
    clusters = fcluster(linked, t=n_clusters, criterion="maxclust")
    df_scaled["Cluster"] = clusters

    pca = PCA(n_components=2)
    X = df_scaled.drop("Cluster", axis=1)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=df_T.index)
    pca_df["Cluster"] = clusters

    return df_scaled, pca_df, linked, clusters

def estimate_optimal_k(linked, max_k=6):
    distances = linked[:, 2]
    accelerations = np.diff(distances, 2)
    k = accelerations[-max_k:].argmax() + 2
    return max(2, min(k, max_k))
def plot_dendrogram(linked):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    dendrogram(linked, ax=ax, truncate_mode="lastp", p=30,
               color_threshold=0.7 * max(linked[:, 2]),
               above_threshold_color="#aaaaaa")
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=13, fontweight="600", color="#1a1a2e")
    ax.set_xlabel("Sample index / cluster size", fontsize=10, color="#555")
    ax.set_ylabel("Distance", fontsize=10, color="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#555")
    plt.tight_layout()
    return fig

def plot_pca(pca_df):
    unique_clusters = sorted(pca_df["Cluster"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    for i, c in enumerate(unique_clusters):
        subset = pca_df[pca_df["Cluster"] == c]
        ax.scatter(subset["PC1"], subset["PC2"],
                   label=f"Cluster {c}",
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   alpha=0.72, s=40, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Principal Component 1", fontsize=10, color="#555")
    ax.set_ylabel("Principal Component 2", fontsize=10, color="#555")
    ax.set_title("PCA — Sample Clusters", fontsize=13, fontweight="600", color="#1a1a2e")
    ax.legend(frameon=True, fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#555")
    plt.tight_layout()
    return fig

def plot_cluster_bar(cluster_counts):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("white")
    labels = [f"Cluster {c}" for c in cluster_counts.index]
    bars = ax.barh(labels, cluster_counts.values,
                   color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(cluster_counts))],
                   height=0.55, edgecolor="white")
    for bar, val in zip(bars, cluster_counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10, color="#333")
    ax.set_xlabel("Number of samples", fontsize=9, color="#555")
    ax.set_title("Samples per cluster", fontsize=11, fontweight="600", color="#1a1a2e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#555")
    plt.tight_layout()
    return fig

def get_ai_analysis(cluster_counts, n_samples, n_genes, n_clusters, linkage_method):
    summary = "\n".join(
        f"  Cluster {c}: {n} samples ({round(n/n_samples*100)}%)"
        for c, n in cluster_counts.items()
    )
    prompt = f"""You are a senior bioinformatics scientist analyzing RNA-seq gene expression data.

Dataset overview:
- Total samples: {n_samples}
- Total genes: {n_genes}
- Clustering method: Hierarchical clustering, {linkage_method} linkage
- Number of clusters identified: {n_clusters}

Cluster distribution:
{summary}

Please provide a clear, scientifically grounded analysis:
1. **Cluster-by-cluster interpretation** — what each cluster likely represents (e.g., cell subtypes, disease states, expression profiles). Use the size and relative proportions as clues.
2. **Overall biological significance** — what does finding {n_clusters} distinct groups suggest about the underlying biology?
3. **Clinical relevance** — potential implications for diagnosis, prognosis, or treatment stratification.
4. **Recommended next steps** — specific downstream analyses (differential expression, pathway enrichment, survival analysis, biomarker discovery, etc.)

Be concise, insightful, and actionable. Use markdown formatting."""
    
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    )
    yield response.text


# ── UI ──────────────────────────────────────────────────────────────────────

st.markdown("## 🧬 Gene Expression Clustering")
st.markdown("Upload an RNA-seq expression CSV. The app preprocesses the data, auto-detects the optimal number of clusters, runs hierarchical clustering, and uses AI to interpret the results.")

st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"], index=0)
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. Upload CSV (genes × samples)
2. Auto-detects optimal *k* from dendrogram
3. Runs hierarchical clustering
4. AI generates cluster insights
""")
    st.markdown("---")
    st.caption("Built with Streamlit · Powered by Gemini")

uploaded_file = st.file_uploader("Upload gene expression CSV", type=["csv"],
                                  help="First column = gene names, remaining columns = samples")

if uploaded_file:
    with st.spinner("Loading and preprocessing data…"):
        try:
            df = load_and_preprocess(uploaded_file)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.stop()

    n_genes, n_samples = df.shape
    st.success(f"✅ Loaded: **{n_genes} genes** × **{n_samples} samples**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Genes", f"{n_genes:,}")
    col2.metric("Samples", f"{n_samples:,}")
    col3.metric("Missing values (after impute)", "0")

    st.divider()

    with st.spinner("Running hierarchical clustering…"):
        df_scaled, pca_df, linked, raw_clusters = run_clustering(df, linkage_method, n_clusters=10)
        optimal_k = estimate_optimal_k(linked)
        df_scaled2, pca_df2, linked2, clusters = run_clustering(df, linkage_method, n_clusters=optimal_k)

    st.markdown(f"### 📊 Results — **{optimal_k} clusters** detected (auto)")

    cluster_counts = pd.Series(clusters).value_counts().sort_index()

    cols = st.columns(min(optimal_k, 6))
    for i, (c, n) in enumerate(cluster_counts.items()):
        with cols[i % len(cols)]:
            pct = round(n / n_samples * 100)
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            border = BORDER_COLORS[i % len(BORDER_COLORS)]
            st.markdown(f"""
<div class="metric-card" style="border-left-color:{border};">
  <div style="font-size:0.8rem;color:#888;font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Cluster {c}</div>
  <div style="font-size:2rem;font-weight:700;color:{color};">{n}</div>
  <div style="font-size:0.85rem;color:#666;">{pct}% of samples</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 PCA Visualization", "🌿 Dendrogram", "📊 Distribution"])

    with tab1:
        fig_pca = plot_pca(pca_df2)
        st.pyplot(fig_pca, use_container_width=True)

    with tab2:
        fig_dend = plot_dendrogram(linked2)
        st.pyplot(fig_dend, use_container_width=True)
        st.caption(f"Cut point for **{optimal_k} clusters**")

    with tab3:
        fig_bar = plot_cluster_bar(cluster_counts)
        st.pyplot(fig_bar, use_container_width=True)

    st.divider()

    st.markdown("### 🤖 AI Cluster Analysis")
    st.markdown("Gemini interprets the biological meaning of your clusters.")

    if st.button("✨ Generate AI Analysis"):
        with st.spinner("Generating insights…"):
            full_text = ""
            placeholder = st.empty()
            for chunk in get_ai_analysis(cluster_counts.to_dict(), n_samples, n_genes, optimal_k, linkage_method):
                full_text += chunk
                placeholder.markdown(f'<div class="ai-box">{full_text}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💾 Download Results")
    result_df = pca_df2.copy()
    result_df.index.name = "sample"
    csv_bytes = result_df.to_csv().encode()
    st.download_button("⬇️ Download cluster assignments (CSV)", data=csv_bytes,
                       file_name="cluster_assignments.csv", mime="text/csv")

else:
    st.info("👆 Upload a CSV file to get started.")
    st.markdown("""
**Expected format:**
```
gene_name, sample_1, sample_2, sample_3, ...
GENE_A,    1.23,     0.45,     2.10,     ...
GENE_B,    0.89,     1.67,     0.33,     ...
```
""")
