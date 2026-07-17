import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


OUTPUT_DIR = "output"
PLOT_DIR = "output/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------------------------------------
# Features used for clustering
# -------------------------------------------------

FEATURES = [
    "CarCount",
    "BikeCount",
    "BusCount",
    "TruckCount",
    "Total",
    "Hour"
]


# -------------------------------------------------
# Elbow Method
# -------------------------------------------------

def elbow_method(df):

    X = df[FEATURES]

    inertia = []

    k_values = range(2,11)

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        inertia.append(model.inertia_)

    plt.figure(figsize=(7,5))

    plt.plot(k_values, inertia, marker="o")

    plt.xlabel("Number of Clusters")

    plt.ylabel("Inertia")

    plt.title("Elbow Method")

    plt.grid(True)

    plt.savefig(f"{PLOT_DIR}/elbow_method.png",dpi=300)

    plt.close()

    return inertia


# -------------------------------------------------
# Train Model
# -------------------------------------------------

def train_kmeans(df,n_clusters=4):

    X = df[FEATURES]

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    df = df.copy()

    df["Cluster"] = labels

    score = silhouette_score(X,labels)

    print("="*60)
    print("KMeans Training Completed")
    print("="*60)
    print(f"Clusters : {n_clusters}")
    print(f"Silhouette Score : {score:.3f}")

    return df,model,score


# -------------------------------------------------
# PCA Visualization
# -------------------------------------------------

def pca_visualization(df):

    X = df[FEATURES]

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X)

    plt.figure(figsize=(8,6))

    plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=df["Cluster"],
        cmap="viridis"
    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("Traffic Pattern Clusters (PCA)")

    plt.colorbar(label="Cluster")

    plt.savefig(f"{PLOT_DIR}/pca_clusters.png",dpi=300)

    plt.close()


# -------------------------------------------------
# Cluster Statistics
# -------------------------------------------------

def cluster_statistics(df):

    summary = df.groupby("Cluster")[FEATURES].mean()

    print("\nCluster Statistics\n")

    print(summary)

    summary.to_csv(
        f"{OUTPUT_DIR}/cluster_statistics.csv"
    )

    return summary


# -------------------------------------------------
# Compare with Traffic Situation
# -------------------------------------------------

def compare_with_actual(df):

    table = pd.crosstab(
        df["Cluster"],
        df["Traffic Situation"]
    )

    print("\nCluster vs Traffic Situation\n")

    print(table)

    table.to_csv(
        f"{OUTPUT_DIR}/cluster_vs_actual.csv"
    )


# -------------------------------------------------
# Save Model
# -------------------------------------------------

def save_model(model):

    joblib.dump(
        model,
        f"{OUTPUT_DIR}/traffic_model.pkl"
    )

    print("\nModel Saved Successfully")


# -------------------------------------------------
# Save Clustered Dataset
# -------------------------------------------------

def save_dataset(df):

    df.to_csv(
        f"{OUTPUT_DIR}/clustered_traffic.csv",
        index=False
    )

    print("Clustered Dataset Saved")


# -------------------------------------------------
# Complete Pipeline
# -------------------------------------------------

def clustering_pipeline(df):

    elbow_method(df)

    clustered_df,model,score = train_kmeans(df)

    pca_visualization(clustered_df)

    cluster_statistics(clustered_df)

    compare_with_actual(clustered_df)

    save_model(model)

    save_dataset(clustered_df)

    return clustered_df,model