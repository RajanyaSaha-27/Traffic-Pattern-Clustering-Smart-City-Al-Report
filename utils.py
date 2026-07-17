import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

OUTPUT_DIR = "output/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------
# Save Figure
# ---------------------------------------------------

def save_plot(filename):

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


# ---------------------------------------------------
# Dataset Overview
# ---------------------------------------------------

def plot_missing_values(df):

    plt.figure(figsize=(10,5))

    sns.heatmap(df.isnull(),
                cbar=False,
                cmap="viridis")

    plt.title("Missing Values Heatmap")

    save_plot("missing_values.png")


# ---------------------------------------------------
# Numeric Distribution
# ---------------------------------------------------

def plot_histograms(df):

    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:

        plt.figure(figsize=(7,4))

        sns.histplot(df[col],
                     kde=True,
                     bins=30)

        plt.title(f"{col} Distribution")

        save_plot(f"hist_{col}.png")


# ---------------------------------------------------
# Boxplots
# ---------------------------------------------------

def plot_boxplots(df):

    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:

        plt.figure(figsize=(6,4))

        sns.boxplot(x=df[col])

        plt.title(f"{col} Boxplot")

        save_plot(f"box_{col}.png")


# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------

def plot_correlation(df):

    plt.figure(figsize=(10,8))

    corr = df.corr(numeric_only=True)

    sns.heatmap(corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f")

    plt.title("Correlation Heatmap")

    save_plot("correlation_heatmap.png")


# ---------------------------------------------------
# Categorical Count Plots
# ---------------------------------------------------

def plot_categorical(df):

    categorical_columns = df.select_dtypes(include="object").columns

    for col in categorical_columns:

        plt.figure(figsize=(8,5))

        sns.countplot(data=df,
                      x=col)

        plt.xticks(rotation=45)

        plt.title(col)

        save_plot(f"{col}_count.png")


# ---------------------------------------------------
# Hourly Traffic
# ---------------------------------------------------

def plot_hourly_traffic(df):

    if "Hour" not in df.columns:
        return

    plt.figure(figsize=(10,5))

    sns.lineplot(data=df,
                 x="Hour",
                 y="Total",
                 estimator="mean",
                 errorbar=None)

    plt.title("Average Traffic by Hour")

    save_plot("hourly_traffic.png")


# ---------------------------------------------------
# Weekday Traffic
# ---------------------------------------------------

def plot_weekday_traffic(df):

    if "Weekday" not in df.columns:
        return

    plt.figure(figsize=(10,5))

    sns.barplot(data=df,
                x="Day of the week",
                y="Total",
                estimator="mean")

    plt.xticks(rotation=30)

    plt.title("Average Traffic by Weekday")

    save_plot("weekday_traffic.png")


# ---------------------------------------------------
# Pie Chart
# ---------------------------------------------------

def plot_traffic_distribution(df):

    if "Traffic Situation" not in df.columns:
        return

    counts = df["Traffic Situation"].value_counts()

    plt.figure(figsize=(7,7))

    plt.pie(counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90)

    plt.title("Traffic Situation Distribution")

    save_plot("traffic_pie.png")


# ---------------------------------------------------
# Cluster Plot
# ---------------------------------------------------

def plot_clusters(df):

    if "Cluster" not in df.columns:
        return

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x="Hour",
        y="Total",
        hue="Cluster",
        palette="Set2"
    )

    plt.title("Traffic Clusters")

    save_plot("traffic_clusters.png")


# ---------------------------------------------------
# Cluster Count
# ---------------------------------------------------

def plot_cluster_count(df):

    if "Cluster" not in df.columns:
        return

    plt.figure(figsize=(7,5))

    sns.countplot(data=df,
                  x="Cluster")

    plt.title("Cluster Distribution")

    save_plot("cluster_count.png")


# ---------------------------------------------------
# All Visualizations
# ---------------------------------------------------

def generate_all_plots(df):

    print("\nGenerating Visualizations...\n")

    plot_missing_values(df)

    plot_histograms(df)

    plot_boxplots(df)

    plot_correlation(df)

    plot_categorical(df)

    plot_hourly_traffic(df)

    plot_weekday_traffic(df)

    plot_traffic_distribution(df)

    print("All plots saved successfully!")