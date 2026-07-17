import os
import pandas as pd

from preprocessing import preprocess_data
from utils import generate_all_plots

from model import (
    elbow_method,
    train_kmeans,
    pca_visualization,
    cluster_statistics,
    compare_with_actual,
    save_model,
    save_dataset
)


# ----------------------------------------------
# Generate Text Report
# ----------------------------------------------

def generate_report(df, silhouette):

    os.makedirs("output/reports", exist_ok=True)

    report_path = "output/reports/traffic_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("="*70+"\n")
        f.write("SMART CITY AI - TRAFFIC PATTERN CLUSTERING REPORT\n")
        f.write("="*70+"\n\n")

        f.write(f"Total Records : {len(df)}\n")
        f.write(f"Number of Clusters : {df['Cluster'].nunique()}\n")
        f.write(f"Silhouette Score : {silhouette:.3f}\n\n")

        f.write("Average Vehicle Count\n")
        f.write("-"*50+"\n")

        f.write(f"Average Cars   : {df['CarCount'].mean():.2f}\n")
        f.write(f"Average Bikes  : {df['BikeCount'].mean():.2f}\n")
        f.write(f"Average Buses  : {df['BusCount'].mean():.2f}\n")
        f.write(f"Average Trucks : {df['TruckCount'].mean():.2f}\n")
        f.write(f"Average Total  : {df['Total'].mean():.2f}\n\n")

        f.write("Cluster Distribution\n")
        f.write("-"*50+"\n")

        counts = df["Cluster"].value_counts().sort_index()

        for c, count in counts.items():
            percent = count / len(df) * 100
            f.write(f"Cluster {c} : {count} ({percent:.2f}%)\n")

        f.write("\n")

        f.write("Cluster vs Traffic Situation\n")
        f.write("-"*50+"\n")

        f.write(str(pd.crosstab(
            df["Cluster"],
            df["Traffic Situation"]
        )))

        f.write("\n\n")

        f.write("SMART CITY AI RECOMMENDATIONS\n")
        f.write("-"*50+"\n")

        f.write("• Increase green signal duration during heavy traffic.\n")
        f.write("• Suggest alternative routes using GPS.\n")
        f.write("• Deploy traffic police in congested areas.\n")
        f.write("• Prioritize emergency vehicles.\n")
        f.write("• Use adaptive traffic signal systems.\n")

    print("\nReport Generated Successfully")

    report_path = "output/reports"

    os.makedirs(report_path, exist_ok=True)

    report_file = os.path.join(
        report_path,
        "traffic_report.txt"
    )

    cluster_count = df["Cluster"].value_counts().sort_index()

    with open(report_file, "w") as f:

        f.write("="*60+"\n")
        f.write("TRAFFIC PATTERN CLUSTERING REPORT\n")
        f.write("="*60+"\n\n")

        f.write(f"Total Records : {len(df)}\n")
        f.write(f"Total Clusters : {df['Cluster'].nunique()}\n")
        f.write(f"Silhouette Score : {silhouette:.3f}\n\n")

        f.write("Cluster Distribution\n")
        f.write("----------------------\n")

        for cluster,count in cluster_count.items():

            percent = count/len(df)*100

            f.write(
                f"Cluster {cluster} : {count} ({percent:.2f}%)\n"
            )

        f.write("\n")

        f.write("Traffic Situation Distribution\n")
        f.write("-------------------------------\n")

        f.write(
            str(
                pd.crosstab(
                    df["Cluster"],
                    df["Traffic Situation"]
                )
            )
        )

        f.write("\n\n")

        f.write("Project Summary\n")
        f.write("----------------\n")

        f.write(
            "Traffic patterns were grouped using "
            "K-Means Clustering. "
            "The generated clusters show different "
            "traffic behaviors based on vehicle counts "
            "and total traffic."
        )

    print("\nReport Generated Successfully")


# ----------------------------------------------
# Complete Pipeline
# ----------------------------------------------

def run_pipeline(data_path):

    print("="*60)
    print("TRAFFIC PATTERN CLUSTERING")
    print("="*60)

    original_df, encoded_df, scaled_df, scaler = preprocess_data(data_path)

    print("\nCreating Visualizations...")

    generate_all_plots(original_df)

    print("\nFinding Optimal K...")

    elbow_method(original_df)

    clustered_df, model, score = train_kmeans(original_df)

    print("\nGenerating PCA Plot...")

    pca_visualization(clustered_df)

    print("\nGenerating Statistics...")

    cluster_statistics(clustered_df)

    compare_with_actual(clustered_df)

    save_model(model)

    save_dataset(clustered_df)

    generate_report(clustered_df, score)

    print("\nPipeline Finished Successfully")

    return clustered_df