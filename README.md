# Traffic Pattern Clustering for Smart City using AI

A Machine Learning project that analyzes urban traffic data using **K-Means Clustering** to discover traffic patterns and generate insights for Smart City traffic management. The project performs data preprocessing, exploratory data analysis (EDA), feature engineering, clustering, visualization, and automated report generation to help understand congestion trends and support intelligent transportation planning.

---

## Project Overview

Urban traffic management is one of the major challenges in modern smart cities. Understanding traffic patterns can help authorities optimize signal timings, reduce congestion, improve emergency response, and enhance road infrastructure planning.

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to group similar traffic conditions based on vehicle counts and time-related features. The complete pipeline includes preprocessing, visualization, clustering, evaluation, and report generation.

---

## Features

- Data preprocessing and cleaning
- Duplicate and missing value handling
- Feature engineering (Hour extraction from Time)
- Exploratory Data Analysis (EDA)
- Correlation analysis
- K-Means clustering
- Elbow Method for optimal cluster selection
- Silhouette Score evaluation
- PCA-based cluster visualization
- Cluster statistics generation
- Automatic report generation
- Trained model export (.pkl)
- Clustered dataset export (.csv)
- Smart City traffic recommendations

---

## Machine Learning Workflow

```
Traffic Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Scaling
        │
        ▼
K-Means Clustering
        │
        ▼
Elbow Method
        │
        ▼
Silhouette Score
        │
        ▼
PCA Visualization
        │
        ▼
Cluster Analysis
        │
        ▼
Report Generation
        │
        ▼
Smart City AI Recommendations
```

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| Time | Traffic recording time |
| Date | Date of observation |
| Day of the week | Weekday information |
| CarCount | Number of cars |
| BikeCount | Number of bikes |
| BusCount | Number of buses |
| TruckCount | Number of trucks |
| Total | Total vehicle count |
| Traffic Situation | Existing traffic category (used only for comparison) |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Project Structure

```
Traffic_Clustering_Project
│
├── data/
│   └── Traffic.csv
│
├── output/
│   ├── plots/
│   ├── reports/
│   ├── clustered_traffic.csv
│   ├── cluster_statistics.csv
│   ├── cluster_vs_actual.csv
│   └── traffic_model.pkl
│
├── preprocessing.py
├── utils.py
├── model.py
├── train.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Traffic-Pattern-Clustering-Smart-City-AI.git
```

Move into the project directory

```bash
cd Traffic-Pattern-Clustering-Smart-City-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python main.py
```

---

## Generated Outputs

After execution, the project automatically generates:

### Visualizations

- Correlation Heatmap
- Missing Value Heatmap
- Histograms
- Boxplots
- Hour-wise Traffic Analysis
- Weekday-wise Traffic Analysis
- Traffic Distribution
- Elbow Method
- PCA Cluster Visualization

### Reports

- Cluster Statistics
- Cluster vs Traffic Situation
- Smart City AI Report

### Saved Files

- Clustered Dataset (.csv)
- Trained K-Means Model (.pkl)

---

## Why K-Means?

This project focuses on discovering **natural traffic patterns** without predefined labels.

K-Means was selected because it:

- Works efficiently on numerical traffic data
- Automatically groups similar traffic conditions
- Is computationally fast
- Is widely used for clustering and pattern discovery
- Supports Smart City traffic analysis

---

## Model Evaluation

The clustering quality is evaluated using:

- **Elbow Method** to determine the optimal number of clusters.
- **Silhouette Score** to measure cluster cohesion and separation.

---

## Smart City Applications

The generated traffic clusters can support:

- Intelligent Traffic Signal Control
- Congestion Detection
- Traffic Flow Analysis
- Route Optimization
- Emergency Vehicle Prioritization
- Urban Infrastructure Planning
- Smart Transportation Systems

---

## Future Improvements

- Real-time traffic prediction
- Live IoT sensor integration
- GPS-based route recommendation
- Deep Learning-based traffic forecasting
- Interactive Streamlit dashboard
- Cloud deployment

---

## Learning Outcomes

Through this project, the following concepts were implemented:

- Data Preprocessing
- Feature Engineering
- Exploratory Data Analysis
- Unsupervised Machine Learning
- K-Means Clustering
- Feature Scaling
- PCA
- Cluster Evaluation
- Data Visualization
- Automated Report Generation

---

## Author

**Rajanya Saha**

B.Tech CSE (AI & ML)


---

If you found this project useful, consider giving it a ⭐ on GitHub.
