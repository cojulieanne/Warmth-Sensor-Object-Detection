# Warmth-Sensor Object Detection

This project explores material/object classification using warmth sensor data. The workflow applies dimensionality reduction and machine learning methods such as **Principal Component Analysis (PCA)** and **Support Vector Machine (SVM)** to identify whether different materials produce distinguishable thermal-response signatures.

The sensor setup uses two warmth-sensing channels, such as top and bottom sensor responses, to capture material-dependent thermal behavior over time.

---

## Project Structure

```text
Warmth-Sensor-Object-Detection/
├── data/
│   └── raw/
│       └── *.csv
├── notebooks/
│   └── 01_exploration.ipynb
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

## Setup

This project uses uv for environment and dependency management.

1. Clone the repository
 ```bash
 git clone git@github.com:your-username/Warmth-Sensor-Object-Detection.git
cd Warmth-Sensor-Object-Detection
```

2. Create and sync the environment
```bash
uv sync
```

## Workflow

1. **Data preparation**  
   Load the raw warmth sensor CSV files, clean the data, and organize each material/object by trial.

2. **Feature extraction**  
   Convert the primary and secondary sensor time-series signals into numerical features using manual descriptors, statistical features, reservoir computing, or ROCKET-style embeddings.

3. **Dimensionality reduction**  
   Apply PCA to reduce the feature space and visualize material/object separation using the first two principal components.

4. **Classification**  
   Train an SVM classifier on the PCA-transformed features to estimate generalized decision regions between material/object classes.

5. **Visualization and analysis**  
   Plot the sensor responses, PCA projections, explained variance, factor loadings, and SVM decision regions to assess class separability.