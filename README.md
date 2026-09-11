

# Gene Expression Analysis using Hierarchical Clustering

[Live Demo](https://geneclustering.streamlit.app/)

An AI-powered gene expression analysis system that uses hierarchical clustering and dimensionality reduction to identify patterns in high-dimensional RNA-seq data, with Generative AI assistance for interpreting clustering results.

## Overview

Gene expression datasets can contain thousands of gene features, making it difficult to identify meaningful patterns directly from the raw data.

This project provides an end-to-end analytical workflow that:

- Preprocesses and standardizes high-dimensional gene expression data
- Identifies gene expression patterns using hierarchical clustering
- Automatically determines an appropriate number of clusters using silhouette score
- Uses PCA for dimensionality reduction and visualization
- Generates dendrograms and cluster visualizations for interpretation
- Uses Gemini 2.5 Flash to convert analytical results into concise, understandable insights
- Provides an interactive Streamlit interface for dataset upload and analysis

## Key Features

### 1. Data Preprocessing
- Loads gene expression data from CSV files
- Handles feature selection and data preparation
- Standardizes numerical features using `StandardScaler`

### 2. Hierarchical Clustering
- Applies Agglomerative Hierarchical Clustering
- Evaluates different cluster configurations
- Selects the optimal number of clusters using silhouette score

### 3. Dimensionality Reduction
Principal Component Analysis (PCA) is used to reduce high-dimensional gene expression data into lower-dimensional representations for visualization and pattern analysis.

### 4. Visualization
The application generates visualizations including:

- Hierarchical clustering dendrogram
- PCA-based cluster visualization
- Cluster distribution plots
- Gene expression pattern visualizations

### 5. Generative AI Analysis
Gemini 2.5 Flash is integrated into the workflow to interpret clustering results and generate concise analytical summaries.

The AI-assisted component helps translate technical clustering outputs into more understandable insights.

### 6. Interactive Web Application
The complete workflow is available through a Streamlit web application that allows users to:

1. Upload a gene expression dataset
2. Preprocess the data
3. Perform clustering
4. Visualize the results
5. Generate AI-assisted interpretations

## Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **SciPy**
- **Streamlit**
- **Google Gemini API**
- **Matplotlib**
- **Seaborn**

## Methodology

```text
Gene Expression Dataset
          ↓
Data Loading & Preprocessing
          ↓
Feature Standardization
          ↓
Hierarchical Clustering
          ↓
Silhouette Score Evaluation
          ↓
Optimal Cluster Selection
          ↓
PCA Dimensionality Reduction
          ↓
Cluster Visualization
          ↓
Gemini 2.5 Flash
          ↓
AI-Assisted Analytical Insights
```
## Model Evaluation

The clustering quality is evaluated using the **Silhouette Score**, which measures how well each data point fits within its assigned cluster compared with other clusters.

The application evaluates different cluster counts and selects the number of clusters with the best silhouette score.

## Generative AI Integration

The project integrates **Google Gemini 2.5 Flash** to interpret the clustering results and generate human-readable analytical insights.

The AI component provides:
- Explanation of identified gene expression clusters
- Interpretation of observed patterns
- Possible biological significance
- Suggested next steps for analysis
- Concise summaries of complex clustering results

## Project Structure

GeneExpressionDataCLusteringML/
├── app.py
├── GeneExpressionData.ipynb
├── dataset/
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml

## Installation

### 1. Clone the Repository

git clone https://github.com/hakeemadnan/GeneExpressionDataCLusteringML.git
cd GeneExpressionDataCLusteringML

### 2. Create a Virtual Environment

python -m venv venv

Activate the environment:

**Windows:**
venv\Scripts\activate

**Linux/macOS:**
source venv/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt

## Gemini API Configuration

Create the file:

.streamlit/secrets.toml

Add your API key:

GEMINI_API_KEY = "your_api_key_here"

> Do not upload or commit your API key to GitHub.

## Run the Application

streamlit run app.py

## How to Use

1. Upload an RNA-seq gene expression CSV dataset.
2. Select the appropriate data orientation if required.
3. Run the preprocessing pipeline.
4. The application standardizes the numerical data.
5. Hierarchical clustering is performed.
6. The optimal number of clusters is determined using the Silhouette Score.
7. View the dendrogram and cluster visualizations.
8. Explore the PCA-based representation of the clusters.
9. Use Gemini 2.5 Flash to generate an interpretation of the clustering results.

## Use Cases

- Gene expression pattern discovery
- Exploratory analysis of genomic datasets
- Unsupervised learning demonstrations
- Biological data visualization
- AI-assisted interpretation of machine learning results
- Educational applications in bioinformatics and data science

## Skills Demonstrated

- Python Programming
- Data Preprocessing
- Exploratory Data Analysis
- Unsupervised Machine Learning
- Hierarchical Clustering
- Dimensionality Reduction using PCA
- Model Evaluation using Silhouette Score
- Data Visualization
- Generative AI Integration
- Gemini API Integration
- Streamlit Application Development

## Author

**Adnan Mushtaq**

- GitHub: https://github.com/hakeemadnan
- LinkedIn: https://www.linkedin.com/in/hakeemadnan
- 
