# Project Overview

This project aims to classify glioma subtypes using data from The Cancer Genome Atlas (TCGA). The primary objectives include:
- Developing machine learning models to predict glioma classes based on metadata and processed gene expression data.
- Conducting survival analysis to assess the impact of biomarkers on patient survival.
- Implementing a CNN-based classification model to identify key biomarkers.

# Directory Structure

This project follows two distinct approaches for biomarker classification:

### 1. Grading Classification Using CNN
- Utilizes EfficientNet, a pre-trained CNN model, to classify biomarkers using medical images.
- Image modalities considered:
  - **Flair**
  - **T1w**
  - **T1wCE**
  - **T2w**

### 2. Pyradiomics-Based Classification
- Leverages radiomic features extracted using the `gdc-client` tool.
- **Directory Breakdown:**
  - `Pyradimics-classification/scr/`: Contains scripts and Jupyter notebooks for preprocessing, model training, evaluation, and deployment.
  - `model_evaluation.ipynb`: Notebook for model evaluation, including data preprocessing, training, performance analysis, and feature importance visualization.
  - `preprocess_data_labels.ipynb`: Notebook for preparing and labeling data.
  - `streamlit_app.py`: Python script for deploying the trained model via Streamlit.
  - `get_data.sh`: Shell script to download TCGA data using the `gdc-client` tool.
  - `survival_analysis.ipynb`: Notebook for survival analysis using Cox Proportional Hazards models to assess the impact of biomarkers on patient survival.
  
### 3. Model Storage and Processed Data
- `models/`: Contains saved pre-trained models and scalers.
  - `scaler.pkl`: Pre-processing scaler for data normalization.
  - `xgb_model.pkl`: Pre-trained XGBoost classification model.
- `processed/`: Stores cleaned datasets for training and testing.
  - `data.csv`: Processed gene expression data.
  - `glioma_labels.csv`: Corresponding labels for glioma classification.
- `metadata/`: Contains metadata related to TCGA samples.
  - `clinical.tsv`: Clinical metadata.
  - `gdc_sample_sheet.tsv`: Sample details from GDC.
  
### 4. Root Directory Files
- `gdc_manifest.txt`: Manifest file listing TCGA dataset files.

# Model Training and Evaluation

### Data Preprocessing
- Loads gene expression data and glioma labels.
- Removes genes with no variance across samples.
- Handles missing values and normalizes data before transposing it for model input.
- Ensures class balance by analyzing label distributions.

### Machine Learning Models
- **Logistic Regression**: Used as a baseline for comparison.
- **Random Forest**: Trained to classify glioma subtypes and analyze feature importance.
- **XGBoost Classifier**: Outperforms Random Forest slightly but struggles to distinguish between "Oligodendroglioma" and "Astrocytoma."
- Model evaluation includes:
  - 5-fold cross-validation
  - Metrics: F1-score, Accuracy, AUC-ROC, and Confusion Matrices

### Model Storage
- The trained **XGBoost model** and **scaler** are saved as `.pkl` files for future use.

# Survival Analysis
- Conducted using the `survival_analysis.ipynb` notebook.
- **Cox Proportional Hazards Model** assesses how biomarkers influence survival rates.
- Identifies significant biomarkers correlated with prolonged or reduced survival.
- Addresses multicollinearity using correlation matrices to refine the feature set.

# Setup Instructions

### 1. Install Dependencies
Use the provided `environment.yml` file to set up a Conda environment with all required dependencies:
```bash
conda env create -f scr/environment.yml
conda activate JupyterLab
```

### 2. Download Data
Run the `get_data.sh` script to fetch data from GDC:
```bash
bash Pyradimics-classification/scr/get_data.sh
```

### 3. Data Preprocessing
Execute the `preprocess_data_labels.ipynb` notebook to clean raw gene expression data and generate classification labels.

### 4. Model Training & Evaluation
Train and evaluate models using the `model_evaluation.ipynb` notebook.

### 5. Survival Analysis
Analyze survival-related biomarkers using the `survival_analysis.ipynb` notebook and clinical data.

# Usage

### 1. Data Preprocessing
Use `preprocess_data_labels.ipynb` to clean and prepare the dataset.

### 2. Model Evaluation
The `model_evaluation.ipynb` notebook provides performance metrics for classification models.

### 3. Survival Analysis
The `survival_analysis.ipynb` notebook helps investigate the relationship between biomarkers and patient survival outcomes.

