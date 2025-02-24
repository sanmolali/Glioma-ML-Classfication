# Project Overview

This project is divided into 2 parts
  - 1st part aims to classify glioma subtypes using data from The Cancer Genome Atlas (TCGA). The primary objectives include:
- Developing machine learning models to predict glioma subtypes based on metadata and processed gene expression data.
- Conducting survival analysis to assess the impact of biomarkers on patient survival.
- The 2nd part of the project aims to predict the genetic profile of the most severe form of brain cancer, i.e. glioblastoma.
    - for this purpose, data is sourced from RSNA and MICCAI, (link:  [RSNA-MICCAI Brain Tumor Radiogenomic Classification](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data?select=train_labels.csv).) the focus is to detect the presence of a key biomarker MGMT promoter methylation.
  - Implementing a CNN-based classification model to identify key biomarkers of glioblastoma. 

# Directory Structure

This project follows two distinct approaches for biomarker identification:

### 1. Glioblastoma genetic profile using CNN:
- Utilizes EfficientNet, a pre-trained CNN model, to identify biomarkers using medical images.
- glioblastoma MRI :
     ![glioblastoma](https://github.com/user-attachments/assets/335f986e-a559-43b9-be9a-d17d936853d6)

  
  ### CNN architecture:
    [MRI-Based Radiogenomic Classification.docx](https://github.com/user-attachments/files/18947214/MRI-Based.Radiogenomic.Classification.docx)

- Image modalities considered:
  - **Flair**
  - **T1w**
  - **T1wCE**
  - **T2w**

#### Modalities:

![Screenshot from 2025-02-23 04-38-46](https://github.com/user-attachments/assets/311c0315-49fe-44a5-86da-91103e104c63)

### 2. Pyradiomics-Based subtypes classification of glioma:
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

### Data Visualizations
`PCA of Scaled Gene Expression Data on Unselected Features`



![image](https://github.com/user-attachments/assets/f153cf21-146d-45fb-b3eb-c5cd919e57da)



`UMAP of Gene Expression Data with Unselected Features`



![image](https://github.com/user-attachments/assets/1b196890-3f18-43e2-9b1a-53e06f9e3e24)



`PCA of Scaled Gene Expression Data with Selected Features`



![image](https://github.com/user-attachments/assets/912a6c7d-3e7b-455f-85eb-337ba1c0177a)



UMAP of Gene Expression Data with Selected Features



![image](https://github.com/user-attachments/assets/253571ff-c0a6-4c01-a086-19369e93d6ac)


### Machine Learning Models
- **Logistic Regression**: Used as a baseline for comparison.

   ` Confusion Matrix Results:`


    ![image](https://github.com/user-attachments/assets/5c2df6d2-d3f2-448d-a17f-8f6b47f3a8c5)


- **Random Forest**: Trained to classify glioma subtypes and analyze feature importance.
    Feature Importance Chart of Random Forest



    ![image](https://github.com/user-attachments/assets/afd91275-47a4-40e5-bd74-e07607e29198)


   `Confusion Matrix Results:`



  ![image](https://github.com/user-attachments/assets/a837c282-3b60-496a-92f2-b03f38471672)

  
- **XGBoost Classifier**: Outperforms Random Forest slightly but struggles to distinguish between "Oligodendroglioma" and "Astrocytoma."
   Feature Importance Chart of XGBoost



   ![image](https://github.com/user-attachments/assets/779e8e17-44d6-4cee-a753-02203eb4d684)




  `Confusion Matrix Results:`



  ![image](https://github.com/user-attachments/assets/66ae91af-a55d-4510-8da5-72639db25e14)



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


# Front End for Classification using Radiomics features

![image](https://github.com/user-attachments/assets/1bb62590-e7fd-42a6-adf2-109d988b0ffa)


