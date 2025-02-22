import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
loaded_xgb_model = joblib.load('./models/xgb_model.pkl')

st.title("Glioma Classification Model with XGBoost")
st.write("This app predicts cancer types based on gene expression data.")

# File Upload Section
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file, index_col=0)  # First column as index (Feature Names)

    # Extract sample name (header of the second column)
    sample_name = df.columns[0]

    # Extract feature values (transpose to get a single row)
    features = df[sample_name].values.reshape(1, -1)

    # Ensure correct feature count
    num_features_expected = loaded_xgb_model.n_features_in_
    num_features_actual = features.shape[1]

    if num_features_actual != num_features_expected:
        st.error(f"Error: Model expects {num_features_expected} features but got {num_features_actual}. Please check your file format.")
    else:
        # Make predictions
        predictions = loaded_xgb_model.predict(features)
        prediction_proba = loaded_xgb_model.predict_proba(features)

        # Map predictions to cancer types
        cancer_type_dict = {0: "Astrocytoma", 1: "Glioblastoma", 2: "Oligodendroglioma"}
        predicted_label = cancer_type_dict[int(predictions[0])]
        prediction_probability = prediction_proba.max()

        # Create a new DataFrame for predictions
        results_df = pd.DataFrame({"Sample_Name": [sample_name], 
                                   "Predicted Cancer Type": [predicted_label], 
                                   "Prediction Probability": [prediction_probability]})

        # Display predictions
        st.write("Predictions:")
        st.write(results_df)

        # Provide download button for results
        csv_output = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv_output, "predictions.csv", "text/csv")

else:
    st.warning("Please upload a CSV file.")
