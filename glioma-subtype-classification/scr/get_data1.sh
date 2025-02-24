#!/bin/bash

date
echo "Running the job now on $(hostname)"

# Ensure Conda is initialized
eval "$(conda shell.bash hook)"

# Define environment name
ENV_NAME="gdc-client"

# Check if the Conda environment exists
if conda env list | grep -q $ENV_NAME; then
    echo "Activating existing conda environment: $ENV_NAME"
    conda activate $ENV_NAME
else
    echo "Creating new conda environment: $ENV_NAME"
    conda create -n $ENV_NAME -c bioconda gdc-client -y
    conda activate $ENV_NAME
fi

# Set the target download directory
BASE_DIR="$HOME/Documents/Projects/fyp/Glioma-ML-Classifier-with-ANOVA-Feature-Selection"
cd "$BASE_DIR"

# Ensure target directory exists
TARGET_DIR="${BASE_DIR}/data"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Current working directory: $(pwd)"

# Path to manifest file
manifest_file="${BASE_DIR}/gdc_manifest.txt"

# Check if the manifest file exists
if [[ ! -f "$manifest_file" ]]; then
    echo "Error: Manifest file not found at $manifest_file"
    exit 1
fi

# Check if gdc-client is installed
if ! command -v gdc-client &> /dev/null; then
    echo "gdc-client not found, installing..."
    conda install -c bioconda gdc-client -y
fi

# Download data from GDC using gdc-client
gdc-client download -m "$manifest_file" -d "$TARGET_DIR"

# Deactivate Conda environment
conda deactivate

echo "Job finished"
echo "-------done-------"
