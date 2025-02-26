# Autism Prediction Using Machine Learning

This project implements a machine learning-based system to predict autism likelihood using a pre-trained model. It includes a training pipeline to preprocess data and train a model, as well as a Streamlit web application for user-friendly predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Autism Prediction system uses a dataset (`train.csv`) to train a machine learning model that predicts autism based on various features. The trained model, along with preprocessing artifacts (encoders and scaler), is saved as `.pkl` files. A Streamlit app (`app.py`) allows users to input data and receive predictions interactively.

## Features
- **Data Preprocessing**: Handles categorical and numerical features using `LabelEncoder` and `StandardScaler`.
- **Model**: Utilizes a Random Forest Classifier (customizable to other algorithms).
- **Web Interface**: Streamlit app for easy prediction input and output.
- **Modularity**: Saved artifacts enable reuse without retraining.

## Requirements
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `streamlit`
  - `pickle` (built-in)

  Dataset
File: train.csv
Description: Contains features (e.g., age, gender, behavioral scores) and a target column (e.g., diagnosis indicating autism presence).
Source: [Specify your dataset source, e.g., "Collected from XYZ study" or "Public dataset from Kaggle"].
Note: Ensure the dataset is in the project directory for training and app functionality.
Training the Model
Ensure train.csv is in the project directory.
Run the training script (example provided separately or embedded in a Jupyter notebook):
python
# See training script for details
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
# ... (preprocessing and training code)
Outputs:
autism_model.pkl: Trained model.
scaler.pkl: Numerical feature scaler.
<column>_label_encoder.pkl: Encoders for each categorical column.
target_encoder.pkl (optional): Encoder for the target variable.
Running the Streamlit App
Ensure all .pkl files and train.csv are in the same directory as app.py.
Launch the app:
bash
streamlit run app.py
Open your browser at http://localhost:8501 to access the interface.
Usage
Training: Run the training script to generate model and preprocessing files (if not already done).
Prediction:
Open the Streamlit app.
Enter values for all features in the form (matches train.csv columns).
Click "Predict" to see the autism likelihood result.
File Structure
autism-prediction/
│
├── train.csv              # Training dataset
├── app.py                 # Streamlit app for predictions
├── autism_model.pkl       # Trained ML model
├── scaler.pkl             # StandardScaler for numerical features
├── <column>_label_encoder.pkl  # LabelEncoders for categorical features
├── target_encoder.pkl     # (Optional) Target variable encoder
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
Contributing
Contributions are welcome! Please:
Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

### Customization Notes
- **Repository URL**: Replace `https://github.com/yourusername/autism-prediction.git` with your actual repo URL or remove the clone step if not hosted.
- **Dataset Source**: Update the "Source" field under [Dataset](#dataset) with where you got `train.csv`.
- **Model Details**: If you used a different model (e.g., SVM instead of Random Forest), update the [Features](#features) section.
- **License**: Add a `LICENSE` file if you want to specify one (e.g., MIT), or remove the license section if not applicable.

### Adding `requirements.txt`
Create a `requirements.txt` file in your project directory with:
pandas
numpy
scikit-learn
streamlit
This ensures others can install dependencies easily.

### Next Steps
1. Save this as `README.md` in your project folder.
2. Test the instructions (e.g., installation, running the app) to ensure they work.
3. Share your dataset’s column names or specific requirements if you need further tailoring!

Let me know if you want adjustments!
