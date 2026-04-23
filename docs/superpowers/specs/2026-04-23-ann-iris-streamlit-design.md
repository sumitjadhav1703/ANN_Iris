# ANN Iris Streamlit Design

## Goal

Build a Streamlit-hosted portfolio app for the Iris ANN notebook. The first screen should be a learning dashboard: dataset overview, interactive prediction controls, confidence output, and a short model explanation.

## Architecture

The project will keep model logic in `iris_model.py` and UI code in `app.py`. The app will use the built-in scikit-learn Iris dataset so Streamlit Cloud can run without a separate `Iris.csv` upload. A lightweight `MLPClassifier` approximates the ANN workflow from the notebook while staying practical for hosted deployment.

## User Experience

The app opens directly to the working dashboard. Users can adjust sepal and petal measurements, click predict, and see the predicted species with probability bars. Supporting sections show species distribution, a scatter plot, model accuracy, and the original notebook reference.

## Verification

Tests cover dataset loading, model training, and prediction output shape. A syntax compile check and pytest run verify the project before pushing.
