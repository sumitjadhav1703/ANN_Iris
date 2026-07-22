# ANN Iris Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

A compact and interactive Streamlit dashboard for predicting Iris flower species using an Artificial Neural Network (ANN). This project serves as an educational example of how to train a machine learning model using `scikit-learn` and deploy it seamlessly using a `Streamlit` web interface. The application uses the built-in Iris dataset, eliminating the need for external CSV files, and trains a small neural network pipeline directly at startup.

## Application Preview

> **Note:** Screenshot placeholder. Add a screenshot of the running Streamlit app here.
>
> `![Application Preview](docs/screenshot.png)`

---

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Streamlit Interface](#streamlit-interface)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Author](#author)
- [License](#license)

---

## Features

- **Interactive Predictions:** Adjust sepal and petal measurements via a sidebar slider to see real-time species predictions.
- **Neural Network Classifier:** Uses a Multi-Layer Perceptron (`MLPClassifier`) to classify species.
- **Confidence Scoring:** Visualizes prediction probabilities for Setosa, Versicolor, and Virginica using a bar chart.
- **Data Visualization:** Provides a dataset preview, a species distribution chart, and an interactive scatter plot (Petal Length vs Petal Width) that highlights the user's input.
- **Self-Contained:** Built with the internal scikit-learn Iris dataset, meaning it works out of the box with no external data files required.
- **Original Notebook Included:** The original training workflow is included as `ANN(1).ipynb` for reference.

## How It Works

This project is built for students and beginners looking to understand the integration between a machine learning model and a web frontend.

### Machine Learning Pipeline

The machine learning logic is encapsulated in `iris_model.py`. The pipeline consists of the following steps:

- **Data Loading & Feature Extraction:** The dataset is loaded using `scikit-learn`'s built-in `load_iris(as_frame=True)`. The model relies on four specific features:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Data Preprocessing:** Before feeding data into the neural network, the features are scaled using `StandardScaler()`. This step ensures that all measurements have a mean of 0 and a variance of 1, which is crucial for the neural network to converge properly.
- **Model Architecture:** The model uses an `MLPClassifier` (Multi-Layer Perceptron), which is a type of Artificial Neural Network.
  - **Hidden Layers:** It contains two hidden layers with 16 neurons in the first layer and 8 in the second `(16, 8)`.
  - **Activation Function:** Uses the Rectified Linear Unit (`relu`) activation function.
  - **Solver:** Uses the `adam` optimizer for weight optimization.
  - **Iterations:** The model is allowed a maximum of 1200 iterations (`max_iter=1200`) to learn the patterns in the data.
- **Training Process:** The dataset is split into training (80%) and testing (20%) sets using `train_test_split`. The split is stratified (`stratify=data.target`), ensuring that both the training and testing sets have a proportional representation of all three Iris species. A fixed `random_state` of 42 is used for reproducibility.
- **Prediction Pipeline:** The trained `Pipeline` (scaler + classifier) exposes a `predict` method to determine the specific class (species) and a `predict_proba` method to calculate the confidence score (probability) for each possible species.

### Streamlit Interface

The frontend (`app.py`) leverages Streamlit to create an interactive web app:
- **Caching:** The application uses `@st.cache_data` to load the dataset and `@st.cache_resource` to train the model only once upon startup. This ensures the app remains fast and responsive.
- **User Inputs:** The sidebar provides sliders bounded by the minimum and maximum expected values for each feature.
- **Visuals:** The app uses Streamlit's native charting (like `st.bar_chart`) and `matplotlib` to render visualizations, such as the scatter plot comparing petal measurements and highlighting the user's specific input in real-time.

## Installation & Requirements

Ensure you have Python 3.8 or newer installed on your system.

1. **Clone the repository (or download the source):**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *The primary dependencies are: `streamlit`, `scikit-learn`, `pandas`, `matplotlib`, and `pytest`.*

## Usage

To start the Streamlit dashboard locally, run the following command in your terminal:

```bash
streamlit run app.py
```
This will start a local web server, and your default web browser will automatically open the dashboard (usually at `http://localhost:8501`).

## Running Tests

This repository includes a basic test suite to verify the model functionality. To run the tests, execute:

```bash
python3 -m pytest
```

## Project Structure

```text
.
├── app.py                 # Main Streamlit application and UI layout
├── iris_model.py          # Machine learning pipeline, model training, and data loading
├── ANN(1).ipynb           # Original Jupyter Notebook with the initial model exploration
├── requirements.txt       # Python dependencies
├── tests/
│   └── test_iris_model.py # Unit tests for the ML pipeline
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

## Deployment

To deploy this app on **Streamlit Community Cloud**:
1. Push this repository to your GitHub account.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app, select your repository, and set the main file path to `app.py`.
4. Click Deploy.

## Author

**Sumit Jadhav**

## License

This project is licensed under the [MIT License](LICENSE).
