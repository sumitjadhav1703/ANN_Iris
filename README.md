# ANN Iris Classifier

Streamlit dashboard for an Artificial Neural Network style Iris flower classifier.

The app lets users adjust sepal and petal measurements, predicts the Iris species, and shows model confidence, dataset preview, species distribution, and a simple scatter plot.

## Features

- Interactive Iris species prediction
- Neural network classifier using `MLPClassifier`
- Built-in Iris dataset, so no external CSV is needed for hosting
- Confidence score chart for Setosa, Versicolor, and Virginica
- Dataset overview and visualization
- Original notebook included as `ANN(1).ipynb`

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```bash
python3 -m pytest
```

## Deploy On Streamlit Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app from this repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

## Project Structure

```text
.
├── app.py
├── iris_model.py
├── ANN(1).ipynb
├── requirements.txt
├── tests/
│   └── test_iris_model.py
└── README.md
```
