from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@dataclass(frozen=True)
class IrisData:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    target_names: pd.Index


@dataclass(frozen=True)
class ModelResult:
    model: Pipeline
    accuracy: float
    test_size: int


def load_iris_frame() -> IrisData:
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    target_names = pd.Index(iris.target_names)
    frame["species"] = frame["target"].map(lambda index: target_names[int(index)])
    features = frame[FEATURE_NAMES].copy()
    target = frame["species"].copy()
    return IrisData(
        frame=frame.drop(columns=["target"]),
        features=features,
        target=target,
        target_names=target_names,
    )


def train_model(random_state: int = 42) -> ModelResult:
    data = load_iris_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        data.features,
        data.target,
        test_size=0.2,
        random_state=random_state,
        stratify=data.target,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    max_iter=1200,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return ModelResult(
        model=model,
        accuracy=float(accuracy_score(y_test, predictions)),
        test_size=len(y_test),
    )


def predict_species(model: Pipeline, measurements: list[float]) -> tuple[str, dict[str, float]]:
    sample = pd.DataFrame([measurements], columns=FEATURE_NAMES)
    label = str(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    scores = {
        str(species): float(probability)
        for species, probability in zip(model.classes_, probabilities)
    }
    return label, scores
