from iris_model import FEATURE_NAMES, load_iris_frame, predict_species, train_model


def test_load_iris_frame_has_expected_shape_and_columns():
    data = load_iris_frame()

    assert data.frame.shape[0] == 150
    assert list(data.features.columns) == FEATURE_NAMES
    assert "species" in data.frame.columns


def test_train_model_returns_accuracy_and_classes():
    data = load_iris_frame()
    result = train_model(random_state=42)

    assert result.accuracy >= 0.85
    assert list(result.model.classes_) == data.target_names.tolist()


def test_predict_species_returns_label_and_probabilities():
    result = train_model(random_state=42)
    label, probabilities = predict_species(
        result.model,
        [5.1, 3.5, 1.4, 0.2],
    )

    assert label in {"setosa", "versicolor", "virginica"}
    assert set(probabilities) == {"setosa", "versicolor", "virginica"}
    assert abs(sum(probabilities.values()) - 1.0) < 0.001
