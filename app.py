import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from iris_model import FEATURE_NAMES, load_iris_frame, predict_species, train_model


st.set_page_config(
    page_title="ANN Iris Classifier",
    page_icon=":material/analytics:",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_iris_frame()


@st.cache_resource
def get_model():
    return train_model(random_state=42)


data = get_data()
model_result = get_model()

st.title("ANN Iris Classifier")
st.caption("A compact neural-network dashboard for predicting Iris flower species.")

st.sidebar.header("Flower Measurements")
inputs = []
defaults = [5.1, 3.5, 1.4, 0.2]
limits = [(4.0, 8.0), (2.0, 4.5), (1.0, 7.0), (0.1, 2.6)]

for feature, default, bounds in zip(FEATURE_NAMES, defaults, limits):
    value = st.sidebar.slider(
        feature.title(),
        min_value=bounds[0],
        max_value=bounds[1],
        value=default,
        step=0.1,
    )
    inputs.append(value)

prediction, probabilities = predict_species(model_result.model, inputs)
probability_frame = (
    pd.DataFrame(
        {
            "Species": list(probabilities.keys()),
            "Confidence": list(probabilities.values()),
        }
    )
    .sort_values("Confidence", ascending=False)
    .reset_index(drop=True)
)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Predicted Species", prediction.title())
metric_b.metric("Model Accuracy", f"{model_result.accuracy:.1%}")
metric_c.metric("Test Samples", model_result.test_size)

left, right = st.columns([1, 1])

with left:
    st.subheader("Prediction Confidence")
    st.bar_chart(probability_frame, x="Species", y="Confidence", height=280)
    st.dataframe(
        probability_frame.assign(
            Confidence=lambda frame: frame["Confidence"].map(lambda value: f"{value:.2%}")
        ),
        hide_index=True,
        use_container_width=True,
    )

with right:
    st.subheader("Measurement Profile")
    input_frame = pd.DataFrame(
        {"Feature": FEATURE_NAMES, "Value": inputs}
    )
    st.bar_chart(input_frame, x="Feature", y="Value", height=280)
    st.info(
        "The model uses four flower measurements and predicts one of three species: "
        "setosa, versicolor, or virginica."
    )

st.divider()

overview_left, overview_right = st.columns([1.1, 1])

with overview_left:
    st.subheader("Dataset Preview")
    st.dataframe(data.frame.head(12), hide_index=True, use_container_width=True)

with overview_right:
    st.subheader("Species Distribution")
    species_counts = data.frame["species"].value_counts().rename_axis("Species").reset_index(name="Count")
    st.bar_chart(species_counts, x="Species", y="Count", height=260)

st.subheader("Petal Length vs Petal Width")
fig, ax = plt.subplots(figsize=(8, 4.5))
for species, group in data.frame.groupby("species"):
    ax.scatter(
        group["petal length (cm)"],
        group["petal width (cm)"],
        label=species.title(),
        s=45,
        alpha=0.8,
    )
ax.scatter(
    inputs[2],
    inputs[3],
    color="#111827",
    marker="X",
    s=130,
    label="Your Input",
)
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.legend()
ax.grid(alpha=0.2)
st.pyplot(fig)

with st.expander("About this project"):
    st.write(
        "This app is based on the ANN Iris notebook in this repository. "
        "For reliable Streamlit hosting, the app loads the built-in Iris dataset "
        "from scikit-learn and trains a small neural network pipeline at startup."
    )
