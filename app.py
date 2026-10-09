import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --------------------------------
# 1. Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")
st.write("Predict the Iris species using a trained Stacking Classifier.")

# --------------------------------
# 2. Load Trained Model
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "stacking_clf.pkl")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)

    if not hasattr(model, "predict"):
        st.error("The loaded object is not a valid prediction model.")
        st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --------------------------------
# 3. Class Labels
# --------------------------------
CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# --------------------------------
# 4. Input Features
# --------------------------------
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider(
        "Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1
    )
    sepal_width = st.slider(
        "Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1
    )

with col2:
    petal_length = st.slider(
        "Petal Length (cm)", 1.0, 7.0, 4.3, 0.1
    )
    petal_width = st.slider(
        "Petal Width (cm)", 0.1, 2.5, 1.3, 0.1
    )

# --------------------------------
# 5. Prediction
# --------------------------------
if st.button("Predict Species", type="primary", use_container_width=True):

    # Feature names must match the training data.
    features = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })

    try:
        # Verify the loaded model
        st.write("Model:", type(model).__name__)

        # Predict species
        prediction = model.predict(features)
        predicted_class = prediction[0]

        # Convert NumPy scalar to Python scalar if necessary
        if isinstance(predicted_class, np.generic):
            predicted_class = predicted_class.item()

        # Convert numeric labels to readable names
        if isinstance(predicted_class, (int, float)) and predicted_class in CLASS_NAMES:
            species = CLASS_NAMES[int(predicted_class)]
        else:
            species = str(predicted_class)

        st.success(f"🌸 Predicted Species: **{species}**")

        # --------------------------------
        # 6. Prediction Probabilities
        # --------------------------------
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_

            st.subheader("Prediction Probabilities")

            for class_label, probability in zip(classes, probabilities):
                if isinstance(class_label, np.generic):
                    class_label = class_label.item()

                if isinstance(class_label, (int, float)) and class_label in CLASS_NAMES:
                    class_name = CLASS_NAMES[int(class_label)]
                else:
                    class_name = str(class_label)

                st.write(f"**{class_name}:** {probability * 100:.2f}%")
                st.progress(float(probability))

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# --------------------------------
# 7. Footer
# --------------------------------
st.divider()
st.caption("Iris Flower Classification • Powered by Scikit-learn")
