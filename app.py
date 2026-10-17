import os
import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")  # hide sklearn version-mismatch warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "delivery_delay.csv")
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_model.joblib")
TARGET = "Delivery_Delay"

st.set_page_config(page_title="Delivery Delay Predictor", page_icon="🚚")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(df):
    """Load the saved model. If it can't be loaded (e.g. sklearn version
    mismatch), retrain the same Decision Tree once - it takes < 1 second."""
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    try:
        model = joblib.load(MODEL_PATH)
        model.predict(X_test.head(1))  # quick check that it works
        source = "saved model file"
    except Exception:
        model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        source = "retrained in app (saved file was incompatible)"
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, source


df = load_data()
model, test_acc, model_source = load_model(df)
features = [c for c in df.columns if c != TARGET]

st.title("🚚 Delivery Delay Predictor")
st.caption(
    f"Decision Tree model · test accuracy {test_acc:.1%} · loaded from {model_source}"
)

st.subheader("Enter delivery details")

# Build an input widget for every feature, using the dataset's ranges
inputs = {}
col1, col2 = st.columns(2)
for i, feat in enumerate(features):
    col = col1 if i % 2 == 0 else col2
    s = df[feat]
    label = feat.replace("_", " ")
    if pd.api.types.is_integer_dtype(s):
        inputs[feat] = col.slider(
            label, int(s.min()), int(s.max()), int(s.median())
        )
    else:
        inputs[feat] = col.number_input(
            label,
            min_value=float(round(s.min(), 2)),
            max_value=float(round(s.max(), 2)),
            value=float(round(s.median(), 2)),
            step=0.5,
        )

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([inputs])[features]
    pred = int(model.predict(input_df)[0])
    if pred == 1:
        st.error("⚠️ This delivery is likely to be **DELAYED**.")
    else:
        st.success("✅ This delivery is likely to be **ON TIME**.")

with st.expander("Which factors matter most?"):
    importance = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False)
    )
    st.bar_chart(importance)
