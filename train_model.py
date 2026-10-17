"""Run this on your PC (NOT on Streamlit Cloud) to compare models and
save decision_tree_model.joblib. Then upload the .joblib file to GitHub."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


def main():
    print("--- Loading Data ---")
    df = pd.read_csv("delivery_delay.csv")
    print(f"Dataset shape: {df.shape}\n")

    X = df.drop("Delivery_Delay", axis=1)
    y = df["Delivery_Delay"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}\n")

    print("--- 1. Logistic Regression ---")
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}\n")

    print("--- 2. Linear SVC ---")
    svc_l = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000))
    svc_l.fit(X_train, y_train)
    print(f"LinearSVC Accuracy: {svc_l.score(X_test, y_test):.4f}\n")

    print("--- 3. SVC Hyperparameter Tuning (cross-validation on TRAIN data) ---")
    # Smaller grid + scaling: finishes in seconds instead of hanging.
    pipe = make_pipeline(StandardScaler(), SVC())
    param_grid = {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.1],
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best SVC params:", grid.best_params_)
    print(f"Best CV score:   {grid.best_score_:.4f}")
    print(f"Test accuracy:   {grid.score(X_test, y_test):.4f}\n")

    print("--- 4. Decision Tree ---")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
    print(classification_report(y_test, y_pred_dt))

    joblib.dump(dt, "decision_tree_model.joblib")
    print("Saved decision_tree_model.joblib")


if __name__ == "__main__":
    main()
