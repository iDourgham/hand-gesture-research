import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.preprocessing import LabelEncoder
#from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from joblib import dump


def load_and_preprocess_data(path="D:\\Study\\MLOps Project\\hand-gesture-research\\data\\normalized_hagrid.csv"):
    """
    Load the preprocessed normalized dataset and perform label encoding and splitting.

    Args:
        path (str): Path to the CSV file containing normalized data.

    Returns:
        X_train, X_test, y_train, y_test: Splitted and ready-to-use datasets.
    """
    # Load normalized dataset
    df = pd.read_csv(path)

    # Encode the label column
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Split features and target
    X = df.drop('label', axis=1)
    y = df['label']

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoder



def train(X_train, y_train):
    """
    Train a RandomForestClassifier and log it to MLflow.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels

    Returns:
        RandomForestClassifier: Trained RandomForestClassifier model
    """
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log model with input-output signature
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="RandomForestGestureClassifier",
        input_example=X_train.iloc[:1],
        signature=signature,
    )

    # Optional: log training data file if needed
    # mlflow.log_artifact("data/normalized_hagrid.csv")

    # Save and log local model file
    os.makedirs("model_with_mlflow", exist_ok=True)
    dump(model, "model_with_mlflow/modelRF1.pkl")
    mlflow.log_artifact("model_with_mlflow/modelRF1.pkl")

    return model
    

def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    mlflow.set_experiment("hand-gesture-classification")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run():

        # Load and preprocess the data
        X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()

        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Train and log model
        model = train(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro", zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro", zero_division=0))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="macro", zero_division=0))

        # Add a tag to the run
        mlflow.set_tag("task", "Hand Gesture Classification")

        #Plot confusion matrix
        display_labels = label_encoder.inverse_transform(model.classes_)
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=display_labels)
        conf_mat_disp.plot()

        # Save and log confusion matrix plot
        plt.title("Confusion Matrix - Hand Gesture Classification")
        os.makedirs("plots", exist_ok=True)
        cm_path = "plots/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        

if __name__ == "__main__":
    main()