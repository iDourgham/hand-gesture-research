# Research: Hand Gesture Classification

This repository contains the implementation, training, and evaluation of machine learning models for **Hand Gesture Classification** using a normalized dataset derived from the HaGRID dataset. The goal is to classify different hand gestures accurately using various classification algorithms.

---

## Project Overview

Hand gesture recognition is an important area in computer vision and human-computer interaction. This project focuses on classifying hand gestures from normalized landmark data using classical and ensemble machine learning models.

The project includes:

- Data preprocessing and label encoding
- Model training using Random Forest, Support Vector Classifier (SVC), and XGBoost classifiers
- Evaluation using accuracy, precision, recall, and F1-score metrics
- Confusion matrix visualization
- Model logging and experiment tracking with MLflow

---

## Dataset

The dataset is a CSV file containing normalized hand gesture features and a target label column. The label column is encoded into numerical classes for training.

---

# Hand Gesture Classification Models Comparison

This repository contains implementations and experiments for classifying hand gestures using various machine learning models. Below is a comparison of the performance metrics of three different models trained and evaluated on the same dataset.

| Model         | Accuracy        | F1 Score       | Precision      | Recall         |
|---------------|-----------------|----------------|----------------|----------------|
| Random Forest | 0.9788          | 0.9785         | 0.9784         | 0.9786         |
| SVC           | 0.9611          | 0.9609         | 0.9616         | 0.9612         |
| XGBoost       | **0.9848**      | **0.9845**     | **0.9844**     | **0.9847**     |

---

## Summary

- **XGBoost** achieved the highest performance across all metrics, making it the best choice among the tested models.
- **Random Forest** performed well, slightly behind XGBoost.
- **SVC** also showed competitive results but lagged behind the ensemble methods.

---

## Notes

- All models were trained on the same preprocessed dataset of normalized hand gesture features.
- Metrics were computed on the test set.
- Performance metrics used:
  - **Accuracy:** Overall correctness of the model.
  - **F1 Score:** Harmonic mean of precision and recall.
  - **Precision:** Correct positive predictions over all positive predictions.
  - **Recall:** Correct positive predictions over all actual positives.

---

Feel free to explore the code and experiment with hyperparameters to improve these results!

---

## Installation

To run the code and experiments, ensure you have the following Python packages installed:

## Setup Instructions

### 3. Create the environment

You can set up the required environment either with Conda or pip.

#### Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate mediapipe_env
```

### Requirements

```bash
mlflow==2.22.0
scipy==1.15.2
psutil==5.9.0
numpy==1.26.4
pandas==2.2.3
seaborn==0.13.2
mediapipe==0.10.21
cv2==4.11.0
xgboost==2.1.1
scikit-learn==1.6.1
```

---

## Contributors
- **Mohamed Mohy** ([GitHub Profile](https://github.com/iDourgham))  
                   ([LinkedIn Profile](https://www.linkedin.com/in/eng-m-mohy/))


