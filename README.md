# MLOps Task 2: Heart Disease Prediction with DVC

This project uses Data Version Control (DVC) to manage machine learning pipelines and track experiments. It is for my MLOps lab assignment.

## Task Overview

The goal is to build a machine learning pipeline to predict heart disease using the `cleaned_merged_heart_dataset.csv` file. DVC helps us track the large dataset, run the workflow, and compare different models.

## Pipeline Steps

The pipeline has three steps defined in `dvc.yaml`:

1. **`preprocess`**: Loads the raw data, splits it into training (80%) and testing (20%) sets, and saves `train.csv` and `test.csv`.
2. **`train`**: Loads the training data, trains a model, and saves the model as `model.joblib`.
3. **`validate`**: Loads the test data and the model, makes predictions, and saves the results (`metrics.json` and `confusion_matrix.png`).

## Experiments

Two different models on two Git branches were created to see which is better:

- **`main` branch**: Logistic Regression model.
- **`random-forest` branch**: Random Forest Classifier.

## How to Run

1. Clone this repository and switch to the branch you want to see.
2. Activate your virtual environment and install the required tools:

```bash
pip install dvc scikit-learn pandas joblib matplotlib seaborn
````

3. Get the data from the local DVC remote:

```bash
dvc pull
```

4. Run the pipeline:

```bash
dvc repro
```

## Viewing Results

You can compare the models across branches without training them again.

To compare the accuracy numbers:

```bash
dvc metrics show --all-branches
```

To compare the confusion matrix pictures:

```bash
dvc plots diff main random-forest
```
