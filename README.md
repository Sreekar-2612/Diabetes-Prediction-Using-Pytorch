# Diabetes Prediction with PyTorch

This repository contains a Jupyter Notebook (`diabetes_prediction.ipynb`) that demonstrates building a neural network using PyTorch to predict diabetes based on various health indicators. The project covers data loading, preprocessing, model definition, training, evaluation, and saving the trained model.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Results](#results)

## Introduction
This project aims to predict the onset of diabetes in patients based on diagnostic measurements. The primary goal is to develop a robust classification model using PyTorch, a popular open-source machine learning library.

## Dataset
The dataset used is the Pima Indians Diabetes Database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It contains several medical predictor variables and one target variable, `Outcome`, indicating whether the patient has diabetes (1) or not (0).

**Features:**
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age (years)

**Target Variable:**
- `Outcome`: Class variable (0 or 1, where 1 means onset of diabetes within five years)

## Prerequisites
To run this notebook, you will need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

## Project Structure
- `diabetes.csv`: The dataset used for training and evaluation.
- `diabetes_prediction.ipynb`: The Jupyter Notebook containing the complete code for data processing, model training, and evaluation.
- `diabetes.pt`: The saved PyTorch model after training.

## Model Architecture
The model is a simple Artificial Neural Network (ANN) implemented using PyTorch's `nn.Module`. It consists of:
- An input layer with 8 features (corresponding to the dataset's features).
- Two hidden layers with ReLU activation functions.
- An output layer with 2 neurons (for binary classification: Diabetic/No Diabetic).

```python
class ANN_Model(nn.Module):
  def __init__(self, input_features=8, hidden1=20, hidden2=20, out_features=2):
    super().__init__()
    self.f_connected1 = nn.Linear(input_features, hidden1)
    self.f_connected2 = nn.Linear(hidden1, hidden2)
    self.out = nn.Linear(hidden2, out_features)
  def forward(self, x):
    x = f.relu(self.f_connected1(x))
    x = f.relu(self.f_connected2(x))
    x = self.out(x)
    return x
```

## Training
The model is trained using `CrossEntropyLoss` as the loss function and `Adam` optimizer. The training process runs for 500 epochs, and the loss is printed every 10 epochs.

## Evaluation
After training, the model's performance is evaluated on a test set. Key evaluation metrics include:
- **Accuracy Score**: Measures the proportion of correctly classified instances.
- **Confusion Matrix**: Provides a detailed breakdown of correct and incorrect predictions for each class.

## Prediction
The notebook also includes an example of how to use the trained model to predict the outcome for new, unseen data points.

## Results
The model achieved an accuracy score of approximately `80.52%` on the test set. The confusion matrix helps visualize the true positives, true negatives, false positives, and false negatives.

### Loss Curve
During training, the loss function decreased consistently, indicating that the model was learning effectively:



