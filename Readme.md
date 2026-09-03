# 🌧️ Rain in Australia — Logistic Regression

My first Machine Learning classification project using Logistic Regression.

## 📌 About the Project

This project was created as my first practical implementation of
**Logistic Regression**, based on what I learned through my Machine Learning
lectures.

The goal is to use weather observations from Australian weather stations to
predict whether it will rain the following day.

The target variable is:

`RainTomorrow` → `Yes` / `No`

---

## 🎯 Problem Statement

Given today's weather conditions at a particular location, predict whether
it will rain tomorrow.

This is a **binary classification problem**, making Logistic Regression a
suitable starting model.

---

## 📊 Dataset

The project uses the **Rain in Australia** dataset from Kaggle.

The dataset contains approximately 10 years of daily weather observations
from multiple Australian weather stations.

The original dataset contains:

- 145,460 rows
- 23 columns
- Numerical and categorical weather features
- 49 different locations

Some of the features include:

- MinTemp
- MaxTemp
- Rainfall
- Evaporation
- Sunshine
- WindGustDir
- WindGustSpeed
- WindDir9am
- WindDir3pm
- Humidity9am
- Humidity3pm
- Pressure9am
- Pressure3pm
- Cloud9am
- Cloud3pm
- Temp9am
- Temp3pm
- RainToday

Target:

- RainTomorrow

---

## 🧹 Data Preprocessing

Before training the model, several preprocessing steps were performed.

### 1. Handling Missing Target Values

Rows with missing values in `RainToday` and `RainTomorrow` were removed.

### 2. Separating Features and Target

The target column was:

```python
target_col = 'RainTomorrow'
