# 🌧️ Rain in Australia — Logistic Regression

> My first hands-on Machine Learning project implementing **Logistic Regression** on a real-world weather dataset.

This project uses the **Rain in Australia** dataset to predict whether it will rain tomorrow at a particular location in Australia based on today's weather conditions.

The project was built while learning Machine Learning through lectures and practical exercises, with the goal of understanding the complete workflow of a **binary classification problem** — from data exploration and preprocessing to model training and evaluation.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Features](#-features)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Train Validation Test Split](#-train-validation-test-split)
- [Logistic Regression](#-logistic-regression)
- [Model Evaluation](#-model-evaluation)
- [Baseline Comparison](#-baseline-comparison)
- [Making Predictions](#-making-predictions)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [What I Learned](#-what-i-learned)
- [Future Improvements](#-future-improvements)
- [Limitations](#-limitations)
- [Conclusion](#-conclusion)

---

## 🔍 Project Overview

Weather prediction is a practical application of Machine Learning where historical weather observations can be used to identify patterns and make predictions about future weather conditions.

In this project, the objective is to predict:

**Will it rain tomorrow?**

The target variable is:

```text
RainTomorrow
```

The target has two possible values:

```text
Yes → It will rain tomorrow
No  → It will not rain tomorrow
```

Since the model predicts one of two possible outcomes, this is a:

**Binary Classification Problem**

The overall Machine Learning workflow used in this project is:

```text
Raw Weather Data
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Missing Value Handling
       ↓
Feature Scaling
       ↓
Categorical Encoding
       ↓
Train / Validation / Test Split
       ↓
Logistic Regression
       ↓
Predictions
       ↓
Model Evaluation
       ↓
Baseline Comparison
```

---

## 🎯 Problem Statement

The Rain in Australia dataset contains approximately 10 years of daily weather observations collected from multiple locations across Australia.

The task can be formulated as:

> Given today's weather observations for a particular location, predict whether it will rain at that location tomorrow.

For example, the model can use information such as:

- Today's temperature
- Today's rainfall
- Humidity
- Atmospheric pressure
- Wind speed
- Wind direction
- Cloud coverage
- Location
- Whether it rained today

to predict the value of:

```text
RainTomorrow
```

---

## 📊 Dataset

The project uses the **Rain in Australia** dataset from Kaggle.

The original dataset contains:

- **145,460 rows**
- **23 columns**
- Approximately 10 years of daily weather observations
- Weather observations from **49 different locations**

The dataset contains both numerical and categorical variables.

### Dataset columns

```text
Date
Location
MinTemp
MaxTemp
Rainfall
Evaporation
Sunshine
WindGustDir
WindGustSpeed
WindDir9am
WindDir3pm
WindSpeed9am
WindSpeed3pm
Humidity9am
Humidity3pm
Pressure9am
Pressure3pm
Cloud9am
Cloud3pm
Temp9am
Temp3pm
RainToday
RainTomorrow
```

After removing rows where `RainToday` or `RainTomorrow` was missing, the dataset contained:

```text
140,787 rows
23 columns
```

---

## 🧾 Features

The dataset contains both numerical and categorical features.

### Numerical Features

| Feature | Description |
|---|---|
| `MinTemp` | Minimum temperature |
| `MaxTemp` | Maximum temperature |
| `Rainfall` | Amount of rainfall |
| `Evaporation` | Evaporation measurement |
| `Sunshine` | Hours of sunshine |
| `WindGustSpeed` | Maximum wind gust speed |
| `WindSpeed9am` | Wind speed at 9 AM |
| `WindSpeed3pm` | Wind speed at 3 PM |
| `Humidity9am` | Humidity at 9 AM |
| `Humidity3pm` | Humidity at 3 PM |
| `Pressure9am` | Atmospheric pressure at 9 AM |
| `Pressure3pm` | Atmospheric pressure at 3 PM |
| `Cloud9am` | Cloud coverage at 9 AM |
| `Cloud3pm` | Cloud coverage at 3 PM |
| `Temp9am` | Temperature at 9 AM |
| `Temp3pm` | Temperature at 3 PM |

### Categorical Features

| Feature | Description |
|---|---|
| `Location` | Weather station/location |
| `WindGustDir` | Wind gust direction |
| `WindDir9am` | Wind direction at 9 AM |
| `WindDir3pm` | Wind direction at 3 PM |
| `RainToday` | Whether it rained today |

### Target Variable

```text
RainTomorrow
```

Possible values:

```text
Yes
No
```

---

# 📈 Exploratory Data Analysis

Before training the model, Exploratory Data Analysis (EDA) was performed to understand the dataset and investigate relationships between different weather variables.

The following libraries were used:

- Matplotlib
- Seaborn
- Plotly

### Visualizations performed

#### 1. Location vs Rainy Days

This visualization was used to understand how rainfall patterns vary across different Australian locations.

#### 2. Temperature at 3 PM vs Rain Tomorrow

This was used to investigate the relationship between afternoon temperature and whether it rains the following day.

#### 3. Rain Today vs Rain Tomorrow

This visualization explored the relationship between today's rainfall and tomorrow's rainfall.

#### 4. Minimum Temperature vs Maximum Temperature

A scatter plot was used to understand the relationship between daily minimum and maximum temperatures.

#### 5. Temperature vs Humidity

The relationship between:

```text
Temp3pm
```

and

```text
Humidity3pm
```

was visualized, with observations separated according to `RainTomorrow`.

These visualizations helped build an understanding of the dataset before applying Machine Learning.

---

# 🧹 Data Preprocessing

Real-world datasets often contain missing values and categorical variables.

Since Machine Learning algorithms generally require numerical input, the data needed to be processed before training the model.

The main preprocessing steps were:

1. Removing rows with missing target values
2. Separating input features and target
3. Identifying numerical and categorical features
4. Handling missing numerical values
5. Scaling numerical features
6. Encoding categorical features

---

## 1. Removing Missing Target Values

Rows containing missing values in:

```text
RainToday
RainTomorrow
```

were removed.

This reduced the dataset from:

```text
145,460 rows
```

to:

```text
140,787 rows
```

The target variable was then separated from the input features.

---

## 2. Separating Features and Target

The target column was:

```python
target_col = 'RainTomorrow'
```

The remaining relevant columns were used as input features.

Conceptually:

```text
Input Features
      ↓
Weather Conditions
      ↓
Machine Learning Model
      ↓
RainTomorrow
```

---

## 3. Identifying Numerical and Categorical Columns

The input features were separated into:

```text
Numerical Features
Categorical Features
```

Numerical columns were identified using:

```python
numeric_cols = train_inputs.select_dtypes(
    include=np.number
).columns.tolist()
```

Categorical columns were identified using:

```python
categorical_cols = train_inputs.select_dtypes(
    'object'
).columns.tolist()
```

This allowed different preprocessing techniques to be applied to each type of feature.

---

## 4. Handling Missing Numerical Values

Several numerical features contained missing values.

Examples include:

- `Evaporation`
- `Sunshine`
- `WindGustSpeed`
- `Pressure9am`
- `Pressure3pm`
- `Cloud9am`
- `Cloud3pm`
- `Humidity9am`
- `Humidity3pm`

Instead of removing all rows containing missing numerical values, **mean imputation** was used.

The following Scikit-learn class was used:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
```

The imputer calculates the mean of each numerical feature and uses that value to replace missing observations.

The same fitted imputer was then used to transform the training, validation, and test datasets.

---

## 5. Scaling Numerical Features

The numerical features had very different ranges.

For example:

```text
Temperature → around -8 to 48
Rainfall    → 0 to 371
Pressure    → around 977 to 1041
```

Using features with very different scales can make optimization less effective.

Therefore, **MinMaxScaler** was used.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

The numerical features were transformed to approximately the range:

```text
0 to 1
```

This places the numerical features on a comparable scale.

---

## 6. Encoding Categorical Features

Machine Learning models generally require numerical inputs, but several features in this dataset are categorical.

Examples:

```text
Location
WindGustDir
WindDir9am
WindDir3pm
RainToday
```

To convert these features into numerical form, **One-Hot Encoding** was used.

```python
from sklearn.preprocessing import OneHotEncoder
```

One-hot encoding creates binary columns representing different categories.

For example:

```text
RainToday

Yes → 1
No  → 0
```

Similarly, different locations and wind directions are represented using binary columns.

The encoder was also configured to handle unknown categories:

```python
handle_unknown='ignore'
```

This prevents errors when a category appears in validation/test/new data that was not present during fitting.

---

# ✂️ Train Validation Test Split

The dataset was divided into three parts:

```text
Training Set
Validation Set
Test Set
```

The project used a **time-based split** based on the year.

The split was:

```text
Training   → Year < 2015
Validation → Year = 2015
Test       → Year > 2015
```

This approach allows the model to learn from earlier weather observations and evaluate its performance on later observations.

### Final Dataset Sizes

After preprocessing and encoding:

| Dataset | Samples | Features |
|---|---:|---:|
| Training | 97,988 | 123 |
| Validation | 17,089 | 123 |
| Test | 25,710 | 123 |

The preprocessing transformed the original weather variables into **123 numerical features**.

---

# 🤖 Logistic Regression

The main Machine Learning algorithm used in this project is:

**Logistic Regression**

from Scikit-learn.

Logistic Regression is a classification algorithm commonly used for binary classification problems.

Instead of directly predicting a continuous numerical value, Logistic Regression calculates a probability.

The basic idea is:

```text
Input Features
      ↓
Weighted Sum
      ↓
Sigmoid Function
      ↓
Probability
      ↓
Classification
```

The sigmoid function converts the model's output into a value between:

```text
0 and 1
```

This value can be interpreted as the probability of a particular class.

For example:

```text
Probability of No Rain = 0.48
Probability of Rain    = 0.52
```

The model would therefore predict:

```text
RainTomorrow = Yes
```

because the probability of `Yes` is higher.

---

# 📊 Model Evaluation

The trained Logistic Regression model was evaluated on:

- Training dataset
- Validation dataset
- Test dataset

### Accuracy Results

| Dataset | Accuracy |
|---|---:|
| Training | **85.19%** |
| Validation | **85.40%** |
| Test | **84.20%** |

### Final Test Accuracy

```text
84.20%
```

The model achieved a similar performance on the training and validation datasets, while achieving approximately **84.20% accuracy on unseen test data**.

---

# 🧮 Confusion Matrix

A confusion matrix was used to understand the model's classification performance in more detail.

A confusion matrix provides information about:

- True Positives
- True Negatives
- False Positives
- False Negatives

The project used:

```python
from sklearn.metrics import confusion_matrix
```

and visualized the normalized confusion matrix using Seaborn.

The training confusion matrix showed that the model performed particularly well at identifying non-rain cases, while predicting rain was more difficult.

This is important because simply looking at accuracy can hide how well the model performs for each class.

---

# 🧪 Baseline Comparison

To determine whether the Logistic Regression model was actually useful, it was compared against simple baseline approaches.

## Random Classifier

A random prediction strategy achieved approximately:

```text
50% accuracy
```

This is expected because the model is randomly choosing between:

```text
Yes
No
```

---

## Always Predict "No"

A second baseline simply predicted:

```text
No
```

for every observation.

This achieved approximately:

```text
77.34% accuracy
```

This relatively high accuracy occurs because the dataset contains significantly more non-rain observations than rain observations.

---

## Logistic Regression

The Logistic Regression model achieved:

```text
84.20% test accuracy
```

Therefore:

```text
Random Baseline       → ~50%
Always-No Baseline    → ~77.34%
Logistic Regression   → ~84.20%
```

The Logistic Regression model therefore outperformed both simple baseline approaches.

---

# 🔮 Making Predictions

After training the model, it can be used to make predictions on new weather observations.

The project demonstrated this using a new observation from the **Katherine** weather station.

The model can make a class prediction using:

```python
model.predict(...)
```

and probability predictions using:

```python
model.predict_proba(...)
```

The class ordering can be checked using:

```python
model.classes_
```

which returned:

```text
['No', 'Yes']
```

Therefore, the probabilities returned by `predict_proba()` correspond to:

```text
Probability[0] → No
Probability[1] → Yes
```

This makes it possible to see not only the final prediction but also how confident the model is.

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Data Manipulation

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn
- Plotly

## Machine Learning

- Scikit-learn

## Development Environment

- Jupyter Notebook

---

# 📦 Main Scikit-learn Components

The following Scikit-learn components were used:

```text
train_test_split
SimpleImputer
MinMaxScaler
OneHotEncoder
LogisticRegression
accuracy_score
confusion_matrix
```

---

# 📁 Project Structure

A possible repository structure for this project is:

```text
Rain-In-Australia/
│
├── README.md
├── Rain_In_Australia.ipynb
├── requirements.txt
│
├── data/
│   └── weatherAUS.csv
│
└── images/
    ├── rainfall_distribution.png
    ├── confusion_matrix.png
    └── visualizations.png
```

If the dataset is not included in the repository, instructions for downloading it can be added to the README.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Rain-In-Australia.git
```

Replace:

```text
YOUR-USERNAME
```

with your GitHub username.

---

## 2. Navigate to the Project Directory

```bash
cd Rain-In-Australia
```

---

## 3. Install Dependencies

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn jupyter
```

Alternatively, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

---

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Rain_In_Australia.ipynb
```

and run the notebook cells sequentially.

---

# 📚 What I Learned

This project was my **first practical implementation of Logistic Regression**.

The main purpose was not just to train a model, but to understand the different stages involved in a Machine Learning project.

Through this project, I learned how to:

- Work with a real-world dataset
- Load and inspect data using Pandas
- Understand dataset structure
- Identify numerical and categorical features
- Perform Exploratory Data Analysis
- Visualize relationships between variables
- Handle missing data
- Perform mean imputation
- Scale numerical features
- Apply One-Hot Encoding
- Split data into training, validation, and test sets
- Understand binary classification
- Train a Logistic Regression model
- Generate class predictions
- Generate prediction probabilities
- Evaluate model accuracy
- Create and interpret confusion matrices
- Compare a Machine Learning model against simple baselines

Most importantly, this project helped me understand the **end-to-end Machine Learning workflow**.

---

# 📈 Future Improvements

There are several ways this project can be improved.

## 1. Try More Classification Algorithms

The next step would be to compare Logistic Regression with other models such as:

- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting
- XGBoost

This would help determine whether a more complex model can improve prediction performance.

---

## 2. Use More Evaluation Metrics

Accuracy is useful, but it does not tell the complete story.

Future versions can include:

```text
Precision
Recall
F1-Score
ROC-AUC
Precision-Recall Curve
```

These metrics are particularly useful when dealing with imbalanced classification datasets.

---

## 3. Hyperparameter Tuning

The Logistic Regression model can be optimized using techniques such as:

```text
GridSearchCV
RandomizedSearchCV
```

Possible parameters to experiment with include:

```text
C
solver
max_iter
class_weight
```

---

## 4. Feature Engineering

Additional features could be created from the existing data.

For example:

```text
Temperature Difference
Pressure Difference
Humidity Difference
Month
Season
Year
```

The `Date` feature could also be processed to extract useful temporal information.

---

## 5. Build a Machine Learning Pipeline

The current preprocessing steps can be combined into a single Scikit-learn pipeline using:

```text
Pipeline
ColumnTransformer
```

This would make the project:

- Cleaner
- More reproducible
- Easier to maintain
- Easier to deploy

---

## 6. Model Deployment

A future version of this project could be deployed as a simple web application.

Possible technologies include:

- Streamlit
- Flask
- FastAPI

A user could enter today's weather conditions and receive:

```text
Rain Tomorrow: Yes

Probability:
Yes → 72%
No  → 28%
```

---

# ⚠️ Limitations

This project is primarily an educational implementation of Logistic Regression.

Some limitations include:

- Only Logistic Regression was used as the main model.
- Extensive hyperparameter tuning was not performed.
- Accuracy alone does not fully describe model performance.
- The dataset contains imbalanced target classes.
- Feature engineering could be improved.
- More advanced Machine Learning algorithms could be compared.
- The model has not been deployed as a production application.

---

# 🎓 Project Context

This project represents my **first hands-on Machine Learning classification project**.

I built this project while learning Machine Learning concepts through lectures and practical implementation.

The goal was to take the concepts learned in theory and apply them to a real-world dataset.

The project helped me understand that building a Machine Learning model involves much more than simply calling:

```python
model.fit(X, y)
```

A complete workflow involves:

```text
Understanding the Problem
        ↓
Understanding the Data
        ↓
Cleaning the Data
        ↓
Exploring the Data
        ↓
Preparing Features
        ↓
Training the Model
        ↓
Evaluating the Model
        ↓
Comparing Baselines
        ↓
Making Predictions
```

This project is my first step toward building more advanced Machine Learning projects.

---

# 📌 Key Results

```text
Dataset
-----------------------------
Original Rows       : 145,460
Cleaned Rows        : 140,787
Locations           : 49

Preprocessed Data
-----------------------------
Training Samples    : 97,988
Validation Samples  : 17,089
Test Samples        : 25,710
Final Features      : 123

Model Performance
-----------------------------
Training Accuracy   : 85.19%
Validation Accuracy : 85.40%
Test Accuracy       : 84.20%

Baselines
-----------------------------
Random Baseline     : ~50%
Always-No Baseline  : ~77.34%
```

---

# 🏆 Conclusion

This project demonstrates the application of **Logistic Regression to a real-world weather classification problem**.

The final model achieved approximately:

```text
84.20% Test Accuracy
```

and performed better than both:

```text
Random Prediction
```

and:

```text
Always Predict "No"
```

The most important outcome of this project was gaining practical experience with the complete Machine Learning workflow.

Starting from raw weather data, I performed:

```text
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Missing Value Imputation
      ↓
Feature Scaling
      ↓
Categorical Encoding
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Prediction
```

This project serves as my **first practical Logistic Regression project** and forms a foundation for future Machine Learning work involving more advanced models, feature engineering, hyperparameter tuning, and deployment.

---

# 👨‍💻 Author

Archit Dubey

First hands-on Machine Learning project using Logistic Regression.

---

## ⭐ If you found this project useful

Feel free to ⭐ star the repository and check out my other Machine Learning projects as I continue learning and building.

````
