# End-to-End ML Pipeline for Customer Churn Prediction

## Objective
The objective of this project is to build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

## Dataset

- Dataset: Telco Customer Churn
- Rows: 7043
- Columns: 21

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook
- Matplotlib


## Workflow

### 1. Data Loading

The dataset was loaded using Pandas.

### 2. Data Cleaning

- Removed customerID
- Converted TotalCharges into numeric
- Filled missing values using Median

### 3. Data Preprocessing

- Numerical Features
  - Median Imputation
  - Standard Scaling

- Categorical Features
  - Most Frequent Imputation
  - One Hot Encoding

### 4. Models Used

- Logistic Regression
- Random Forest Classifier

### 5. Hyperparameter Tuning

- GridSearchCV

### 6. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score


## Results

### Logistic Regression

Accuracy: **80.55%**

### Random Forest

Accuracy: **77.79%**

The Logistic Regression model performed better than the Random Forest model on this dataset.

## Project Structure

Task2 End to End ML Pipeline
│
├── data
├── images
├── models
├── notebooks
├── best_churn_model.pkl
├── README.md
└── Task2 End to End ML Pipeline.ipynb

## Output

The trained model was saved using Joblib.

best_churn_model.pkl
## Author
Developed as part of the DevelopersHub AI/ML Engineering Internship.