# Loan Prediction Project

This project is a Loan Eligibility Prediction system using Python and Machine Learning. The goal is to predict whether a loan will be approved based on applicant information.

 Dataset
File: `loan_prediction_dataset.csv`
Entries: 614
Features: 13
Target: `Loan_Status`
Dataset includes features like: `Gender`, `Married`, `Education`, `ApplicantIncome`, `Credit_History`, etc.

 Project Steps

Step 1: Import Required Libraries
Libraries used for data handling, visualization, and machine learning:
pandas, numpy – Data manipulation and numerical operations
matplotlib, seaborn – Data visualization
scikit-learn – Machine Learning models and utilities



# Step 2: Load Dataset
Load the CSV dataset into a pandas DataFrame.

df = pd.read_csv("loan_prediction_dataset.csv")

# Step 3: Dataset Overview

Check first few rows, shape, and info:

df.head()
df.shape
df.info()
df.describe()

# Step 4: Missing Values Handling

Missing values are handled using mode for categorical and median for numerical columns:

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
similarly for other columns
 
# Step 5: Encode Categorical Features

Convert categorical features to numerical using LabelEncoder:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Step 6: Data Visualization

Loan Status Distribution:

sns.countplot(x='Loan_Status', data=df)

# Credit History vs Loan Status:

sns.countplot(x='Credit_History', hue='Loan_Status', data=df)

# Applicant Income Distribution:

plt.hist(df['ApplicantIncome'], bins=30)

# Step 7: Feature Selection

Selected important features for the model:

Credit_History, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Property_Area, Education, Married
# Step 8: Stratified Train-Test Split

Use StratifiedShuffleSplit to maintain class proportions.

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Step 9: Machine Learning Models
Decision Tree

Simple and interpretable

Accuracy measured on test set
# Code
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=8, min_samples_split=6, random_state=42)
dt.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt.predict(X_test))

Random Forest

Ensemble method with higher accuracy and less overfitting

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, min_samples_leaf=2, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

# Step 10: Model Evaluation

Confusion matrix and classification report

Compare Decision Tree vs Random Forest

Model	Accuracy (%)	Advantage
Decision Tree	85.0	Simple and easy to interpret
Random Forest	89.2	Higher accuracy, less overfitting

# Step 11: Feature Importance

Random Forest feature importance visualization:

plt.barh(features, rf.feature_importances_)

# Step 12: Correlation Heatmap

Check feature correlation using seaborn heatmap:
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

# Requirements
pandas==2.1.1
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.12.3
scikit-learn==1.3.2

# Summary
Dataset cleaning, missing value handling, and encoding performed.
Random Forest outperforms Decision Tree for this dataset.
Visualizations help understand feature importance and distributions.

# Usage

Clone the repo

Place loan_prediction_dataset.csv in the project folder

# Install requirements:

pip install -r requirements.txt

Run the notebook to train models and visualize results.