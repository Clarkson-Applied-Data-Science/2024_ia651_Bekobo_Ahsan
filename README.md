# 2024_ia651_Bekobo_Ahsan
Code for 2024_ia651_Bekobo_Ahsan
#  Predictive Modeling of Loan Approval Status: A Machine Learning Approach
![Screenshot 2024-07-22 183831](https://github.com/user-attachments/assets/a2a8efaf-be86-4637-b4cd-941d909564b1)

# Project overview

The Loan Approval Prediction project aims to analyze a dataset containing information about loan applicants and their respective loan approval outcomes. The dataset provides valuable insights into various factors that influence the decision-making process of loan approvals. By leveraging data exploration, statistical analysis, and predictive modeling techniques, this project seeks to understand the patterns and relationships within the data to develop a predictive model for loan approval status.
## Dataset
The dataset used in this project is sourced from Kaggle, "LoanApprovalPrediction.csv". It contains information about loan applicants, including demographic data (gender, marital status), financial details (income, loan amount), and credit history. The dataset consists of 598 entries and 13 columns after removing duplicates.

# Fields in the Dataset:

Loan_ID: Unique identifier for each loan application

Gender: Gender of the applicant👨👩

Married: Marital status of the applicant (Yes/No)👰💍

Dependents: Number of dependents of the applicant👨‍👩‍👧‍👦

Education: Applicant's education level (Graduate/Not Graduate)

Self_Employed: Whether the applicant is self-employed (Yes/No)

ApplicantIncome: Income of the applicant

CoapplicantIncome: Income of the co-applicant

LoanAmount: Loan amount applied for

Loan_Amount_Term: Term of the loan in months

Credit_History: Credit history score

Property_Area: Area where the property to be purchased is located (Urban/Semiurban/Rural)

Target varaiable is Loan_Status: Loan approval status (Y = Yes, N = No)

# Purpose and Prediction
The goal of this project is to predict whether a loan application will be approved (Loan_Status) based on various applicant features. This prediction is crucial for financial institutions to automate and streamline the loan approval process, ensuring efficient use of resources and improving customer satisfaction.

## Process Overview
The project involved several stages:
Graphical Analysis





# Data Exploration
## Data Visualization
Histograms and Boxplots: Histograms and boxplots were used to visualize the distribution and spread of numerical features such as ApplicantIncome, CoapplicantIncome, and LoanAmount. 


Countplots: Countplots were employed to explore the distribution of categorical features like Gender, Married, Education, Self_Employed, Property_Area, and Credit_History with respect to loan approval status (Loan_Status). These plots revealed insights into the distribution of applicants across different categories and their respective loan approval rates.
Scatterplots to observe x y relationships between quantitatives variables.

![Screenshot 2024-07-23 235152](https://github.com/user-attachments/assets/a46f6507-96a7-4f58-8901-b973c4966ee6) ![image](https://github.com/user-attachments/assets/5ba7a486-f508-42ca-a9db-8586989d0d43)


## Data Cleaning
we dropped duplicates and handles missing values. Outliers were not removed as they

Missing values were handled as follows:
Loan_Amount_Term: Imputed with the mode (most frequent term).
LoanAmount: Imputed with the mean.
Credit_History and Dependents: Imputed with the mode.
After imputation, the dataset was verified to have no missing values (loan.isna().sum()).
![Screenshot 2024-07-23 235949](https://github.com/user-attachments/assets/efc6f855-62ea-46d1-bcbe-c2f469b55918)


## Data Preprocessing
Distribution of Features
Numerical Features:

ApplicantIncome and CoapplicantIncome exhibit high skewness.
LoanAmount and Dependents also show noticeable skewness, requiring potential transformations for modeling.
Categorical Features:

Showed imbalances in all categories more males than females, more graduates than non-graduates) but it reflects the actual world

## Correlation Analysis
Correlation Matrix: Visualized to identify relationships between numerical variables.
Positive correlations observed between income levels and loan amounts.
![output](https://github.com/user-attachments/assets/2be28ea7-5106-4712-863b-1e80cfdb83bb)


Principal Component Analysis (PCA): Used to identify key features contributing to variance.

Top features for each principal component were identified, showing which variables have the most influence.


Class Distribution of Loan_Status: The dataset shows a class imbalance towards loan approvals (Y), which may require addressing during model training.
X and Y Variables:

X Variables: Gender, Married, Education, Self_Employed, Property_Area, Credit_History, Loan_Amount_Term, ApplicantIncome, CoapplicantIncome, LoanAmount, Dependents
Y Variable: Loan_Status (Binary classification: Y/N)
Observations:

# Model training

Decision Tree Classifier:
Starting with a Decision Tree model not only provides a foundational understanding of the data but also assists in visualizing and interpreting the decision boundaries. it also gives a baseline to compare other models

Model Training: Utilized DecisionTreeClassifier with varying depths (1 to 20) to find optimal max_depth.
Performance Metrics: Evaluated models based on training and test accuracy, as well as F1 scores.
Validation: Identified the best model based on maximum test accuracy achieved.
We visualize decision tree at the best depth.
![output2](https://github.com/user-attachments/assets/bc71ded6-509e-4154-a04e-e4b7e0603b75)
