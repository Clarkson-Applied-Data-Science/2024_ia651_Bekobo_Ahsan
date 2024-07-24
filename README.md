# 2024_ia651_Bekobo_Ahsan
Code for 2024_ia651_Bekobo_Ahsan
#  Predictive Modeling of Loan Approval Status: A Machine Learning Approach
![Screenshot 2024-07-22 183831](https://github.com/user-attachments/assets/a2a8efaf-be86-4637-b4cd-941d909564b1)


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

##Purpose and Prediction
The goal of this project is to predict whether a loan application will be approved (Loan_Status) based on various applicant features. This prediction is crucial for financial institutions to automate and streamline the loan approval process, ensuring efficient use of resources and improving customer satisfaction.

##Process Overview
The project involved several stages:
Graphical Analysis
Histograms and Boxplots: Histograms and boxplots were used to visualize the distribution and spread of numerical features such as ApplicantIncome, CoapplicantIncome, and LoanAmount. 

Countplots: Countplots were employed to explore the distribution of categorical features like Gender, Married, Education, Self_Employed, Property_Area, and Credit_History with respect to loan approval status (Loan_Status). These plots revealed insights into the distribution of applicants across different categories and their respective loan approval rates.
Scatterplots to observe x y relationships between quantitatives variables.



##EDA Insights

Missing Values
Missing values were handled as follows:

Loan_Amount_Term: Imputed with the mode (most frequent term).
LoanAmount: Imputed with the mean.
Credit_History and Dependents: Imputed with the mode.
After imputation, the dataset was verified to have no missing values (loan.isna().sum()).

Distribution of Features
Numerical Features:

ApplicantIncome and CoapplicantIncome exhibit high skewness.
LoanAmount and Dependents also show noticeable skewness, requiring potential transformations for modeling.
Categorical Features:

Showed imbalances in all categories more males than females, more graduates than non-graduates) but it reflects the actual world

Correlation Analysis
Correlation Matrix: Visualized to identify relationships between numerical variables.
Positive correlations observed between income levels and loan amounts.
Feature Importance
Principal Component Analysis (PCA): Used to identify key features contributing to variance.
Top features for each principal component were identified, showing which variables have the most influence.

Class Distribution of Loan_Status: The dataset shows a class imbalance towards loan approvals (Y), which may require addressing during model training.
X and Y Variables:

X Variables: Gender, Married, Education, Self_Employed, Property_Area, Credit_History, Loan_Amount_Term, ApplicantIncome, CoapplicantIncome, LoanAmount, Dependents
Y Variable: Loan_Status (Binary classification: Y/N)
Observations:


Feature Distributions: Notably, features like Credit_History and Property_Area could strongly influence loan approval decisions due to their distributions and relationships with Loan_Status.
