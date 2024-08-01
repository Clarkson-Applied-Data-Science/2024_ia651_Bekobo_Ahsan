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
Loan_Amount_Term: Imputed 14 missing variables with the mode (most frequent term).
LoanAmount: Imputed 21 variables with the median to prevent skewed variables.
Credit_History and Dependents: Imputed 12 missing dependent variables and 49 credit history missing with the mode of each category.
After imputation, the dataset was verified to have no missing values (loan.isna().sum()).
![Screenshot 2024-07-23 235949](https://github.com/user-attachments/assets/efc6f855-62ea-46d1-bcbe-c2f469b55918)


## Data Preprocessing
-Distribution of Features
Numerical Features:

ApplicantIncome and CoapplicantIncome exhibit high skewness.
LoanAmount and Dependents also show noticeable skewness, requiring potential transformations for modeling.
-Categorical Features:
Showed imbalances in all categories more males than females, more graduates than non-graduates) but it reflects the actual world
Class Distribution of Loan_Status: The dataset shows a class imbalance towards loan approvals (Y), which will be addressed during model training using Smote.

![Screenshot 2024-07-24 050152](https://github.com/user-attachments/assets/f0f59f65-c587-4f97-8a81-bbc123a9a4ba)
## Correlation Analysis
Correlation Matrix: Visualized to identify relationships between numerical variables.
Positive correlations observed between income levels and loan amounts.
![output](https://github.com/user-attachments/assets/2be28ea7-5106-4712-863b-1e80cfdb83bb)


Principal Component Analysis (PCA): Used to identify key features contributing to variance.


![Screenshot 2024-07-24 045705](https://github.com/user-attachments/assets/faa9e06c-530a-482e-9ada-093794711db1)

Top features for each principal component were identified, showing which variables have the most influence.
![Screenshot 2024-07-24 000145](https://github.com/user-attachments/assets/dba42b81-a005-4961-842c-e85690332355)




# Model Selection and evaluation metrics
X and Y Variables:

X Variables: Gender, Married, Education, Self_Employed, Property_Area, Credit_History, Loan_Amount_Term, ApplicantIncome, CoapplicantIncome, LoanAmount, Dependents
Y Variable: Loan_Status (Binary classification: Y/N)

The dataset was split into training (80%) and test (20%) sets using stratified sampling to maintain class distribution.


Four  different models were selected: Decision Tree Logistic Regression, SVC with radial basis function kernel, and Random Forest Classifier.

starting with a Decision Tree model not only provides a foundational understanding of the data but also assists in visualizing and interpreting the decision boundaries. it also gives a baseline to compare other models and the importance of hyperparameters on evaluation metrics

Each of the other three model was trained using a pipeline that included standard scaling of features and hyperparameter tuning using GridSearchCV with different cross-validation strategies (StratifiedKFold and KFold with 5 folds).

## Evaluation Metrics:
To evaluate our model we used: Accuracy,F1-score, Recall, Precison, Confusion Matrix, ROC Curve, Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) were plotted to evaluate the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity).

# Model training
## Decision Tree Classifier:
Starting with a Decision Tree model not only provides a foundational understanding of the data but also assists in visualizing and interpreting the decision boundaries. it also gives a baseline to compare other models and the importance of hyperparameters on evaluation metrics

Model Training: Utilized DecisionTreeClassifier with varying depths (1 to 20) to find optimal max_depth.
Performance Metrics: Evaluated models based on training and test accuracy, as well as F1 scores.
Validation: Identified the best model based on maximum test accuracy achieved.
We visualize decision tree at the best depth.
![output2](https://github.com/user-attachments/assets/bc71ded6-509e-4154-a04e-e4b7e0603b75)

## Decision Tree with Hyperparameters and cross-validation



## Pipeline Construction and Hyperparameter Tuning for lm, SVM, RF:

The introduction of streamlined pipelines for Logistic Regression, SVM, and the Random Forest Classifier proved to be monumental in refining the preprocessing and model fitting processes.
As a standout success, the advanced grid search with cross-validation was a critical step in fine-tuning the hyperparameters for each remarkable model.


1. Best Model Hyperparameters
Logistic Regression

Best Score: 0.8060
Best Parameters:
logistic__C: 0.1
logistic__penalty: 'l1'

Support Vector Classifier (SVC)

Best Score: 0.8077

Best Parameters:
svc__C: 1
svc__degree: 1
svc__gamma: 'scale'
svc__kernel: 'linear'
Random Forest Classifier

Best Score: 0.8094
Best Parameters:
rf__n_estimators: 100
rf__max_depth: 10
rf__min_samples_split: 10
rf__min_samples_leaf: 4

# Results and Metrics
Accuracy seems similar accross for logistic and SVC Model. 


![Screenshot 2024-08-01 102253](https://github.com/user-attachments/assets/4a42ef48-90d7-4725-818d-92d435d1d539)

This is a bar plot displaying the accuracy for each model.
![Screenshot 2024-08-01 102324](https://github.com/user-attachments/assets/29a892e8-826c-44da-b9a6-1050accf5d7e)
![Screenshot 2024-08-01 102308](https://github.com/user-attachments/assets/fe450d77-482a-44a2-a436-1d1952b9c780)


Judging the models therefore requires another metrics . Roc was used to make a decision on the best model. ROC is particularly helpful for binary classifications as it evaluates the model accross all thresholds
## SMOTE Process
To address the challenge of class imbalance in the loan approval prediction dataset, we applied SMOTE, a widely used technique for over-sampling the minority class by generating synthetic samples. Here's an overview of the SMOTE process and its impact on model performance:

SMOTE works by generating synthetic examples rather than simply duplicating existing ones. This approach helps in balancing the class distribution and improving the model's ability to learn from the minority class examples. Here are the steps we followed:

Data Preparation:

Split the original dataset into training (70%) and testing (30%) sets using train_test_split.
Applying SMOTE:

Used the SMOTE function from imblearn.over_sampling module to over-sample the minority class (True - approved loans) in the training set only (X_train, y_train).

# Pipeline Construction and Hyperparameter Tuning for lm, SVM, RF, DT:
Pipelines were defined for each classifier to streamline the training and evaluation process.The best KFold strategy for each classifier was determined through grid search. This time we added Decision Tree to observe the impact of a balanced dataset on all the models cited.
# Results and Metrics Summary:

A Confusion Matrix and Roc curve for each model.

![Screenshot 2024-08-01 092435](https://github.com/user-attachments/assets/435072e5-7b3e-4a2b-a3d7-69f97296e9ca)
![Screenshot 2024-08-01 092446](https://github.com/user-attachments/assets/aae95378-0d64-4f51-b725-15612cbcaf11)
![Screenshot 2024-08-01 092500](https://github.com/user-attachments/assets/81b8aa96-1306-46dd-b36d-6203fec82ab7)
![Screenshot 2024-08-01 092508](https://github.com/user-attachments/assets/1a901744-81d7-4ad7-8b0a-a9c27f76fc88)
![Screenshot 2024-08-01 092602](https://github.com/user-attachments/assets/952abfce-972c-4ea5-adb1-2856ba270801)
![Screenshot 2024-08-01 092611](https://github.com/user-attachments/assets/a193c432-4632-44cf-8b89-7d8765084f95)
![Screenshot 2024-08-01 093026](https://github.com/user-attachments/assets/13bd58cc-aa39-4a17-9151-bed7e3718661)

A bar plot of the accuracy for each model and performance graph displaying all metrics.


![Screenshot 2024-08-01 093034](https://github.com/user-attachments/assets/cd7b396a-99b5-42a2-a7d6-2e37f76b5386)
![Screenshot 2024-08-01 093057](https://github.com/user-attachments/assets/c609b631-93bb-4484-8b19-05511b43b467)
![Screenshot 2024-08-01 093114](https://github.com/user-attachments/assets/e41e3359-cb95-4be6-b3e3-9dd43b4a6fea)

## Testing data
![Screenshot 2024-08-01 113304](https://github.com/user-attachments/assets/4164ae55-20c7-4a7a-93f7-a81df5b88301)

# Model building on test data, Results and comparison
Prediction vs Actual
![Screenshot 2024-08-01 105406](https://github.com/user-attachments/assets/ae5113bd-27c7-4c36-b018-c9124a1eb70e)

## Conclusion and Recommendations
Findings:
hyperparameters does not always increase performance. The Decision Tree model at depth 3 balanced simplicity with performance effectively.
Logistic Regression and SVC models demonstrated competitive performance with optimal parameter tuning.

Recommendations:
Model Deployment: Consider deploying the Decision Tree model (depth 3) for simplicity and good performance unless higher interpretability is needed.
