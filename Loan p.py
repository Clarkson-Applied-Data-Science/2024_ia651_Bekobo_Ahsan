# %% [markdown]
# # Predictive Modeling of Loan Approval Status: A Machine Learning Approach

# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tabulate import tabulate


# %% [markdown]
# ## Importing Dataset

# %%
loan = pd.read_csv("LoanApprovalPrediction.csv")
loan = loan.drop_duplicates()

# Load top 5 Sample
loan.head(5)


# %% [markdown]
# ### Dataset dimensions and statistics

# %%
print("The shape =", loan.shape)
rows,cols = loan.shape
num_features = cols - 1
num_data = rows * cols

# Print the information about the dataset
print(f"Number of Rows of loan dataset: {rows}")
print(f"Number of Columns of loan datset: {cols}")
print(f"Number of Features of loan dataset: {num_features}")
print(f"Number of All entries in loan dataset: {num_data}")


# %%
# Variables and Data Types
variables = loan.columns
data_types = loan.dtypes

variables_table = pd.DataFrame({'Variable': variables, 'Data Type': data_types})
print("Variables and Data Types:")
print(variables_table)
table_data = [(var, dtype) for var, dtype in zip(variables, data_types)]



# %%
loan.describe()

# %%
loan.describe(include=object)

# %% [markdown]
# ### Missing Data

# %%
# Handling missing variables
loan.isna().sum().sum()
loan.isnull().sum()


# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
class_distribution = loan['Loan_Status'].value_counts()

# Plot the class distribution
plt.figure(figsize=(8, 6))
ax = class_distribution.plot(kind='bar', color='steelblue')
plt.xlabel('Class')
plt.ylabel('Count')

plt.title('Loan response Distribution ')
ax.grid(False)
# Add value labels on top of each bar
for i, count in enumerate(class_distribution):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')


plt.show()

# %%
categorical= ['Gender', 'Married',  'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']


numerical = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Dependents','Loan_Amount_Term']



# %% [markdown]
# ### Counter Plot

# %%
fig, axes = plt.subplots(4, 2, figsize=(12, 15))

# Plot each categorical column
for idx, cat_col in enumerate(categorical):
    row, col = idx // 2, idx % 2
    sns.countplot(x=cat_col, data=loan, hue='Loan_Status', ax=axes[row, col],palette='Set2')
    axes[row, col].set_title(f'Countplot of {cat_col} by Loan Status')
    axes[row, col].set_xlabel(cat_col)
    axes[row, col].set_ylabel('Count')
    axes[row, col].legend(title='Loan Status')

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Display the plots
plt.show()

# %% [markdown]
# ### Histogram and Box Plot

# %%
for col in numerical:
    # Create a figure with two subplots (histogram and boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    sns.histplot(loan[col], ax=axes[0], kde=True)  # Use sns.histplot for histogram with KDE
    axes[0].set_title(f'Histogram of {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')

    # Boxplot
    sns.boxplot(x=loan[col], ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Value')

    # Adjust layout
    plt.tight_layout()

    # Show plots
    plt.show()

# %% [markdown]
# ### Scatter Plot

# %%
graduate_df = loan[loan['Education'] == 'Graduate']

# Create scatterplot with color-coded Loan_Status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Dependents', y='LoanAmount', size='ApplicantIncome',hue='Loan_Status', data=graduate_df, palette='Set2', s=100)
plt.title('Scatterplot of Applicant Income vs Number of dependents (Colored by Loan Status for Graduates)')
plt.xlabel('Dependents')
plt.ylabel('Loan Amount')
plt.legend(title='Loan Status')
plt.grid(True)
plt.show()

# %%
graduate_d = loan[loan['Education'] == 'Non-Graduate']

# Create scatterplot with color-coded Loan_Status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', size='Dependents',hue='Loan_Status', data=graduate_df, palette='Set2', s=100)
plt.title('Scatterplot of Applicant Income vs Loan Amount (Colored by Loan Status for NonGraduates)')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.legend(title='Loan Status')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Correlation Matrix

# %%
corr_matrix = loan[numerical].corr()*100

# Plotting the correlation matrix
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, linewidth=.5, vmin=0, vmax=100,
            fmt=".1f", cmap=sns.color_palette("flare", as_cmap=True))
plt.title('Correlation Matrix')
plt.show()

# %%
loan[numerical].skew().sort_values()

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Mode Imputation

# %%
#Loan_Amount_Term: Imputed 14 missing variables with the mode (most frequent term).
#LoanAmount: Imputed 21 variables with the median to prevent skewed variables.
#Dependents: Imputed 12 missing dependent variables and Credit_History: 49 credit history missing with the mode of each category.
loan = loan.drop(columns=['Loan_ID'])
loan["Loan_Amount_Term"] = loan["Loan_Amount_Term"].fillna(loan["Loan_Amount_Term"].mode()[0])
loan["LoanAmount"] = loan["LoanAmount"].fillna(loan["LoanAmount"].median())
loan["Credit_History"] = loan["Credit_History"].fillna(loan["Credit_History"].mode()[0])
loan["Dependents"] = loan["Dependents"].fillna(loan["Dependents"].mode()[0])
loan.isna().sum()

# %% [markdown]
# ### Standardization - Square root

# %%

#scaler = RobustScaler()
#loan[['ApplicantIncome', 'CoapplicantIncome']] = scaler.fit_transform(loan[['ApplicantIncome', 'CoapplicantIncome']])
loan.ApplicantIncome = np.sqrt(loan.ApplicantIncome)
loan.CoapplicantIncome = np.sqrt(loan.CoapplicantIncome)
loan.LoanAmount = np.sqrt(loan.LoanAmount)

# %%
loan = pd.get_dummies(loan, drop_first=True)

newColunmsNames = {'Gender_Male': 'Gender',
                   'Married_Yes': 'Married',
                   'Education_Not Graduate': 'Education',
                   'Self_Employed_Yes': 'Self_Employed',
                   'Loan_Status_Y': 'Loan_Status'}

#Assigning new columns names
loan.rename(columns=newColunmsNames, inplace=True)
loan

# %%
# Seperating feature and Result

X = loan.drop(['Loan_Status'], axis=1)

y = loan['Loan_Status']

# %% [markdown]
# ### Principal Component Analysis (PCA)
# Used to identify key features contributing to variance. Top features for each principal component were identified, showing which variables have the most influence.

# %%

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
pca = PCA(n_components=10, random_state=42)

pca.fit(scaled_data)

# Transform the data to principal components
pca_components = pca.transform(scaled_data)

# Convert PCA components to DataFrame (for visualization or further analysis)
pca_df = pd.DataFrame(data=pca_components, columns=[f"PC{i+1}" for i in range(pca.n_components_)])

# Plot explained variance ratio in a scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100, align='center', alpha=0.8)
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1), [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)])
plt.ylabel('Variance Explained (%)')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# Print top features for each principal component
components_df = pd.DataFrame(pca.components_, columns=X.columns)

for i, component in enumerate(components_df.values):
    print(f"Top features for Principal Component {i + 1}:")
    top_features = X.columns[np.abs(component).argsort()[::-1][:5]]  # Top 5 features
    table = tabulate(pd.DataFrame(top_features), headers=['Top Features'], showindex=False, tablefmt='pretty')
    print(table)
    print()
    #print(top_features)
    #print()






# %% [markdown]
# ### Clustering Analysis

# %%

from sklearn.cluster import KMeans
X_pca = pca.fit_transform(scaled_data)

# Perform clustering on PCA-transformed data
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Visualize clusters in PCA space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=60)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering in PCA-reduced space')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Training and testing data

# %%

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify =y,random_state =42)


# %% [markdown]
# ## Decision Tree Classifier (Overfitted)

# %%
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

plt.figure(figsize=(15, 10))
plot_tree(model,
          feature_names=X.columns,
          class_names=[str(cls) for cls in model.classes_],
          filled=True)

plt.title("Decision Tree Visualization")
plt.show()

# %%
#
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
training_accuracy = []
test_accuracy = []
training_f1 = []
test_f1 = []
tree_depths = []
test_roc_auc = []
max_val_accuracy = 0
best_depth = 0
for depth in range(1,20):
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    tree_clf.fit(X_train,y_train)
    y_training_pred = tree_clf.predict(X_train)
    y_test_pred = tree_clf.predict(X_test)

    training_acc = accuracy_score(y_train,y_training_pred)
    train_f1 = f1_score(y_train,y_training_pred)
    test_acc = accuracy_score(y_test,y_test_pred)
    test_f1_score = f1_score(y_test,y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    training_accuracy.append(training_acc)
    test_accuracy.append(test_acc)
    training_f1.append(train_f1)
    test_f1.append(test_f1_score)
    tree_depths.append(depth)

    # Track maximum validation accuracy and corresponding depth
    if test_acc > max_val_accuracy:
        max_val_accuracy = test_acc
        best_depth = depth

# Create a DataFrame from the collected metrics
Tuning_Max_depth = {
    "Training Accuracy": training_accuracy,
    "Test Accuracy Accuracy": test_accuracy,
    "Training F1": training_f1,
    "test F1": test_f1,
    "Test ROC AUC": test_roc_auc,
    "Max_Depth": tree_depths
}
Tuning_Max_depth_df = pd.DataFrame.from_dict(Tuning_Max_depth)

# Print the results
print(f"Maximum Validation Accuracy: {max_val_accuracy} at depth: {best_depth}")

# Print the entire DataFrame
print(Tuning_Max_depth_df)


# %% [markdown]
# ## Building Model, Hyper Parameter Tuning, Training and Testing
# List of Models
# 
# * Logistic Regression
# * SVC
# * Random Forest Classifier

# %%
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

# %%
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score


# Pipelines for each classifier
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=1000))
])

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True))
])

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

# Hyperparameters for grid search
params = {
    'LogisticRegression': {
        'logistic__C': [0.1, 1, 10, 100],
        'logistic__penalty': ['l2']
    },
    'SVC': {
        'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svc__degree': [1, 2, 3, 4],
        'svc__C': [1, 10, 100],
        'svc__gamma': ['scale', 'auto']
    },
    'RandomForestClassifier': {
        'rf__n_estimators': [10, 50, 100],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }
}

# Different KFold strategies for cross-validation
kfold_strategies = {
    'StratifiedKFold_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold_3': StratifiedKFold(n_splits=3, shuffle=True, random_state=42),

    'KFold_5': KFold(n_splits=5, shuffle=True, random_state=42)
}


best_kfold_strategies = {}
for clf_name, clf_pipeline in [('LogisticRegression', logistic_pipeline),
                               ('SVC', svc_pipeline),
                               ('RandomForestClassifier', rf_pipeline)]:
    best_score = -np.inf
    best_kfold_name = None
    for kfold_name, kfold_strategy in kfold_strategies.items():
        grid_search = GridSearchCV(clf_pipeline, param_grid=params[clf_name], cv=kfold_strategy)
        grid_search.fit(X, y)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_kfold_name = kfold_name
            best_kfold_strategy = kfold_strategy

    best_kfold_strategies[clf_name] = best_kfold_strategy
    print(f"Best KFold Strategy for {clf_name}: {best_kfold_name}")
    print(f"Best Score: {best_score:.4f}")

metrics_data = []
# Train and evaluate models using the best KFold strategy
for clf_name, best_kfold_strategy in best_kfold_strategies.items():
    clf_pipeline = {
        'LogisticRegression': logistic_pipeline,
        'SVC': svc_pipeline,
        'RandomForestClassifier': rf_pipeline
    }[clf_name]

    grid_search = GridSearchCV(clf_pipeline, param_grid=params[clf_name], cv=best_kfold_strategy)
    grid_search.fit(X, y)

    best_clf = grid_search.best_estimator_
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Model: {clf_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print()

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{clf_name} Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.grid(False)
    plt.show()

    # Plot ROC Curve
    if hasattr(best_clf, 'predict_proba'):
        y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = best_clf.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{clf_name} ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{clf_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")
    print()

    metrics_data.append({
        'Model': clf_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'ROC AUC': roc_auc
    })

# DataFrame for better visualization
metrics_df = pd.DataFrame(metrics_data)

# Print the metrics table
print("Metrics Table:")
print(metrics_df)
print()

# Plot the results
plt.figure(figsize=(12, 8))

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score','ROC AUC']:
    plt.plot(metrics_df['Model'], metrics_df[metric], marker='o', label=f'{metric}')

plt.xlabel('Model')
plt.ylabel('Metric Value')
plt.title('Performance Metrics of Different Models')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(metrics_df['Model'], metrics_df['Accuracy'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models on Test Data with Best Cross-Validation Strategy')
plt.ylim([0.0, 1.0])
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# ### Applying SMOTE

# %%
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score


# Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Pipelines for each classifier
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=1000))
])

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True))
])

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeClassifier(random_state=42))
])

# Hyperparameters for grid search
params = {
    'LogisticRegression': {
        'logistic__C': [0.1, 1, 10, 100],
        'logistic__penalty': ['l2']
    },
    'SVC': {
        'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svc__degree': [1, 2, 3, 4],
        'svc__C': [1, 10, 100],
        'svc__gamma': ['scale', 'auto']
    },
    'RandomForestClassifier': {
        'rf__n_estimators': [10, 50, 100],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    },

    'DecisionTreeClassifier': {
        'dt__max_depth': range(1, 15),
        'dt__min_samples_split': range(2, 20, 2),
        'dt__min_samples_leaf': range(1, 20, 2),
        'dt__criterion': ['gini', 'entropy']
    }

}
# Different KFold strategies for cross-validation
kfold_strategies = {
    'StratifiedKFold_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold_3': StratifiedKFold(n_splits=3, shuffle=True, random_state=42),

    'KFold_5': KFold(n_splits=5, shuffle=True, random_state=42)
}


best_kfold_strategies = {}
for clf_name, clf_pipeline in [('LogisticRegression', logistic_pipeline),
                               ('SVC', svc_pipeline),
                               ('RandomForestClassifier', rf_pipeline),('DecisionTreeClassifier', dt_pipeline)]:
    best_score = -np.inf
    best_kfold_name = None
    for kfold_name, kfold_strategy in kfold_strategies.items():
        grid_search = GridSearchCV(clf_pipeline, param_grid=params[clf_name], cv=kfold_strategy)
        grid_search.fit(X, y)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_kfold_name = kfold_name
            best_kfold_strategy = kfold_strategy

    best_kfold_strategies[clf_name] = best_kfold_strategy
    print(f"Best KFold Strategy for {clf_name}: {best_kfold_name}")
    print(f"Best Score: {best_score:.4f}")

metrics_data = []
# Train and evaluate models using the best KFold strategy
for clf_name, best_kfold_strategy in best_kfold_strategies.items():
    clf_pipeline = {
        'LogisticRegression': logistic_pipeline,
        'SVC': svc_pipeline,
        'RandomForestClassifier': rf_pipeline,
        'DecisionTreeClassifier': dt_pipeline
    }[clf_name]

    grid_search = GridSearchCV(clf_pipeline, param_grid=params[clf_name], cv=best_kfold_strategy)
    grid_search.fit(X, y)

    best_clf = grid_search.best_estimator_
    best_clf.fit(X_train_smote, y_train_smote)

    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Model: {clf_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print()

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{clf_name} Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.grid(False)
    plt.show()

    # Plot ROC Curve
    if hasattr(best_clf, 'predict_proba'):
        y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = best_clf.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{clf_name} ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{clf_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")
    print()

    metrics_data.append({
        'Model': clf_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'ROC AUC': roc_auc
    })

# Create a DataFrame for better visualization
metrics_df = pd.DataFrame(metrics_data)

# Print the metrics table
print("Metrics Table:")
print(metrics_df)
print()

# Plot the results using plt.plot with markers
plt.figure(figsize=(12, 8))

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
    plt.plot(metrics_df['Model'], metrics_df[metric], marker='o', label=f'{metric}')

plt.xlabel('Model')
plt.ylabel('Metric Value')
plt.title('Performance Metrics of Different Models')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(metrics_df['Model'], metrics_df['Accuracy'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models on Test Data with Best Cross-Validation Strategy')
plt.ylim([0.0, 1.0])
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# ## Feature importance based on Random Forest

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Perform GridSearchCV for RandomForestClassifier using the best KFold strategy
grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=params['RandomForestClassifier'],
    cv=best_kfold_strategies['RandomForestClassifier']
)
grid_search.fit(X, y)

# Extract the best RandomForestClassifier
best_rf_pipeline = grid_search.best_estimator_

# Train the best RandomForest model
best_rf_pipeline.fit(X_train_smote, y_train_smote)

# Make predictions and evaluate
y_pred = best_rf_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Extract feature importances
feature_importances = best_rf_pipeline.named_steps['rf'].feature_importances_

# Get feature names from the DataFrame, or create default names if X is not a DataFrame
if hasattr(X, 'columns'):
    feature_names = X.columns
else:
    feature_names = [f'Feature {i}' for i in range(len(feature_importances))]

# Create a DataFrame for feature importances
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Print feature importances
print("\nFeature Importances (sorted):")
print(importances_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances for RandomForestClassifier')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()


# %% [markdown]
# ## Testing on New Data

# %%
# Test set
test_data = {
    "Gender": ["Female", "Male", "Female", "Male"],
    "Married": ["Yes", "No", "Yes", "No"],
    "Dependents": [1, 0, 2, 1],
    "Education": ["Not Graduate", "Graduate", "Not Graduate", "Graduate"],
    "Self_Employed": ["No", "Yes", "Yes", "No"],
    "ApplicantIncome": [5000, 30000, 0, 4000],
    "CoapplicantIncome": [2000, 25000, 0, 1000],
    "LoanAmount": [150, 100, 10000, 80],
    "Loan_Amount_Term": [360, 180, 240, 120],
    "Credit_History": [1, 0, 1, 1],
    "Property_Area": ["Urban", "Semiurban", "Rural", "Urban"],

}
Loan_Status: ["Approved", "Denied", "Approved", "Approved"]
# Create the DataFrame
df_test = pd.DataFrame(test_data)

# Display the DataFrame
print(df_test)

# %%
test_dummies = pd.get_dummies(df_test, drop_first=True)
newColunmsNames = {'Gender_Male': 'Gender',
                   'Married_Yes': 'Married',
                   'Education_Not Graduate': 'Education',
                   'Self_Employed_Yes': 'Self_Employed',
                   'Loan_Status_Y': 'Loan_Status'}

#Assigning new columns names
test_dummies.rename(columns=newColunmsNames, inplace=True)

test_dummies

# %%
y_pred = best_rf_pipeline.predict(test_dummies)

Loan_Statu= ["True", "False", "True", "True"]
results_df = pd.DataFrame({
    'Actual_Loan_Status':Loan_Statu ,
    'Predicted_Loan_Status': y_pred
})

# Display the first few rows of the DataFram
results_df.head()



