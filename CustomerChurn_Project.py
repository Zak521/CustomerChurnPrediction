## Zak Kotschegarow
## Customer Churn Prediction

## import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## load data in
df = pd.read_csv(r"C:\Users\16054\OneDrive - South Dakota State University - SDSU\DA_Projects\Python_Projects\WA_Fn-UseC_-Telco-Customer-Churn.csv")
#print(df)

## total charges has some blanks/null
df.columns = df.columns.str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

#print(df['TotalCharges'].isnull().sum()) ## 11 rows null/blank

#print(df[df['TotalCharges'].isnull()]) ## shows tthe rows, the tennure is 0, so they are a new customer

## fill it with 0
df['TotalCharges'].fillna(0, inplace = True)

## check churn percentages

## value counts
#print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize = True))

## churn distribution
sns.countplot(x = 'Churn', data = df)
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
#plt.show()

## correlation heatmap
## numeric cols
num_cols = df.select_dtypes(include = ['int64', 'float64'])

## create the correlation matrix
corr = num_cols.corr()
## heatmap of only the numeric variables
plt.figure(figsize = (10,6))
sns.heatmap(corr, annot = True, cmap = 'coolwarm')
plt.title('Correlation Heatmap')
plt.show()

## the group of the different contract types and how they churn
df.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean()
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
contract_churn.plot(kind='bar', stacked=True)
plt.title('Churn Percentage by Contract Type')
plt.ylabel('Proportion')
plt.show()

## from above,  loaded in the data set saw that totalcharges had some blanks, fixed those
## look at the percent of people who churn and people who dont, and various plots with churn rate


########################

## Preprocessing the data

## Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No':0})

## Drop Customer ID
df.drop('customerID', axis = 1, inplace = True)

## One-hot encode categorical features
cat_cols = df.select_dtypes(include = 'object').columns ## selects all columns with data type 'object'
df_encoded = pd.get_dummies(df, columns = cat_cols, drop_first = True) ## converts categorical var into our one-hot encoding

## Separate features and target
## X will hold all the input independent variables, dropping 'Churn' since that's the output
X = df_encoded.drop('Churn', axis = 1)
## y will hold the target dependent variable, which is the 'Churn' column
y = df_encoded['Churn']

## Split data
from sklearn.model_selection import train_test_split
## Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, stratify = y, random_state = 42
)
## stratify=y ensures the class distribution in y is maintained in both splits
## random_state ensures reproducibility

print(df.head())
print(df.dtypes)

## Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

## LogisticRegression, lr is my trained logistiuc regression model
lr = LogisticRegression(max_iter = 1000)
lr.fit(X_train, y_train) ## trains logistic regression model using the training data

## Prediction
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

## Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba)) ## 0.841 ROC AUC

print('_______________________________')
## from above, have a target, churn, droped customerID, selected columns with 'object' data type,
## then convert the variables into our one-hot encoding,  seperate features and target, spilt the data,
## train our logistic regression modle to predict churn or not churn

##  random forest ## 

from sklearn.ensemble import RandomForestClassifier

## Initialize  Random Forest model with a fixed random state for reproducibility
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)  ## Fit the model on the training data
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba)) ## 0.825


## The feature importances, how much each feature contributed to the model
importances = pd.Series(rf.feature_importances_, index = X_train.columns)
importances.nlargest(10).plot(kind='barh')
plt.show()

#df['Churn'].value_counts()
#print(df['Churn'].value_counts())


## Power bi dashboard csv file##

df['PredictedChurn'] = rf.predict(X)
df['PredictedProb'] = rf.predict_proba(X)[:, 1]

## Save to CSV for Power BI
df.to_csv('customer_churn_predictions.csv', index=False)




