Customer Churn Prediction with Python and Power BI

Overview
This project predicts customer churn for a telecom company using machine learning models and presents key insights with Power BI. It demonstrates the end-to-end process of:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Predictive modeling (Logistic Regression & Random Forest)
- Model evaluation (Confusion Matrix, Classification Report, ROC AUC)
- Exporting results to CSV for use in a Power BI dashboard

---

Exploratory Data Analysis
Key findings:
- Churn rate is highest among customers on month-to-month contracts
- Customers with higher monthly charges are more likely to churn
- Customers with short tenure often leave early

Visuals created using `matplotlib` and `seaborn`:
- Churn distribution bar plot
- Correlation heatmap
- Churn by contract type (stacked bar chart)

---

Machine Learning Models

1. Logistic Regression
- ROC AUC: 0.841
- Evaluated using confusion matrix and classification report
- Interpretable model showing linear relationships between features and churn

2. Random Forest Classifier
- ROC AUC: 0.825
- Feature importance plot included
- Predictions exported for Power BI use

PowerBI
Constructed a interactive dashboard to convey finding and insights about customer churn rates


## ⚙️ How to Run the Code
1. Clone this repository
2. Install dependencies:  
