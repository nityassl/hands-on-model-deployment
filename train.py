import numpy as np
import pandas as pd
import pickle
import time

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Part 1: Credit Score Classification

feature_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
               'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
               'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 
                'Amount_invested_monthly', 'Monthly_Balance']
label_col = 'Credit_Score'

df = pd.read_csv('data/credit-score/train.csv')

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

cols = feature_cols + [label_col]
df = df[cols].dropna()

X = np.array(df[feature_cols].values)
y = np.array(df[label_col])

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=2023, test_size=0.2)

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rf', RandomForestClassifier(n_estimators=10))
])

print("Training Model...")
pipeline.fit(X_train, y_train)
print("Done Training. Validating Model...")

print(classification_report(y_val, pipeline.predict(X_val)))

pickle.dump(pipeline, open('pipeline/credit_score_classification_pipeline.pkl', 'wb'))

# Part 2: Medical Insurance Cost Prediction
# akan dijadikan latihan

df_insurance = pd.read_csv('data/medical-insurance/insurance.csv')

label_enc_sex = LabelEncoder().fit(df_insurance['sex'])
label_enc_smoker = LabelEncoder().fit(df_insurance['smoker'])
label_enc_region = LabelEncoder().fit(df_insurance['region'])

pickle.dump(label_enc_sex, open('pipeline/insurance_label_enc_sex.pkl', 'wb'))
pickle.dump(label_enc_smoker, open('pipeline/insurance_label_enc_smoker.pkl', 'wb'))
pickle.dump(label_enc_region, open('pipeline/insurance_label_enc_region.pkl', 'wb'))

df_insurance['sex'] = label_enc_sex.transform(df_insurance['sex'])
df_insurance['smoker'] = label_enc_smoker.transform(df_insurance['smoker'])
df_insurance['region'] = label_enc_region.transform(df_insurance['region'])

pipeline_insurance = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rf', RandomForestRegressor(n_estimators=10))
])

X = np.array(df_insurance.drop('charges', axis=1).values)
y = np.array(df_insurance['charges'])
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=2023, test_size=0.2)

print("Training Model...")
pipeline_insurance.fit(X_train, y_train)
print("Done Training. Validating Model...")

print("R^2 score:", pipeline_insurance.score(X_val, y_val))
print("MAE:", mean_absolute_error(y_val, pipeline_insurance.predict(X_val)))
print("RMSE:", np.sqrt(mean_squared_error(y_val, pipeline_insurance.predict(X_val))))

pickle.dump(pipeline_insurance, open('pipeline/insurance_regression_pipeline.pkl', 'wb'))


# Part 3: Visualize Clustering from Customers Data
df_customers = pd.read_csv('data/customer-data/customers.csv')
df_customers_preprocess = df_customers.copy(deep=True)

df_customers_preprocess = df_customers_preprocess.drop('CustomerID', axis=1)
df_customers_preprocess = df_customers_preprocess.drop('Gender', axis=1)

label_enc_profession = LabelEncoder().fit(df_customers_preprocess['Profession'])
pickle.dump(label_enc_profession, open('pipeline/customer_label_enc_profession.pkl', 'wb'))

df_customers_preprocess['Profession'] = label_enc_profession.transform(df_customers_preprocess['Profession'])

X = df_customers_preprocess.to_numpy()
pipeline_pca = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=2))
])
pipeline_clustering = Pipeline([
    ('scaler', MinMaxScaler()),
    ('kmeans', KMeans(n_clusters=5, random_state=2023))
])

print("Training Model...")
pipeline_clustering.fit(X)
pipeline_pca.fit(X)
print("Done Training")

pickle.dump(pipeline_clustering, open('pipeline/customer_clustering_pipeline.pkl', 'wb'))
pickle.dump(pipeline_pca, open('pipeline/customer_pca_pipeline.pkl', 'wb'))
