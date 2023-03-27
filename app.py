from flask import Flask, jsonify, request, render_template
from flask_pydantic import validate

import numpy as np
import pickle
import pandas as pd
import plotly
import plotly.express as px
import json

from dto.dto import *

app = Flask(__name__)
pipeline_credit = pickle.load(open('pipeline/credit_score_classification_pipeline.pkl', 'rb'))
pipeline_insurance = pickle.load(open('pipeline/insurance_regression_pipeline.pkl', 'rb'))
pipeline_customers_clustering = pickle.load(open('pipeline/customer_clustering_pipeline.pkl', 'rb'))
pipeline_customers_pca = pickle.load(open('pipeline/customer_pca_pipeline.pkl', 'rb'))

label_enc_sex = pickle.load(open('pipeline/insurance_label_enc_sex.pkl', 'rb'))
label_enc_smoker = pickle.load(open('pipeline/insurance_label_enc_smoker.pkl', 'rb'))
label_enc_region = pickle.load(open('pipeline/insurance_label_enc_region.pkl', 'rb'))
label_enc_profession = pickle.load(open('pipeline/customer_label_enc_profession.pkl', 'rb'))

@app.route('/api/credit-score/predict', methods=['POST'])
@validate()
def predict_credit_score(body: CreditScoreInputData):
    pipeline_input = np.array([v for k,v in body.dict().items()]).reshape(1,-1)
    result = pipeline_credit.predict(pipeline_input)
    return jsonify({'score': result[0]})

@app.route('/api/medical-insurance/predict', methods=['POST'])
@validate()
def predict_medical_insurance(body: InsuranceInputData):
    data_input = body.dict()

    data_input['sex'] = label_enc_sex.transform([data_input['sex']])[0]
    data_input['smoker'] = label_enc_smoker.transform([data_input['smoker']])[0]
    data_input['region'] = label_enc_region.transform([data_input['region']])[0]

    pipeline_input = np.array([v for k,v in data_input.items()]).reshape(1,-1)

    result = pipeline_insurance.predict(pipeline_input)
    return jsonify({'charges': result[0]})

@app.route('/')
def visualize_cluster():
    df_customers = pd.read_csv('data/customer-data/customers.csv')

    df_customers_preprocess = df_customers.copy(deep=True)

    df_customers_preprocess = df_customers_preprocess.drop('CustomerID', axis=1)
    df_customers_preprocess = df_customers_preprocess.drop('Gender', axis=1)

    df_customers_preprocess['Profession'] = label_enc_profession.transform(df_customers_preprocess['Profession'])

    X = df_customers_preprocess.to_numpy()
    y = pipeline_customers_clustering.predict(X)

    X_scaled_pca = pipeline_customers_pca.transform(X)

    bar_fig = px.bar(
        df_customers['Profession'].value_counts(), 
        title='Profession Types',
        labels={
            "index": "Profession",
            "value": "Count"
        }
    )

    box_fig = px.box(
        df_customers, 
        x="Profession", 
        y="Spending Score (1-100)",
        title="Spending Score Distribution by Profession")

    scatter_fig = px.scatter(
        x=X_scaled_pca[:,0], 
        y=X_scaled_pca[:,1], 
        text=df_customers['CustomerID'],
        color=y,
        title='Scatter Plot of Customer Data Clustering (PCA)')
    
    bar_graph_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)
    box_graph_json = json.dumps(box_fig, cls=plotly.utils.PlotlyJSONEncoder)
    scatter_graph_json = json.dumps(scatter_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(
        'plotly.html', 
        bar_graph_json=bar_graph_json,
        box_graph_json=box_graph_json,
        scatter_graph_json=scatter_graph_json)
