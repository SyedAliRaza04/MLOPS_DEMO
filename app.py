#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_salary():
    data = request.json
    years_of_experience = data['years_of_experience']
    regressor = LinearRegression()
    df = pd.read_csv('Salary_Data.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    regressor.fit(X, y)
    salary = regressor.predict([[years_of_experience]])[0]
    return jsonify({'salary': salary})

if __name__ == '__main__':
    app.run()

