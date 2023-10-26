# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:09:04 2023

@author: L
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    data = request.form.getlist('Breast')
    print(data)
    
    features = np.array([float(item) for item in data], dtype=float)
    inputDataReshape = features.reshape(1,-1)
    prediction = model.predict(inputDataReshape)

    if prediction == 1:
        result = 'Malignant'
    else:
        result = 'Benign'

    return render_template('index.html', Breast = f'The test results indicate that the tumor is {result}')

if __name__ == '__main__':
    app.run(port=8080, debug=True)