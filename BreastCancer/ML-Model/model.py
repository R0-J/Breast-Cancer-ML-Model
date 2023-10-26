# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:14:42 2023

@author: L
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Read the CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Drop the 'id' and 'Unnamed: 32' columns from the DataFrame
df = df.drop(['id', 'Unnamed: 32'], axis=1)


# Create an instance of the LabelEncoder class
le = LabelEncoder()

# Encode the 'diagnosis' column in the DataFrame
df['diagnosis_encoded'] = le.fit_transform(df['diagnosis'])


# Split the DataFrame into X & Y training sets
X = df.drop(['diagnosis','diagnosis_encoded'], axis=1)
Y = df[['diagnosis_encoded']]


# Create a Logistic Regression model object
LR = LogisticRegression()

# Fit the model using the training data
LR.fit(X, Y)

pickle.dump(LR, open('model.pkl','wb'))

# Make predictions on the trainging data
print(LR.predict(X))
