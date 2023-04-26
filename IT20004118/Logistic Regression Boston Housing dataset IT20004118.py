#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# In[54]:


# Load the dataset
data = pd.read_csv('HousingData.csv')


# In[55]:


# Drop rows with missing values since there are N/A values
data.dropna(inplace=True)


# In[56]:


# Create a binary target variable HIGH_MEDV
median_MEDV = data['MEDV'].median()
data['HIGH_MEDV'] = np.where(data['MEDV'] > median_MEDV, 1, 0)


# In[57]:


# Split the dataset into training and test sets
X = data.drop(['MEDV', 'HIGH_MEDV'], axis=1)
y = data['HIGH_MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


# Train a logistic regression model
logreg = LogisticRegression(max_iter=10000) # increase max_iter to 10000
logreg.fit(X_train, y_train)


# In[59]:


# Predict on the test set and compute accuracy and confusion matrix
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# In[60]:


print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)


# In[ ]:




