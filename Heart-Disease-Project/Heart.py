#!/usr/bin/env python
# coding: utf-8

# 
# # Heart Disease Prediction

# In[1]:


# Importing Libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# for displaying all feature from dataset:
pd.pandas.set_option('display.max_columns', None)


# In[3]:


# Reading Dataset:
dataset = pd.read_csv("Heart_data.csv")
# Top 5 records:
dataset.head()


# In[6]:


dataset.tail()


# - age
# - sex (1 = male; 0 = female)
# - chest pain type (4 values)
# - resting blood pressure
# - serum cholestoral in mg/dl
# - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
# - resting electrocardiographic results (values 0,1,2)
# - maximum heart rate achieved
# - exercise induced angina (1 = yes; 0 = no)
# - oldpeak = ST depression induced by exercise relative to rest
# - the slope of the peak exercise ST segment
# - number of major vessels (0-3) colored by flourosopy
# - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[4]:


# Shape of dataset:
dataset.shape


# In[5]:


# Cheaking Missing (NaN) Values:
dataset.isnull().sum()


# - **There is No NaN Values present.**

# In[6]:


# Datatypes:
dataset.dtypes


# In[7]:


# Description:
dataset.describe()


# In[8]:


# Target feature:
print("Heart Disease People     : ", dataset['target'].value_counts()[1])
print("Not Heart Disease People : ", dataset['target'].value_counts()[0])


# In[9]:


# Printing How many Unique values present in each feature: 
for feature in dataset.columns:
    print(feature,":", len(dataset[feature].unique()))


# In[10]:


# Correlation using Heatmap:
plt.figure(figsize=(12,8))
sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
plt.show()


# - **There is No Multi-Collinearity between two independent feature.**

# In[11]:


dataset.head()


# In[12]:


# Independent and Dependent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[13]:


X.head()


# In[14]:


y.head()


# In[15]:


# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# In[16]:


print(X_train.shape)
print(X_test.shape)


# - **We are not doing Standardization and Normalization of our dataset, as we using Ensemble Technique.**

# In[17]:


# Importing Performance Metrics:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[18]:


# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[19]:


# AdaBoostClassifier:
from sklearn.ensemble import AdaBoostClassifier
AdaBoost = AdaBoostClassifier()
AdaBoost = AdaBoost.fit(X_train,y_train)

# Predictions:
y_pred = AdaBoost.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[20]:


# GradientBoostingClassifier:
from sklearn.ensemble import GradientBoostingClassifier
GradientBoost = GradientBoostingClassifier()
GradientBoost = GradientBoost.fit(X_train,y_train)

# Predictions:
y_pred = GradientBoost.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

import pickle
from sklearn.ensemble import RandomForestClassifier
with open('Heart.pkl', 'wb') as f:
    pickle.dump(RandomForest, f)


# In[ ]:





# In[ ]:




