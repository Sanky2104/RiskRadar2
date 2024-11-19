#!/usr/bin/env python
# coding: utf-8

# # Medical Insurance Cost - Machine Learning Project

# In[1]:


# Importing Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading Train dataset:
train_data = pd.read_csv('Train_Data.csv')
# Top 5 records:
train_data.head()


# In[3]:


# Shape of dataset:
train_data.shape


# In[4]:


# Cheacking for NaN Values (Missing Values):
train_data.isnull().sum()


# In[5]:


# Insights of dataset:
train_data.info()


# In[6]:


# Description of dataset (Numerical):
train_data.describe()


# In[7]:


# Description of dataset (Categorical):
train_data.describe(include=['O'])


# #### Medical Insurance Charges:

# In[8]:


# Histrogram of Medical Insurance Charges:
plt.figure(figsize=(8,5))
sns.histplot(train_data['charges'], kde=True)
plt.title('Medical Insurance Charges', fontsize=20)
plt.show()


# In[9]:


# Boxplot of Medical Insurance Charges:
plt.figure(figsize=(8,5))
sns.boxplot(train_data['charges'])
plt.title('Medical Insurance Charges (Boxplot)', fontsize=20)
plt.show()


# #### Age:

# In[10]:


# Histrogram of Age:
plt.figure(figsize=(8,5))
sns.histplot(train_data['age'], kde=True)
plt.title('Age', fontsize=20)
plt.show()


# In[11]:


# Boxplot of Age:
plt.figure(figsize=(8,5))
sns.boxplot(train_data['age'])
plt.title('Age (boxplot)', fontsize=20)
plt.show()


# #### Body Mass Index:

# In[12]:


# Histrogram of Body Mass Index:
plt.figure(figsize=(8,5))
sns.histplot(train_data['bmi'], kde=True)
plt.title('Body Mass Index', fontsize=20)
plt.show()


# In[13]:


# Boxplot of Body Mass Index:
plt.figure(figsize=(8,5))
sns.boxplot(train_data['bmi'])
plt.title('Body Mass Index (Boxplot)', fontsize=20)
plt.show()


# #### children:

# In[14]:


# Histrogram of children:
plt.figure(figsize=(8,5))
sns.histplot(train_data['children'], kde=True)
plt.title('childrens', fontsize=20)
plt.show()


# In[15]:


# Boxplot of children:
plt.figure(figsize=(8,4))
sns.boxplot(train_data['children'])
plt.title('childrens (Boxplot)', fontsize=20)
plt.show()


# #### Sex:

# In[16]:


# Value Counts:
print("Male   :", train_data['sex'].value_counts()[0])
print("Female :", train_data['sex'].value_counts()[1])

# Visualization:
plt.figure(figsize=(6,4))
sns.countplot(train_data['sex'])
plt.title('Sex', fontsize=20)
plt.show()


# #### Smokers:

# In[17]:


# Value Counts:
print("Smokers     :", train_data['smoker'].value_counts()[1])
print("Non-Smokers :", train_data['smoker'].value_counts()[0])

# Visualization:
sns.countplot(train_data['smoker'])
sns.countplot(train_data['smoker'])
plt.title('Smokers', fontsize=20)
plt.show()


# #### Region:

# In[18]:


# Value Counts:
print("South-East region :", train_data['region'].value_counts()[0])
print("North-West region :", train_data['region'].value_counts()[1])
print("South-West region :", train_data['region'].value_counts()[2])
print("North-East region :", train_data['region'].value_counts()[3])

# Visualization:
sns.countplot(train_data['region'])
sns.countplot(train_data['region'])
plt.title('Regions', fontsize=20)
plt.show()


# In[19]:


# top 5 records:
train_data.head()


# In[20]:


# Rounding up & down Age:
train_data['age'] = round(train_data['age'])


# In[21]:


# top 5 records, after rounding up & down Age:
train_data.head()


# In[22]:


# Encoding:
train_data = pd.get_dummies(train_data, drop_first=True)


# In[23]:


# top 2 records, after encoding:
train_data.head(2)


# In[24]:


# Columns of dataset:
train_data.columns


# In[25]:


# Rearranging columns to see better: 
train_data = train_data[['age','sex_male','smoker_yes','bmi','children','region_northwest','region_southeast','region_southwest','charges']]
train_data.head(2)


# In[26]:


# Splitting Independent & Dependent Feature:
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]


# In[27]:


# top 2 records of Independent feature:
X.head(2)


# In[28]:


# top 2 records of Dependent Feature:
y.head(2)


# In[29]:


# Train Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# # Model Building:

# In[30]:


# Importing Performance Metrics:
from sklearn.metrics import mean_squared_error, r2_score


# In[31]:


# Linear Regression:
from sklearn.linear_model import LinearRegression
LinearRegression = LinearRegression()
LinearRegression = LinearRegression.fit(X_train, y_train)

# Prediction:
y_pred = LinearRegression.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


# In[32]:


# Ridge:
from sklearn.linear_model import Ridge
Ridge = Ridge()
Ridge = Ridge.fit(X_train, y_train)

# Prediction:
y_pred = Ridge.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


# In[33]:


# Lasso:
from sklearn.linear_model import Lasso
Lasso = Lasso()
Lasso = Lasso.fit(X_train, y_train)

# Prediction:
y_pred = Lasso.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


# In[34]:


# Random Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor = RandomForestRegressor()
RandomForestRegressor = RandomForestRegressor.fit(X_train, y_train)

# Prediction:
y_pred = RandomForestRegressor.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


# In[35]:


# Creating a pickle file for the classifier
import pickle
filename = 'MedicalInsuranceCost.pkl'
pickle.dump(RandomForestRegressor, open(filename, 'wb'))


# In[ ]:




