#!/usr/bin/env python
# coding: utf-8

# 
# # Liver Disease Prediction

# #### Content
# This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). This data set contains 441 male patient records and 142 female patient records.
# 
# Any patient whose age exceeded 89 is listed as being of age "90".
# 
# Columns:
# 
# - Age of the patient
# - Gender of the patient
# - Total Bilirubin
# - Direct Bilirubin
# - Alkaline Phosphotase
# - Alamine Aminotransferase
# - Aspartate Aminotransferase
# - Total Protiens
# - Albumin
# - Albumin and Globulin Ratio
# - Dataset: field used to split the data into two sets (patient with liver disease, or no disease)

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
dataset = pd.read_csv("C:\\Users\\gupta\\OneDrive\\Desktop\\Major\\Trials\\Liver-Disease-Prediction-Project\\Dataset\\Liver_data.csv")
# Top 5 records:
dataset.head()


# In[4]:


# Last 5 records:
dataset.tail()


# In[5]:


# Shape of dataset:
dataset.shape


# In[6]:


# Cheaking Missing (NaN) Values:
dataset.isnull().sum()


# - 'Albumin_and_Globulin_Ratio' feature contain 4 NaN values.

# In[7]:


# Mean & Median of "Albumin_and_Globulin_Ratio" feature:
print(dataset['Albumin_and_Globulin_Ratio'].median())
print(dataset['Albumin_and_Globulin_Ratio'].mean())


# In[8]:


# Filling NaN Values of "Albumin_and_Globulin_Ratio" feature with Median :
dataset['Albumin_and_Globulin_Ratio'] = dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].median())


# In[9]:


# Datatypes:
dataset.dtypes


# In[10]:


# Description:
dataset.describe()


# In[11]:


# Target feature:
print("Liver Disease Patients      :", dataset['Dataset'].value_counts()[1])
print("Non Liver Disease Patients  :", dataset['Dataset'].value_counts()[2])

# Visualization:
sns.countplot(dataset['Dataset'])
plt.show()


# In[12]:


# Histrogram of Age:
plt.figure(figsize=(8,5))
sns.histplot(dataset['Age'], kde=True)
plt.title('Age', fontsize=20)
plt.show()


# In[13]:


dataset.head()


# In[14]:


# Gender feature:
print("Total Male   :", dataset['Gender'].value_counts()[0])
print("Total Female :", dataset['Gender'].value_counts()[1])

# Visualization:
sns.countplot(dataset['Gender'])
plt.show()


# In[15]:


# Printing How many Unique values present in each feature: 
for feature in dataset.columns:
    print(feature,":", len(dataset[feature].unique()))


# In[16]:


# Label Encoding
dataset['Gender'] = np.where(dataset['Gender']=='Male', 1,0)


# In[17]:


dataset.head()


# In[18]:


# Correlation using Heatmap:
plt.figure(figsize=(12,8))
sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
plt.show()


# #### There is Multi-Collinearity found on our dataset.

# In[19]:


dataset.columns


# 1. Multicollinearity betwwen **'Total_Bilirubin'** and **'Direct_Bilirubin'** is **0.87%**
# 2. Multicollinearity betwwen **'Alamine_Aminotransferase'** and **'Aspartate_Aminotransferase' **is **0.79%**
# 3. Multicollinearity betwwen **'Total_Protiens'** and **'Albumin'** is **0.78%**
# 4. Multicollinearity betwwen **'Albumin'** and **'Albumin_and_Globulin_Ratio'** is **0.69%**

# Usually we drop that feature which has above 0.85% multicollinearity between two independent feature.
# Here we have only 'Total_Bilirubin' and 'Direct_Bilirubin' feature which has 0.87% mutlicollinearity. So we drop one of the feature from them
# and other independent feature has less multicollinearity, less than 0.80% So we keep that feature. 

# In[20]:


# Droping 'Direct_Bilirubin' feature:
dataset = dataset.drop('Direct_Bilirubin', axis=1)


# In[21]:


dataset.columns


# In[22]:


sns.distplot(dataset['Albumin'])


# In[23]:


# Calculate the boundaries of Total_Protiens feature which differentiates the outliers:
uppper_boundary=dataset['Total_Protiens'].mean() + 3* dataset['Total_Protiens'].std()
lower_boundary=dataset['Total_Protiens'].mean() - 3* dataset['Total_Protiens'].std()

print(dataset['Total_Protiens'].mean())
print(lower_boundary)
print(uppper_boundary)


# In[24]:


##### Calculate the boundaries of Albumin feature which differentiates the outliers:
uppper_boundary=dataset['Albumin'].mean() + 3* dataset['Albumin'].std()
lower_boundary=dataset['Albumin'].mean() - 3* dataset['Albumin'].std()

print(dataset['Albumin'].mean())
print(lower_boundary)
print(uppper_boundary)


# In[25]:


# Lets compute the Interquantile range of Total_Bilirubin feature to calculate the boundaries:
IQR = dataset.Total_Bilirubin.quantile(0.75)-dataset.Total_Bilirubin.quantile(0.25)

# Extreme outliers
lower_bridge = dataset['Total_Bilirubin'].quantile(0.25) - (IQR*3)
upper_bridge = dataset['Total_Bilirubin'].quantile(0.75) + (IQR*3)

print(lower_bridge)
print(upper_bridge)

# if value greater than upper bridge, we replace that value with upper_bridge value:
dataset.loc[dataset['Total_Bilirubin'] >= upper_bridge, 'Total_Bilirubin'] = upper_bridge


# In[26]:


# Lets compute the Interquantile range of Alkaline_Phosphotase feature to calculate the boundaries:
IQR = dataset.Alkaline_Phosphotase.quantile(0.75) - dataset.Alkaline_Phosphotase.quantile(0.25)

# Extreme outliers
lower_bridge = dataset['Alkaline_Phosphotase'].quantile(0.25) - (IQR*3)
upper_bridge = dataset['Alkaline_Phosphotase'].quantile(0.75) + (IQR*3)

print(lower_bridge)
print(upper_bridge)

# if value greater than upper bridge, we replace that value with upper_bridge value:
dataset.loc[dataset['Alkaline_Phosphotase'] >= upper_bridge, 'Alkaline_Phosphotase'] = upper_bridge


# In[27]:


# Lets compute the Interquantile range of Alamine_Aminotransferase feature to calculate the boundaries:
IQR = dataset.Alamine_Aminotransferase.quantile(0.75) - dataset.Alamine_Aminotransferase.quantile(0.25)

# Extreme outliers
lower_bridge = dataset['Alamine_Aminotransferase'].quantile(0.25) - (IQR*3)
upper_bridge = dataset['Alamine_Aminotransferase'].quantile(0.75) + (IQR*3)

print(lower_bridge)
print(upper_bridge)

# if value greater than upper bridge, we replace that value with upper_bridge value:
dataset.loc[dataset['Alamine_Aminotransferase'] >= upper_bridge, 'Alamine_Aminotransferase'] = upper_bridge


# In[28]:


# Lets compute the Interquantile range of Aspartate_Aminotransferase feature to calculate the boundaries:
IQR = dataset.Aspartate_Aminotransferase.quantile(0.75) - dataset.Aspartate_Aminotransferase.quantile(0.25)

# Extreme outliers
lower_bridge = dataset['Aspartate_Aminotransferase'].quantile(0.25) - (IQR*3)
upper_bridge = dataset['Aspartate_Aminotransferase'].quantile(0.75) + (IQR*3)

print(lower_bridge)
print(upper_bridge)

# if value greater than upper bridge, we replace that value with upper_bridge value:
dataset.loc[dataset['Aspartate_Aminotransferase'] >= upper_bridge, 'Aspartate_Aminotransferase'] = upper_bridge


# In[29]:


# Lets compute the Interquantile range of Albumin_and_Globulin_Ratio feature to calculate the boundaries
IQR = dataset.Albumin_and_Globulin_Ratio.quantile(0.75) - dataset.Albumin_and_Globulin_Ratio.quantile(0.25)

# Extreme outliers
lower_bridge = dataset['Albumin_and_Globulin_Ratio'].quantile(0.25) - (IQR*3)
upper_bridge = dataset['Albumin_and_Globulin_Ratio'].quantile(0.75) + (IQR*3)

print(lower_bridge)
print(upper_bridge)

# if value greater than upper bridge, we replace that value with upper_bridge value:
dataset.loc[dataset['Albumin_and_Globulin_Ratio'] >= upper_bridge, 'Albumin_and_Globulin_Ratio'] = upper_bridge


# In[30]:


# Top 5 records:
dataset.head()


# In[31]:


# Description after deal with outliers by IQR:
dataset.describe()


# In[32]:


# Independent and Dependent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[33]:


# top 5 records of Independent features:
X.head()


# In[34]:


# top 5 records of dependent features:
y.head()


# In[35]:


# SMOTE Technique:
from imblearn.combine import SMOTETomek
smote = SMOTETomek()
X_smote, y_smote = smote.fit_resample(X,y)


# In[36]:


# Counting before and after SMOTE:
from collections import Counter
print('Before SMOTE : ', Counter(y))
print('After SMOTE  : ', Counter(y_smote))


# In[37]:


# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote, test_size=0.3, random_state=33)


# In[38]:


print(X_train.shape)
print(X_test.shape)


# In[39]:


# Feature Importance :
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Apply SelectKBest Algorithm
ordered_rank_features=SelectKBest(score_func=chi2,k=9)
ordered_feature=ordered_rank_features.fit(X,y)

dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(X.columns)

features_rank=pd.concat([dfcolumns,dfscores],axis=1)

features_rank.columns=['Features','Score']
features_rank.nlargest(9, 'Score')


# #### There is no need of Standardization and Normalization of our dataset, as we using Ensemble Technique.

# In[40]:


# Importing Performance Metrics:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[41]:


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


# In[43]:


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


# In[44]:


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


# ####  RandomizedSearchCV

# In[45]:


# Importing RandomizedSearchCV:
from sklearn.model_selection import RandomizedSearchCV


# In[46]:


# Number of trees in random forest:
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)] 

# Number of features to consider at every split:
max_features = ['auto', 'sqrt','log2']

# Maximum number of levels in tree:
max_depth = [int(x) for x in np.linspace(100, 100,20)]

# Minimum number of samples required to split a node:
min_samples_split = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]

# Minimum number of samples required at each leaf node:
min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]


# In[47]:


# Create the random grid:
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)


# In[48]:


rf = RandomForestClassifier()
rf_randomcv = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose = 2,
                               random_state = 0, n_jobs = -1)

# fit the randomized model:
rf_randomcv.fit(X_train,y_train)


# In[49]:


# Best parameter of RandomizedSearchCV:
rf_randomcv.best_params_


# In[50]:


# Creating model using best parameter of RandomizedSearchCV:
RandomForest_RandomCV = RandomForestClassifier(criterion = 'entropy', n_estimators = 2000, max_depth = 100, max_features = 'log2',
                                               min_samples_split = 3, min_samples_leaf = 2)
RandomForest_RandomCV = RandomForest_RandomCV.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest_RandomCV.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #### GridSearchCV

# In[51]:


# Importing GridSearchCV:
from sklearn.model_selection import GridSearchCV


# In[52]:


# Best parameter:
rf_randomcv.best_params_


# In[53]:


param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'max_depth': [rf_randomcv.best_params_['max_depth']-50,
                  rf_randomcv.best_params_['max_depth'],
                 rf_randomcv.best_params_['max_depth']+50],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf']-1,
                         rf_randomcv.best_params_['min_samples_leaf'],
                         rf_randomcv.best_params_['min_samples_leaf']+1],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 50, 
                     rf_randomcv.best_params_['n_estimators'], 
                     rf_randomcv.best_params_['n_estimators'] + 50]
}

print(param_grid)


# In[54]:


# Fit the grid_search to the data:
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv=5 , n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train)


# In[55]:


# Best Parameter of GridSearchCV:
grid_search.best_params_


# In[56]:


# Creating model using best parameter of GridSearchCV:
RandomForest_gridCV = RandomForestClassifier(criterion='entropy', n_estimators=1950, max_depth=150, max_features='log2', 
                                             min_samples_split=2, min_samples_leaf=1)
RandomForest_gridCv = RandomForest_gridCV.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest_gridCV.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #### - We saw that after doing RandomizedSearchCV and GridSearchCV, Our accuracy, Precision, Recall, f1-Score doesn't increase. 

# In[57]:


# Creating a pickle file for the classifier

# import pickle
# filename = 'Liver.pkl'
# pickle.dump(RandomForestClassifier, open(filename, 'wb'))

import pickle
from sklearn.ensemble import RandomForestClassifier
with open('Liver2.pkl', 'wb') as f:
    pickle.dump(RandomForest_gridCv, f)

# In[ ]:




