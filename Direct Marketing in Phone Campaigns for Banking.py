#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Introduction and important caveats

# This notebook addresses the challenges of classifying new customers based on previous customer data 
# regarding the purchase of term deposits for a bank. The challenge and data are avaiailable at Kaggle:
# https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets
# The author of the challenge requires that we cite the original article analyzing this dataset:
# S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing.
# Decision Support Systems, Elsevier, 62:22-31, June 2014

# According to the authors of the article, this dataset is real world data adapted for public use:
# "Real-world data were collected from a Portuguese marketing campaign 
# related with bank deposit subscription." (MORO, S. & CORTEZ, P. 2011, p. 1)
# We have not read the article or challenge submissions so our analysis is not skewed.

## Data cleaning, preparation and transformation strategies

# According to the authors of the challenge on Kaggle, the data is relatively clean and without missing values on the 18 columns
# However, we see the following problems with the data that must be addressed first:
# 1. Obtain the year variable, dates can be important to detect economic and business fluctuations
# 2. Convert 'unknown' and '-1' values to missing and decide for imputation, feature engineering or dropping
# 3. Convert string columns to boolean and categorical
# 4. Transform categorical collumns to dummie columns
# 5. Address class imbalance in label 'y' with resampling (therefore, we will not use the suggested test split)


### Importing Required Packages and Modules

## Importing Packages

import pandas as pd
import numpy as np
from calendar import month_abbr
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


# In[2]:


### Loading and Cleaning Data

## Loading data from local file downloaded from Kaggle challenge page and marking origin 

test = pd.read_csv('test.csv', sep=';')
train = pd.read_csv('train.csv', sep=';')

## Checking if Test records are contained in Train, since this is not clear

# Creating 'origin' collumn for identification after combining
test = test.assign(origin='test')
train = train.assign(origin='train')

# Combining Training and Test datasets
combined = pd.concat([train, test])

# Creating 'dup' column to identify any duplicates
combined['dup'] = combined.drop(['origin'], axis = 1).duplicated()

rows_dup = combined[(combined.dup == True)].shape[0]
print('Rows duplicated in combined dataset: %s' % (rows_dup))

uniq_rec = combined[(combined.origin == 'test') & (combined.dup == False)].shape[0]
print('Unique rows in test dataset: %s' % (uniq_rec))

# We can see that all records in 'test' are also in 'train', so we can procede with only train, which we will rename to df
# After droppinh the no longer needed origin collumn

df = train.drop(['origin'], axis=1)


# In[3]:


## Obtaining year variable
start_09 = df[df.month == 'jan'].index[1]
end_09 = df[df.month == 'dec'].index[-1]
end_08 = start_09 -1
start_10 = end_09 +1

df_08 = df.iloc[:end_08,:].assign(year=2008)
df_09 = df.iloc[start_09:end_09,:].assign(year=2009)
df_10 = df.iloc[start_10:,:].assign(year=2010)

df_year = pd.concat([df_08, df_09, df_10])

# Converting month to numeric
df_year['month'] = df_year.month.str.title()
abbr_to_num = {name: num for num, name in enumerate(month_abbr) if num}
df_year['month'] = df_year.month.map(abbr_to_num)


# In[4]:


## Addressing 'unknown' values in datasets

df_year.replace(to_replace='unknown', value=np.nan, inplace=True)

# Here we address the '-1' value in pdays collumn to create a new feature 'contacted in previous campaign'
df_year['contact_prev_camp'] = np.select([df_year.pdays == -1, df_year.pdays != -1], ['no', 'yes'])
df_year.pdays.replace(to_replace=-1, value=np.nan, inplace=True)

# Let´s investigate presence of missing values
print('df missing percentages: ')
print(df.isnull().mean())

# Considering close to 80% of records contain missing values for pdays and poutcome and 30% for contact,
# we will drop these features

redu = ['pdays', 'poutcome', 'contact']
df_redu = df_year.drop(columns=redu)


# In[5]:


## Converting yes/no variables to 1 and 0

# Eye-balling yes/no variables
print('First 5 rows of df_redu')
print(df_redu.head())

# Defining columns
yesno = ['default', 'housing', 'loan', 'y', 'contact_prev_camp']

# Replacing values
df_redu[yesno] = df_redu[yesno].replace(to_replace=['no', 'yes'], value=[0, 1])


# In[117]:


## Mising Data - education and job variables
# Investigating if education and job are MCAR for less than 5% of data

# Obtaining the modes of education and job in full dataset
mode_educ = str(df_redu['education'].mode()[0])
mode_job = str(df_redu['job'].mode()[0])

print('Education Most Frequent Value: %s' % (mode_educ))
print('Job Most Frequent Value: %s' % (mode_job))

# Checking conversion rate for mode categories and missing
conv_rate_sec_educ = round(df_redu[(df_redu.education == 'secondary') & (df_redu.y == 1)].shape[0] / df_redu[df_redu.education == 'secondary'].shape[0], 2)
print('Conversion Rate for Secondary Educ: %s' % (conv_rate_sec_educ))

conv_rate_missing_educ = round(df_redu[(df_redu.education.isnull()) & (df_redu.y == 1)].shape[0] / df_redu[df_redu.education.isnull()].shape[0], 2)
print('Conversion Rate for unknown Educ: %s' % (conv_rate_missing_educ))

conv_rate_bluec_job = round(df_redu[(df_redu.job == 'blue-collar') & (df_redu.y == 1)].shape[0] / df_redu[df_redu.job == 'blue-collar'].shape[0], 2)
print('Conversion Rate for Blue-Collar Job: %s' % (conv_rate_bluec_job))

conv_rate_missing_job = round(df_redu[(df_redu.job.isnull()) & (df_redu.y == 1)].shape[0] / df_redu[df_redu.job.isnull()].shape[0], 2)
print('Conversion Rate for unknown Job: %s' % (conv_rate_missing_job))

# As we can see, the differences in outcomes among these groups DO NOT WARRENT 100% the use of Mode Imputation.
# Therefore, we will drop the lines with missing values in these columns

df_clean = df_redu.dropna()
nrows_full = df_redu.shape[0]
nrows_clean = df_clean.shape[0]

print('Number of rows in full dataset: %s' % (nrows_full))
print('Number of rows in data dataset: %s' % (nrows_clean))

#Final checks for completeness of datasetsa

print('df_clean missing percentages')
print(df_clean.isnull().mean())

df_clean.head()


# In[118]:


## Preparing our Data for XGBoost - Encoding Categorical Variavles 

# Defining Categorical Collumns
cats = ['job', 'marital', 'education']

# Converting Strings to Category Variables
df_clean[cats] = df_clean[cats].astype('category')

# One-Hot Encoding of Categorical Variables
df_clean = pd.get_dummies(df_clean, columns=cats, prefix=['j', 'ma', 'e'], drop_first=True)

# Converting Boolean Variables
try:
    yesno.remove('y')
except:
    print('y already removed')

df_clean[yesno] = df_clean[yesno].astype('bool')


# In[126]:


## Investigating All Feature Variance and Covariance

df_norm = df_clean / df_clean.mean()
print('Full Feature Normalized Variance:')
print(df_norm.var())

# Only the 'year' variable that we've constructed has a variance with order of magnitude in the 100 millionth (8.913434e-08)
# but we will keep it for now since it may help model prediction

## Plotting Feature Correlations
import seaborn as sns

# Correlation Matrix
corr = df_clean.corr()

# Setting up Correlation Heatmap plot
print('Correlation Heatmaps: Full & Reduced Feature sets')
sns.set_theme(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# We see that ma_married is highly correlated with ma_single, as e_secondary is with e_tertiary. We will drop one of each.

redund = ['ma_single', 'e_secondary']

df_redu = df_clean.drop(redund, axis=1)

corr = df_redu.corr()

sns.set_theme(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Feature Correlation seems within acceptable bounds after removing redundant variables


# In[127]:


## Dealing with class imbalance

bal = round(df_redu.y.value_counts()[1] / df_redu.shape[0], 2)
print('Percent of positive class in sample: %s' % (bal))
print('Number of obs:', df_redu.shape[0])

# Upsampling minority class
min = df_redu[df_redu.y == 1]
maj = df_redu[df_redu.y == 0]
min_up = resample(min, replace=True, n_samples=maj.shape[0])
df_bal = pd.concat([maj, min_up])

# Checking Balance of New dataframe

bal = round(df_bal.y.value_counts()[1] / df_bal.shape[0], 2)
print('-----------------------------------------------')
print('Percent of positive class in sample: %s' % (bal))
print('Number of obs:', df_bal.shape[0])


# In[128]:


# Splitting data - 80% for training and tuning, 20% held out for performance validation

X_train, X_test, y_train, y_test = train_test_split(df_bal.drop('y', axis=1), df_bal['y'], test_size=0.2, random_state=123)


# In[129]:


## Training the Model
from sklearn.metrics import f1_score, roc_auc_score

# Instantiating Classifier
xgb_clf = xgb.XGBClassifier(objective = 'reg:logistic'
                            , n_estimators = 10
                            , seed = 100
                            , eval_metric=roc_auc_score)

# Fiitting Model to Training Data
xgb_clf.fit(X_train, y_train)


# In[130]:


## Plotting Feature Importances
print('Feature Importances in XGBoost Classifier: ')
print(pd.DataFrame(xgb_clf.feature_importances_, index=X_test.columns
                   , columns=['feature_importance']).sort_values(by='feature_importance', axis=0, ascending=False))

# We can see that year, housing and duration have the strongest feature importance for our simple model
# while contact_prev_camp, j_entrepreneur, j_student, j_technician have zero importance for model predictions


# In[131]:


### Making Predictions on the Test Set and Evaluating Model Performance
from sklearn.metrics import classification_report, roc_auc_score

y_pred_train = xgb_clf.predict(X_train)
y_pred = xgb_clf.predict(X_test)

## Evaluating Model Performance

# Computing Accutacy
print('Train Accuract Score:', accuracy_score(y_train, xgb_clf.predict(X_train)))
print('Test Accuracy Score:', accuracy_score(y_test, y_pred))

# Displaying Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=xgb_clf.classes_)

print(classification_report(y_train, y_pred_train))
print('-----------------------------------------------')
print(classification_report(y_test, y_pred))
print('ROC AUC score: %s' % (roc_auc_score(y_test, y_pred)))

# As we can see, the model is not overfitting the training data, which is positive


# In[132]:


## Hyperpamater Tuning

# Let´s see how much we can improve our baseline performance of 89% ROC AUC score
# Now using cross-validation, 5-fold for symetry with the validation hold-out proportion - This takes a few minutes

params_grid = {
    'n_estimators': [200, 500, 750, 900],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'booster': ['gbtree', 'gblinear'],
    'gamma': [0, 0.5, 1, 1.5, 2],
    'reg_alpha': [0, 0.1, 0.5, 1, 1.5, 2],
    'reg_lambda': [0, 0.1, 0.5, 1, 1.5, 2],
    'base_score': [0.001, 0.025, 0.05, 0.1],
    'max_depth' : [3, 6, 10, 15]
}

rs = RandomizedSearchCV(xgb.XGBClassifier(n_jobs=-1, eval_metric=roc_auc_score), params_grid, n_jobs=-1, cv=5, scoring='roc_auc')
rs.fit(X_train, y_train)

print('Best params:', rs.best_params_)
print('Best score:', rs.best_score_)

# As we start to allow more extreme hyperparameter values, our model starts overfitting
# at around 99% ROC AUC for the full training set and 95,8% of the test hold-out set


# In[133]:


## Evaluating Tuned Model Performance

y_train_pred = rs.predict(X_train)
y_pred = rs.predict(X_test)

print('Classification Report Training: ')
print(classification_report(y_train, y_train_pred))
print('Roc auc score training: ', roc_auc_score(y_train, y_train_pred))
print('-----------------------------------------------')
print('Classification Report Test Data: ')
print(classification_report(y_test, y_pred))
print('Roc auc score test', roc_auc_score(y_test, y_pred))


# In[134]:


best = rs.best_estimator_

## Plotting Feature Importances
print('Feature Importances in XGBoost Classifier: ')
print(pd.DataFrame(best.feature_importances_, index=X_test.columns
                   , columns=['feature_importance']).sort_values(by='feature_importance', axis=0, ascending=False))

# We can see that year, housing and duration continue to tbe the most important features in model predictions
# and only contact_prev_camp has zero importance


# In[135]:


## Hyperpamater Tuning - Part 2

# Let´s see if we can improve our tuned-model performance of 95.8% ROC AUC score on the hold-out test set
# This takes a few minutes

params_grid = {
    'n_estimators': [200, 500, 750, 900],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'booster': ['gbtree', 'gblinear'],
    'gamma': [0, 0.5, 1, 1.5, 2],
    'reg_alpha': [0, 0.1, 0.5, 1, 1.5, 2],
    'reg_lambda': [0, 0.1, 0.5, 1, 1.5, 2],
    'base_score': [0.001, 0.025, 0.05, 0.1],
    'max_depth' : [3, 6, 10, 15]
}

rs = RandomizedSearchCV(xgb.XGBClassifier(n_jobs=-1, eval_metric=roc_auc_score), params_grid, n_jobs=-1, cv=5, scoring='roc_auc')
rs.fit(X_train.drop('contact_prev_camp', axis=1), y_train)

print('Best params:', rs.best_params_)
print('Best score:', rs.best_score_)

# We don´t get better performance, but we get more overfitting
# at around 99.5% ROC AUC for the full training set and 95,79% of the test hold-out set


# In[136]:


## Evaluating Tuned Model Performance - Part 2

y_train_pred = rs.predict(X_train.drop('contact_prev_camp', axis=1))
y_pred = rs.predict(X_test.drop('contact_prev_camp', axis=1))

print('Classification Report Training: ')
print(classification_report(y_train, y_train_pred))
print('Roc auc score training: ', roc_auc_score(y_train, y_train_pred))
print('-----------------------------------------------')
print('Classification Report Test Data: ')
print(classification_report(y_test, y_pred))
print('Roc auc score test', roc_auc_score(y_test, y_pred))


# In[139]:


best = rs.best_estimator_

## Plotting Feature Importances
print('Feature Importances in XGBoost Classifier: ')
print(pd.DataFrame(best.feature_importances_, index=X_test.drop('contact_prev_camp', axis=1).columns
                   , columns=['feature_importance']).sort_values(by='feature_importance', axis=0, ascending=False))


# In[140]:


## We can conclude that a high perfomance classifier is achievable with the data provided.
# The steps we employed are as follows:

# 1. Cleaning and standardizing data, allowing the use of a wide range of tools
# 2. Feature Engineering, especially the construction of the 'year' variable which proved most important of all
# 3. Balancing the dataset with upsampling on the minority class, allowing us to follow through with tools for balanced data
# 4. Using a tree-based model out of the box like xgboost brings feature selection and a 89% ROC AUC score on the validation set.
# 5. Tuning hyperparameters with Random Search brings our validaction ROC AUC score to over 95%, with accuracy balanced between classes.

