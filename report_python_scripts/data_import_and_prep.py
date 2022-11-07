# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:41:08 2021

@author: User
"""

import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
%matplotlib inline
import matplotlib.pyplot as plt

hrate_df = pd.read_csv("data_import/freelancer_tgr.csv", sep = ";", decimal = ",")


#Gender
hrate_df['Rate_Std'].mean()
hrate_df['Rate_Std'].min()
hrate_df['Rate_Std'].max()

hrate_df = hrate_df.loc[(hrate_df['Rate_Std'] < 350) & (hrate_df['Rate_Std'] >= 15)]

hrate_df.groupby('Gender').agg({'Rate_Std': ['mean', 'count']})

hrate_df['Gender'] = hrate_df['Gender'].replace('andy', 'unknown')
hrate_df['Gender'] = hrate_df['Gender'].replace('mostly_female', 'female')
hrate_df['Gender'] = hrate_df['Gender'].replace('mostly_male', 'male')
hrate_df['Gender'] = hrate_df['Gender'].replace('unknown', np.NaN)

hrate_df.groupby('Gender').agg({'Rate_Std': ['mean', 'count']})
d = {'female': True, 'male': False}
hrate_df['Female'] = hrate_df['Gender'].map(d)
hrate_df = hrate_df.drop(columns = 'Gender')

#JobTitle
all_titles = [x for x in set(hrate_df['Title']) if "," not in x]
all_titles = [x.replace('[\'', '') for x in all_titles]
all_titles = [x.replace('\']', '') for x in all_titles]



def find_title(title):
    for jp in hrate_df['Title']:
        if title in jp:
            yield True
        else:
            yield False

jobtitles = [list(find_title(at)) for at in all_titles]      
jobtitles = pd.DataFrame(jobtitles).transpose()
jobtitles.columns = all_titles

jobtitles.sum()
# blockchain, sharepoint-develop and ruby have no enough obs
# also drop UNKNOWN
jobtitles = jobtitles.drop(columns=['blockchain', 'sharepoint-develop', 'ruby', 'UNKNOWN'])

#Analytics Dataframe

analytics_df = pd.concat(
    [hrate_df[['ref_id', 'Female', 'Rate_Std']].reset_index(drop=True), 
     jobtitles.reset_index(drop=True)], axis=1)


analytics_df.head()
analytics_df['Rate_Std'].describe()


# Split

analytics_df_v1 = analytics_df.loc[analytics_df['Female'].isna() == False]
xTrain, xTest, yTrain, yTest = train_test_split(analytics_df_v1.drop(columns=['Rate_Std', 'ref_id']), #pred 
                                                analytics_df_v1['Rate_Std'], # regressand
                                                test_size=0.20, # 20% of the samples for testing
                                                random_state=42) 

linModel = linear_model.LinearRegression()
linModel.fit(X=xTrain.values, y=yTrain.values)


# Test the model
y_predLin = linModel.predict(xTest.values)
pd.DataFrame({'Actual': yTest, 'Predicted': y_predLin})


print('Mean Absolute Error:', mean_absolute_error(yTest.values, y_predLin))
mseLin = mean_squared_error(yTest.values, y_predLin)
print('The Mean Squared Error = {0:.3f}'.format(mseLin))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(yTest.values, y_predLin)))
r2_Lin = r2_score(yTest.values, y_predLin)
print('Goodness of Fit / R-Squared: {0:.3f}'.format(r2_Lin))

cv_score = cross_val_score(linModel, 
                analytics_df_v1.drop(columns=['Rate_Std', 'ref_id']), 
                analytics_df_v1['Rate_Std'], 
                cv=10)
cv_score.mean()

# The model's coefficients
coeff_df = pd.DataFrame(linModel.coef_, xTrain.columns, columns=['Coefficient'])
coeff_df


# import skills
pd.set_option('display.float_format', lambda x: '%.3f' % x)

skills = pd.read_csv("data_import/freelancer_skills.csv", sep = ";", decimal = ",")

skills.head()
skills = skills.rename(columns={"Unnamed: 0": "ref_id"})
skills = skills.fillna(0)

skillsum = skills.sum()
del skillsum['ref_id']

skillsum.describe()
skillsum.hist(bins=[0, 10, 20, 30, 40, 50, 100])
skill_selector = list(skillsum[skillsum > 10].index)
skills_selected = skills[skill_selector]

skills_selected


# supervised dimensionality reduction
skillcodes = list(skills.columns) 

main_cat = []

for sc in skillcodes:
    if '_' not in sc:
        main_cat.append(sc)

def calc_sum_skillcat(skillcode):
    sel_cols = [col for col in skills.columns if skillcode + '_' in col]
    sel_cols.append('B1')
    return skills[sel_cols].sum(axis=1) / len(sel_cols)
    
sum_skillcat = [calc_sum_skillcat(sc) for sc in main_cat]
sum_skillcat = pd.concat(sum_skillcat, axis=1, keys = main_cat)
sum_skillcat = pd.concat([skills['ref_id'].reset_index(drop=True), sum_skillcat], axis=1)

analytics_df_v2 = pd.merge(analytics_df, sum_skillcat, 'left')
analytics_df_v3 = analytics_df_v2.loc[analytics_df_v2['Female'].isna() == False]

xTrain, xTest, yTrain, yTest = train_test_split(analytics_df_v3.drop(columns=['Rate_Std', 'ref_id', 'Female']), #pred 
                                                analytics_df_v3['Rate_Std'], # regressand
                                                test_size=0.20, # 20% of the samples for testing
                                                random_state=42) 

linModel = linear_model.LinearRegression()
linModel.fit(X=xTrain.values, y=yTrain.values)


# Test the model
y_predLin = linModel.predict(xTest.values)
pd.DataFrame({'Actual': yTest, 'Predicted': y_predLin})


print('Mean Absolute Error:', mean_absolute_error(yTest.values, y_predLin))
mseLin = mean_squared_error(yTest.values, y_predLin)
print('The Mean Squared Error = {0:.3f}'.format(mseLin))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(yTest.values, y_predLin)))
r2_Lin = r2_score(yTest.values, y_predLin)
print('Goodness of Fit / R-Squared: {0:.3f}'.format(r2_Lin))

cv_score = cross_val_score(linModel, 
                analytics_df_v3.drop(columns=['Rate_Std', 'ref_id', 'Female']), 
                analytics_df_v3['Rate_Std'],
                scoring="r2",
                cv=10)
cv_score.mean()




def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))

y = analytics_df_v2['Rate_Std']
X = analytics_df_v2.drop(columns=['Rate_Std', 'ref_id', 'Female'])

# creating a range of degrees between 0 and 20
degree = np.arange(1, 3)

# computing training and validation scores for each polynomial fit
# validation_curve() parameters
# estimator, traing vector, target vector, name of the parameter to be varied, 
# parameter to be varied, cross-validation split
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree', degree, cv=3)

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');               
                         
                         
                         
# unsupervised dimensionality reduction
from sklearn import  cluster
X = skills.drop(columns=['ref_id'])
X.shape

agglo = cluster.FeatureAgglomeration(n_clusters=32)
agglo.fit(X)
X_reduced = agglo.transform(X)
X_reduced.shape


# Elastic net (Lasso and ridge regression)







