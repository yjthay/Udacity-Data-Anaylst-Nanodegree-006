#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle
import pandas as pd
import sklearn
os.chdir('C:/Users/YJ/Documents/1) Learning/Udacity - Data Analyst/Machine Learning/ud120-projects-master/final_project')
#os.chdir("C:/ud120-projects-master/final_project")

from tester import * 
from poi_Pipelines import *
from final_project_scripts import *

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary'] 

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Creating a list with all the features that are available to us
for i in data_dict:
    for j in data_dict[i]:
        if j not in features_list:
            features_list.append(j)

# 1)  Removing email_address from the features list given that it should not  
#     have any predictive power
# 2)  After removing the email address in the data, we look at the data in a 
#     DataFrame using the Pandas framework.
# 3)  Replacing the "NaN" with np.nan so as to ensure that the statistics  
#     generated by the describe function is sensible
# 4)  Looking at the data, we noticed that it is extremely varied where the  
#     data for emails are in thousands whilst bonus etc are in millions
# 5)  Lastly remove, 'THE TRAVEL AGENCY IN THE PARK' from the dataset as it 
#     does not have any meaningful data and seems to be a data entry error.
features_list.remove('email_address')
data = pd.DataFrame.from_dict(data_dict).transpose()
data.drop('THE TRAVEL AGENCY IN THE PARK')
print "Number of NaN in each feature"
print((data=="NaN").describe())
data = data.replace("NaN",np.nan)
print "We note that we have a total of "+ str(len(features_list)) + " features"
print(data.describe())
### Task 2: Remove outliers

# Removing the obvious error of 'TOTAL' and upon manual inspection of the data,
# I noticed that there are 2 data points that seems to be mis-entered.  
# I created a function correct_records to correct these 2 data points.
data = data.drop('TOTAL')
data = correct_records(data)

# Formatting the data, choosing features of interest, set nan as zero and 
# splitting the data into labels and features
data = featureFormatpd(data, features_list, zero_NaN=True)
labels, features = targetFeatureSplitpd(data,'poi')

### Task 3: Create new feature(s)

# We will look to scale the data then look to add more features to the data set 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = nparray_to_dataframe(scaler.fit_transform(features), \
                    features.index, features.columns)

scaled_features['frac_from_poi']= data['from_poi_to_this_person']/ \
                                    data['from_messages']
scaled_features['frac_to_poi'] = data['from_this_person_to_poi']/ \
                                    data['to_messages']
scaled_features['net_worth'] = data['salary'] + \
                                data['bonus'] + \
                                data['total_stock_value']
features_list.append('frac_from_poi')
features_list.append('frac_to_poi')
features_list.append('net_worth')

data = labels.to_frame().merge(scaled_features, left_index=True, right_index=True)
data = featureFormatpd(data, features_list, zero_NaN=True)
labels, features = targetFeatureSplitpd(data,'poi')

# Next we prepared a pipeline that chooses the optimal number of features to
# use in a simple LinearSVC
select_k = select_features_svc(features,labels).best_params_


from sklearn.feature_selection import SelectKBest, chi2, f_classif
select_chi2 = SelectKBest(score_func = chi2, k = select_k['select__k'])
select_f = SelectKBest(score_func = f_classif, k = select_k['select__k'])
select_chi2.fit(features, labels)
select_f.fit(features, labels)

print features_score(select_f,features).sort_values(by='scores')
print features_score(select_chi2,features).sort_values(by='scores')

#####################################################
features = transform(select_chi2,features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import precision_score, recall_score

svc = svc_tune(features_train,labels_train,tune = False)
gaussian = gaussian_tune(features_train,labels_train,tune = False)
log = log_tune(features_train,labels_train,tune = False)
rfc = rfc_tune(features_train,labels_train,tune = False)
ada = ada_tune(features_train,labels_train,tune = False)

print "\n"
print "SVC (untuned) - "
print_score(svc, features_test, labels_test)
print "\n"
print "Gaussian (untuned) - "
print_score(gaussian, features_test, labels_test)
print "\n"
print "Log (untuned) - "
print_score(log, features_test, labels_test)
print "\n"
print "Random Forest (untuned) - "
print_score(rfc, features_test, labels_test)
print "\n"
print "Adaboost (untuned) - "
print_score(ada, features_test, labels_test)
print "\n"

svc = svc_tune(features_train,labels_train,tune = True)
gaussian = gaussian_tune(features_train,labels_train,tune = True)
log = log_tune(features_train,labels_train,tune = True)
rfc = rfc_tune(features_train,labels_train,tune = True)
ada = ada_tune(features_train,labels_train,tune = True)

print "\n"
print "SVC (tuned) - "
print_score(svc, features_test, labels_test)
print "\n"
print "Gaussian (tuned) - "
print_score(gaussian, features_test, labels_test)
print "\n"
print "Log (tuned) - "
print_score(log, features_test, labels_test)
print "\n"
print "Random Forest (tuned) - "
print_score(rfc, features_test, labels_test)
print "\n"
print "Adaboost (tuned) - "
print_score(ada, features_test, labels_test)
print "\n"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from sklearn.ensemble import RandomForestClassifier
clf =  RandomForestClassifier(criterion = 'gini',n_estimators = 5)

my_dataset = labels.to_frame().merge(features, left_index=True, right_index=True)
features_list = list(my_dataset.columns)

my_dataset = my_dataset.transpose().to_dict()

dump_classifier_and_data(clf, my_dataset, features_list)

#financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

#email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

#POI label: [‘poi’] (boolean, represented as integer)