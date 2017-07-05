
# coding: utf-8

# # Enron Dataset Exploration
# ### by Mohamad Zeini Jahromi

# ## Dataset 

# In[2]:

import sys
import os
import pickle
# path = "C:\Users\Mo\Dropbox\DAND\P5\ud120-projects-master\Final_project"
# os.chdir(path)
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Number of entries  : ', len(data_dict)
print 'Number of features : ', len(data_dict.values()[0])
poi_tot = 0
for k, v in data_dict.items():
    if v['poi'] == True:
        poi_tot += 1
print 'Number of POI      : ', poi_tot
data_dict.items()[0]


# ## Features

# In[3]:

dic_nan = {}
for key, value in data_dict.items():
    for k, v in value.items():
        if k not in dic_nan.keys():
            dic_nan[k] = 0
        if v == 'NaN':
            dic_nan[k] += 1
for k, v in dic_nan.items(): 
    print("{: >25} {: >10}".format(*[k, v]))


# ## Remove outliers

# In[4]:

import pandas as pd
from numpy import mean
df = pd.DataFrame(data_dict)
df = df.convert_objects(convert_numeric=True)
df = df.transpose()
# df = df.fillna(0)
df.head()


# In[5]:

df.groupby(['poi']).describe().round()


# Visualize the outliers by plotting "salary" vs. "bonus" features. 

# In[6]:

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
import seaborn as sb

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

for k, v in data_dict.items():
     if (v['salary'] != 'NaN' and v['salary'] > 10**6) or (v['bonus'] != 'NaN' and v['bonus'] > 10**7): 
            print("{: >20} {: >15} {: >15} {: >15}".format(*[k, v['salary'], v['bonus'], str(v['poi'])]))


# Remove 'TOTAL' entry .

# In[7]:

data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# ## Features score
# 

# In[50]:

my_features = ['poi', 'salary', 'total_payments', 'bonus',
               'total_stock_value', 'shared_receipt_with_poi', 
               'long_term_incentive', 'exercised_stock_options', 'other', 
               'deferred_income', 'expenses', 'restricted_stock',
               'from_poi_to_this_person', 'from_this_person_to_poi', 
               'from_messages', 'to_messages']

my_dataset = data_dict
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# K-best features
from sklearn.feature_selection import SelectKBest
k_features = SelectKBest(k=10)
k_features.fit(features, labels)

k_list = zip(k_features.get_support(), my_features[1:], k_features.scores_)
k_list = sorted(k_list, key=lambda x: x[2], reverse=True)
for i in k_list: 
    print("{: >25} {: >25}".format(*[i[1], i[2]]))


# ## Create new features

# In[51]:

for k, v in my_dataset.items():
    from_poi_to_this_person = v["from_poi_to_this_person"]
    to_messages = v["to_messages"]
    
    fraction = 0. 
    if from_poi_to_this_person != 'NaN' and to_messages != 'NaN':
        fraction_from_poi = float(from_poi_to_this_person) / to_messages
    my_dataset[k]["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = v["from_this_person_to_poi"]
    from_messages = v["from_messages"]
    if from_this_person_to_poi != 'NaN' and from_messages != 'NaN':
        fraction_to_poi = float(from_this_person_to_poi) / from_messages
    my_dataset[k]["fraction_to_poi"] = fraction_to_poi

my_features = ['poi', 'salary', 'total_payments', 'bonus',
               'total_stock_value', 'shared_receipt_with_poi', 
               'long_term_incentive', 'exercised_stock_options', 'other', 
               'deferred_income', 'expenses', 'restricted_stock', 
               'fraction_from_poi', 'fraction_to_poi']

# K-best features
from sklearn.feature_selection import SelectKBest
k_features = SelectKBest(k=10)
k_features.fit(features, labels)

k_list = zip(k_features.get_support(), my_features[1:], k_features.scores_)
k_list = sorted(k_list, key=lambda x: x[2], reverse=True)
for i in k_list: 
    print("{: >25} {: >25}".format(*[i[1], i[2]]))


# ## Feature selection process

# In[39]:

from sklearn.feature_selection import SelectKBest
def find_kbest(grid_search, features, labels, parameters): 
    for i in range(2, 6):
        print "================ K best features, k = {0} ================".format(i)
        kb = SelectKBest(k = i)
        k_features = kb.fit_transform(features, labels)       
        k_list = zip(kb.get_support(), my_features[1:], kb.scores_)
        k_list = sorted(k_list, key=lambda x: x[2], reverse=True)
        for item in k_list: 
            if item[0] == True: print item[1:]        
        cross_val(grid_search, k_features, labels, parameters)


# In[40]:

from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
def doPCA(grid_search, features, labels, parameters):    
    for i in range(2, 6):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        print "========= Principal Components features, n = {0} =========".format(i)    
        pca = PCA(n_components = i)
        pca.fit(features_train)
        pca_features = pca.transform(features)
        cross_val(grid_search, pca_features, labels, parameters) 


# ## Cross validation

# The following shows Stratified Shuffle Split cross validation used in our study.

# In[61]:

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def cross_val(grid_search, features, labels, parameters):
    
    cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
    acc, precision, recall, fscore, support = [], [], [], [], []

    for train_indices, test_indices in cv:
        #make training and testing sets
        features_train= [features[ii] for ii in train_indices]
        features_test= [features[ii] for ii in test_indices]
        labels_train=[labels[ii] for ii in train_indices]
        labels_test=[labels[ii] for ii in test_indices]

        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        
        acc += [accuracy_score(predictions, labels_test)]
        report = precision_recall_fscore_support(labels_test, predictions)
        precision += [report[0][1]]
        recall += [report[1][1]]
        fscore += [report[2][1]]
        support += [report[3][1]]
    print '=================================='
    print 'Accuracy:', mean(acc)
    print 'Precision:', mean(precision)
    print 'Recall:', mean(recall)
    print 'Fscore:', mean(fscore)
    print 'Support:', mean(support)
    #print classification_report(labels_test, predictions)
    if len(parameters.keys()) != 0:
        print '========= Best Parameters ========='
        best_params = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print '%s=%r, ' % (param_name, best_params[param_name])


# ##  Trying four algorithms 

# ## Random Forest Classifier

# In[28]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
'''
parameters = {'n_estimators':[2,5,10], 'min_samples_split': [2,3,5], 
              'min_impurity_split' : [1e-7,1e-15,1e-20],'warm_start' : [True, False]}
clf = RandomForestClassifier()
grid_search = GridSearchCV(clf, parameters)
find_kbest(grid_search, features, labels, parameters)


# In[31]:

doPCA(grid_search, features, labels, parameters)
'''

# ## AdaBoost Classifier

# In[32]:

from sklearn.ensemble import AdaBoostClassifier
'''
parameters = {'n_estimators': [10, 20, 40],
               'algorithm': ['SAMME', 'SAMME.R'],
               'learning_rate': [.5, 1, 1.5]}

clf = AdaBoostClassifier()
grid_search = GridSearchCV(clf, parameters)
find_kbest(grid_search, features, labels, parameters)


# In[33]:

doPCA(grid_search, features, labels, parameters)
'''

# ## Decision Tree Classifier

# In[34]:

from sklearn import tree
'''
parameters = {'criterion': ['gini', 'entropy'],
               'min_samples_split': [2, 10, 20],
               'max_depth': [None, 2, 5, 10],
               'min_samples_leaf': [1, 5, 10],
               'max_leaf_nodes': [None, 5, 10, 20]}
clf = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(clf, parameters)
find_kbest(grid_search, features, labels, parameters)


# In[35]:

doPCA(grid_search, features, labels, parameters)
'''

# ## Naive Bayse

# In[62]:

from sklearn.naive_bayes import GaussianNB
'''
parameters = {}
clf = GaussianNB()
find_kbest(clf, features, labels, parameters )


# In[63]:

doPCA(grid_search, features, labels, parameters)
'''

# ## Testing top two classifiers 

# In[75]:
'''
from tester import dump_classifier_and_data
from tester import main

my_features = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
              'salary', 'deferred_income']
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, my_features)
main()
'''

# In[76]:
from tester import dump_classifier_and_data
from tester import main

my_features = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus']
clf = RandomForestClassifier(min_impurity_split=1e-07, min_samples_split=5,  
                             n_estimators=10, warm_start=False)
dump_classifier_and_data(clf, my_dataset, my_features)
main()

# ## Final results  

# **Random Forest Classifier** with the following K-best features (k = 3) and tuned parameters has higher Precision (0.60) and Recall (0.33) values and we recommend this classifier as our final predictive model.
# 
# ** Precision :**  0.38
# 
# ** Recall  :**  0.28 
# 
# ** Selected features :** 
# 
# * exercised_stock_options
# * total_stock_value
# * bonus
# 
# ** Optimum tuned parameters:** 
# 
# * n_estimators        : 10
# * min_samples_split   : 5 
# * min_impurity_split  : 1e-7
# * warm_start          : False
