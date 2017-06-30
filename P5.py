
# coding: utf-8

# # Enron Dataset Exploration
# ### by Mohamad Zeini Jahromi
# ## Introduction
# Enron Corporation was an American energy, commodities, and services company based in Houston, Texas. Before its bankruptcy on December 2, 2001, Enron employed approximately 20,000 staff and was one of the world's major electricity, natural gas, communications and pulp and paper companies, with claimed revenues of nearly $101 billion during 2000. Fortune named Enron "America's Most Innovative Company" for six consecutive years.
# 
# At the end of 2001, it was revealed that its reported financial condition was sustained by institutionalized, systematic, and creatively planned accounting fraud, known since as the Enron scandal. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. 
# 
# The objective of this project is to come up with a predictive model for identifying employees who have committed fraud ("Person of Interest" or POI). I will explore the Enron email and financial dataset and test different classifiers to find the most accurate one in terms of identifying POI label. 

# ## Dataset 
# The dataset contained 146 records with 21 financial and email features. POI label is a Boolean label (True or False) and shows whether a person committed fraud or not. 18 out of 146 records were labeled as a "Person Of Interest" (POI). The following shows entries for the first person, 'METTS MARK'.

# In[2]:

import sys
import os
import pickle
path = "C:\Users\Mo\Dropbox\DAND\P5\ud120-projects-master\Final_project"
os.chdir(path)
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
# Many of features have missing values and assigned 'NaN' as the value entries. The following list shows which features have the most 'NaN' values. I will not include the features with more than 100 'NaN's in the final features list.

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
# The following table represents the statistical summary of our dataset. First we created a dataframe (first table) and grouped all the entries by 'poi' status and exclude all the 'NaN' values (second table). The significant differences between the average and maximum values, shows where we should look for possible outliers.

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


# Let's visualize the outliers by plotting "salary" vs. "bonus" features. Also we will print out persons with 'salary' above 1 million dollars and 'bonus' above 10 million dollars.

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


# Obviously 'TOTAL' entry is an extreme outlier and is not a real person and we will remove it from the dataset. Jefferey Skilling, Kenneth Lay are notable guys related to the fraud case. Next, we replot the data points for further investigation.

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


# ## Create new features
# 
# Before we create and engineer new features, let's look at our features list along with their scores. I used 'SelectKBest' to select top 10 most effective features.

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


# Scaling the 'from_this_person_to_poi' and 'from_poi_to_this_person' by the total number of emails sent and received, respectively, might help us identify those have low amounts of email activity overall, but a high percentage of email activity with POI's. 
# 
# In addition to our initial feature list, I created two new features based on number of emails 'sent to' or 'received from' POI features as follows:
# * fraction_from_poi
# * fraction_to_poi
# 
# The following list shows our new features list along with their respective score. Although our two new features have higher scores than previous ones but they are still in lower part of our features list and might not be included in the final feature set. Also, I want to point out that 'shared_receipt_with_poi' feature have higher score than our new features and represents stronger relationship.

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
# 
# I used two feature selection process, select K-Best and PCA, to select the best features. Select k-best removes all but the k highest scoring features and PCA is a process of transforming the data and projecting it to a lower dimensional space. We iterate between different number of K-Best and PCA to find the optimum number of features. The following are functions to perform the iteration process.

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


# ##  Selection of algorithms 
# 
# I will use four algorithms along with the parameter tuning process to find the most accurate and predictive model. The algorithms used are as follows:
# 
# * Random Forest Classifier
# * Decision Tree Classifier
# * AdaBoost Classifier
# * Gaussian Naive Bayse Classifier
# 
# ##  Parameters tuning  
# 
# In addition to the feature selection process by K-Best and PCA, the grid search function has been used to assign different number of parameters and choose the best classifier to maximize the precision and recall and the overall accuracy.
# 
# Grid search function construct a grid of all the combinations of parameters, tries each combination, and then reports back the best combination for specific algorithm.
# 
# Parameters tuning refers to the adjustment of the algorithm when training, in order to improve the fit on the test set. Parameter can influence the outcome of the learning process, the more tuned the parameters, the more biased the algorithm will be to the training data and test harness. The strategy can be effective but it can also lead to more fragile models and overfit the test harness but don't perform well in practice. For every algorithms, I tried to tune couple of effective paremeters.
# 
# ## Cross validation
# 
# The purpose of cross validation is to test the model multiple times and make balance between model bias and variance. Overfitting, occurs when we have a high variance in our model and is one of the most challenging things to avoid from in machine learning. 
# 
# Cross validation enable us to first fit a model on a portion of our data set (train) and then try the model on the remainder of the dataset (test) and calculate the overall accuracy. 
# 
# I used Stratified Shuffle Split cross validation with the test size of 30% to evaluate prediction accuracy of different classifiers. This algorithm essentially creates multiple train and test datasets out of our dataset and calculate the overall performance.
# 
# We are using "recall" and "precision" parameters to evaluate our classifier's prediction performance. Recall is the ability of classifier to identify all the positive values while precision is its ability to not falsely assign positive values to those values that are actually negative. Using these two we can understand how accurate our model is in identifying the POI persons within the entire dataset.
# 
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


# ## Random Forest Classifier
# For start, we try Random Forest Classifier along with the parameter tuning function (Grid Search). We iterate through two loops of features selection (K-Best and PCA) to find the best combination of features and algorithm's parameters.

# In[28]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

parameters = {'n_estimators':[2,5,10], 'min_samples_split': [2,3,5], 
              'min_impurity_split' : [1e-7,1e-15,1e-20],'warm_start' : [True, False]}
clf = RandomForestClassifier()
grid_search = GridSearchCV(clf, parameters)
find_kbest(grid_search, features, labels, parameters)


# In[31]:

doPCA(grid_search, features, labels, parameters)


# ## AdaBoost Classifier
# AdaBoost Classifier is the second classifier we are trying. Here again the parameter tuning function (Grid Search) and two loops of features selection (K-Best and PCA) has been used to find the best combination of features and algorithm's parameters.

# In[32]:

from sklearn.ensemble import AdaBoostClassifier
parameters = {'n_estimators': [10, 20, 40],
               'algorithm': ['SAMME', 'SAMME.R'],
               'learning_rate': [.5, 1, 1.5]}

clf = AdaBoostClassifier()
grid_search = GridSearchCV(clf, parameters)
find_kbest(grid_search, features, labels, parameters)


# In[33]:

doPCA(grid_search, features, labels, parameters)


# ## Decision Tree Classifier
# Decision Tree Classifier is the third classifier we are testing on our dataset. The parameter tuning function (Grid Search) and two loops of features selection (K-Best and PCA) are being used to find the best combination of features and algorithm's parameters.

# In[34]:

from sklearn import tree
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


# ## Naive Bayse
# Finally, we are trying Gaussian Naive Bayse on our dataset. Since we don't have many options for parameter tuning, we used the default values. Two loops of features selection (K-Best and PCA) are being used to find the best combination of features.

# In[62]:

from sklearn.naive_bayes import GaussianNB
parameters = {}
clf = GaussianNB()
find_kbest(clf, features, labels, parameters )


# In[63]:

doPCA(grid_search, features, labels, parameters)


# ## Discussion 
# In this project, we have evaluated and tested four different algorithms on our dataset. We applied parameters tuning function and iterate through two features selection loops (K-Best and PCA) to find the most predictive and accurate model. 
# 
# The precision and recall were used as the evaluation metrics. Precision is how often our class prediction (POI vs. non-POI) is right when we guess that class. Recall is how often we guess the class (POI vs. non-POI) when the class actually occurred. Due to the nature of the dataset, it is more important to make sure we don't miss any POI's and here accuracy is not a good measurement as even if non-POI are all flagged, the accuracy score will yield that the model is a success.
# 
# In our cross validation algorithm, I used 100 iterations to calculate the precision and recall values and I reported the means of these values at the end. On the other hand, the cross validation provided in 'tester.py' file uses StratifiedShuffleSplit method with 1000 folds and also uses the cumulative predictions and test results to calculate the precision and recall parameters and therefore it returns even better evaluation metrics.
# 
# For these reason, I picked the top two classifiers and checked the evaluation metrics using provided 'tester.py' file to see which one is better. 
# 
# The top two classifiers with tuned parameters and selected features are as follows:
# 
# ### Gaussian Naive Bayse with K-best features (k = 5) 
# 
# ** Precision :**  0.47
# 
# ** Recall  :**  0.37 
# 
# ** Selected features :** 
# 
# * exercised_stock_options
# * total_stock_value
# * bonus
# * salary
# * deferred_income
# 
# ### Random Forest Classifier with K-best features (k = 3) 
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
# 
# After saving our dataset, features list and classifier we ran the provided 'tester.py' to see the final evaluation metrics.

# In[75]:

from tester import dump_classifier_and_data
my_features = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
              'salary', 'deferred_income']
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, my_features)
get_ipython().magic(u'run tester.py')


# In[76]:

from tester import dump_classifier_and_data
my_features = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus']
clf = RandomForestClassifier(min_impurity_split=1e-07, min_samples_split=5,  
                             n_estimators=10, warm_start=False)
dump_classifier_and_data(clf, my_dataset, my_features)
get_ipython().magic(u'run tester.py')


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
