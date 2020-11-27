# Income Prediction Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')

# Data Preparation

data = pd.read_csv('Income_prediction.csv')
data.head() # for viewing the initial rows

data.columns # returns out all column names
data.info()  #returns data type of each column and its count

data['workclass'].unique()
for col in data[['workclass','occupation','native.country']]:
    data[col] = data[col].replace('?',np.nan)

data.dropna(how = 'any', inplace = True)    

#Check for object values and inspecting labels
data['occupation'].unique()
data['education'].unique()
data['relationship'].unique()
data['workclass'].unique()
data['workclass'].value_counts()   

#Label Encoding
#converting the labels into numeric form

from sklearn.preprocessing import LabelEncoder

X1 = data[['occupation']]
lm = LabelEncoder()
a = ['occupation']
for i in np.arange(len(a)):
    X1[a[i]] = lm.fit_transform(X1[a[i]])
data['occupation'] = X1   

X2 = data[['education']]
lm = LabelEncoder()
b = ['education']
for i in np.arange(len(b)):
    X2[b[i]] = lm.fit_transform(X2[b[i]])
data['education'] = X2 

X3 = data[['workclass']]
lm = LabelEncoder()
a = ['workclass']
for i in np.arange(len(a)):
    X3[a[i]] = lm.fit_transform(X3[a[i]])

data['workclass'] = X3

X4 = data[['native.country']]
lm = LabelEncoder()
a = ['native.country']
for i in np.arange(len(a)):
    X4[a[i]] = lm.fit_transform(X4[a[i]])
data['native.country'] = X4

X5 = data[['marital.status']]
lm = LabelEncoder()
a = ['marital.status']
for i in np.arange(len(a)):
    X5[a[i]] = lm.fit_transform(X5[a[i]])
data['marital.status'] = X5

X6 = data[['relationship']]
lm = LabelEncoder()
a = ['relationship']
for i in np.arange(len(a)):
    X6[a[i]] = lm.fit_transform(X6[a[i]])
data['relationship'] = X6

inc = data[['income']]
lm = LabelEncoder()
a = ['income']
for i in np.arange(len(a)):
    inc[a[i]] = lm.fit_transform(inc[a[i]])
data['income'] = inc

print(data)

data.info() 

y = pd.DataFrame(data['income'])
data.income.value_counts(normalize = True)
data1 = data.drop('income',axis = 1).  # dropping data points that are not required

#Applyng dummy values and concatinating them to the main data values
sx = pd.get_dummies(data1['sex'])
rc = pd.get_dummies(data1['race'])

data1 = pd.concat([data1,sx,rc],axis=1)

data1 = data1.drop(['sex','race'],axis = 1)

#Training the Model
#Splitting dataset into train and test data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data1, y, test_size=0.2,random_state = 2)
print(x_test.shape)
print(y_test.shape)

# Trialing different algorithms

#1)Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
rft = RandomForestClassifier(n_estimators = 120,
                             criterion = 'entropy',
                             max_depth = 24,
                             max_features = 'auto',
                             bootstrap = False,
                             verbose = 2,
                             warm_start = True,
                             random_state = 2,
                             n_jobs = -1
                            )
rft.fit(x_train,y_train)
y_pred = rft.predict(x_test)

print('Accuracy score = ',accuracy_score(y_test,y_pred))
print('Precision score =', precision_score(y_test,y_pred, average = 'binary'))
print('Recall score =',recall_score(y_test,y_pred, average = 'binary'))
print('f1 score = ',f1_score(y_test,y_pred,average = 'binary'))
confusion_matrix(y_test,y_pred)

# 2) Logistic Regression

from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(class_weight = {0:0.39, 1:1})
logit.fit(x_train,y_train)
y_pred = logit.predict(x_test)

print('Accuracy score = ',accuracy_score(y_test,y_pred))
print('Precision score =', precision_score(y_test,y_pred))
print('Recall score =',recall_score(y_test,y_pred))
print('f1 score = ',f1_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

# 3) Decision Tree

from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint 
from sklearn.model_selection import RandomizedSearchCV 
  
param_dist = {"max_depth": [3, None], 
              "max_features": randint(1, 9), 
              "min_samples_leaf": randint(1, 9), 
              "criterion": ["gini", "entropy"]} 
dt_model = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(dt_model, param_dist, cv = 5) 
tree_cv.fit(x_train,y_train)
y_pred = tree_cv.predict(x_test)
   
# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_)) 
print("Best score is {}".format(tree_cv.best_score_)) 

print('Accuracy score = ',accuracy_score(y_test,y_pred))
print('Precision score =', precision_score(y_test,y_pred))
print('Recall score =',recall_score(y_test,y_pred))
print('f1 score = ',f1_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

# 4) KNN

from sklearn.neighbors import KNeighborsClassifier
metric_k = []
neighbors = np.arange(1,25)

# finding most probable k value ( nearest neighbours )
for k in neighbors:
    classifier = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    metric_k.append(acc)
    
plt.plot(neighbors,metric_k,'o-')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.grid()

classifier = KNeighborsClassifier(n_neighbors = 18, metric = 'minkowski', p = 2)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

print('Accuracy score = ',accuracy_score(y_test,y_pred))
print('Precision score =', precision_score(y_test,y_pred, average = 'binary'))
print('Recall score =',recall_score(y_test,y_pred, average = 'binary'))
print('f1 score = ',f1_score(y_test,y_pred,average = 'binary'))
confusion_matrix(y_test,y_pred)

# Conclusion - Best Deployed Model - Random Forest Classifier¶
#Accuracy score = 0.8604342781369136 | Precision score = 0.7582329317269076 | Recall score = 0.6356902356902356 | f1 score = 0.6915750915750916
