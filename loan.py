# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 23:17:35 2018

@author: Aneesh
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier




def prediction(X_test, clf_object):
 
	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred

def cal_accuracy(y_test, y_pred):
	 
	print("Confusion Matrix: ",
	confusion_matrix(y_test, y_pred))
	 
	print ("Accuracy : ",
	accuracy_score(y_test,y_pred)*100)
	 
	print("Report : ",
	classification_report(y_test, y_pred)) 


#print(data.info())

#print(data['lid'])
def cleandata(data1):
    data1.columns = ['lid','gen','married','dep','edu','sel','appinc','coinc','lamt','lamt_term','cred_his','prop_area','l_stat']    
    data1['cred_his'].fillna(randint(0,1),inplace = True)
    avg1 = data1['lamt_term'].mean()
    data1['lamt_term'].fillna(avg1 , inplace =True)
    avg2 = data1['lamt'].mean()
    data1['lamt'].fillna(int(avg2) , inplace = True)
    data1['sel'].fillna('Yes',inplace = True)
    data1['gen'].fillna('Male',inplace = True)
    data1['dep'].fillna('2',inplace =True)
    data1['married'].fillna(randint(0,1),inplace =True)
    data1['gen'] = pd.factorize(data1.gen)[0] # male = 0 female = 1
    data1['married'] = pd.factorize(data1.married)[0]# not married = 0 married = 1
    data1['edu'] = pd.factorize(data1.edu)[0] # 0 for graduate 1 for undergraduate
    data1['sel'] = pd.factorize(data1.sel)[0] #0 not self employ 1 for self employed
    data1['prop_area'] = pd.factorize(data1.prop_area)[0]# 0 for urban 1 for rural 2 for suburban
    data1['l_stat'] = pd.factorize(data1.l_stat)[0]# 0 for approved 1 for rejected
    data1['dep']= pd.factorize(data1.dep)[0]
    # normalizing data
    data1['appinc'] = data1['appinc']/data1['appinc'].max()
    data1['coinc'] = data1['coinc']/data1['coinc'].max()
    data1['lamt'] = data1['lamt']/data1['lamt'].max()
    data1['lamt_term'] = data1['lamt_term']/data1['appinc'].max()
    
    return(data1)
#df has the cleaned data
data = cleandata(pd.read_csv('train.csv'))
X = data[data.columns[1:-1]]
y= data[data.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clf = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=5, min_samples_leaf=8)
clf1 = LogisticRegression()
clf2 = GaussianNB()
clf3 = svm.SVC(gamma='scale')
clf4 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf4.fit(X_train,y_train)
y_pred_gini = prediction(X_test, clf4)
cal_accuracy(y_test, y_pred_gini)













