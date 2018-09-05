__author__ = 'Saurabh'
import numpy as np
from sklearn import preprocessing, cross_validation,neighbors,svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.csv')
df.replace('?',-9999,inplace=True)

#dropping id as it cant be include for features and labels, outliers
df.drop(['id'],1,inplace=True)

#everything except class can be features and label would be class
X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

#shuffling the data and training and testing chunckes, test size is 20%
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

#KNN
clf=svm.SVC()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)

example_measures=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures=example_measures.reshape(len(example_measures),-1)
prediction=clf.predict(example_measures)
print prediction