__author__ = 'Saurabh'
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import  warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

##dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
##new_features=[5,7]
accuracies=[]
#25 times testing the block
for i in range(25):
    def k_nearest_neighnors(data,predict,k=3):
        #print(predict)
        if(len(data)>=k):
            warnings.warn('K is set to value less than total voting groups')
        distances=[]
        for group in data:
            #print(group)
            #print('*')
            for features in data[group]:
                euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
                distances.append([euclidean_distance,group])
                #print(distances)
        votes=[i[1] for i in sorted(distances)[:k]]
        #print (votes)
        vote_result = Counter(votes).most_common(1)[0][0]
        return vote_result

    df = pd.read_csv('breast-cancer-wisconsin.data.csv')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    full_data=df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size=0.2
    train_set={2:[],4:[]}
    test_set={2:[],4:[]}
    #80%  train data and 20% test data
    train_data=full_data[:-int(test_size*len(full_data))]
    test_data=full_data[-int(test_size*len(full_data)):]

    #populating training set
    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    #print(test_set)

    correct=0.0
    total=0.0

    #testing the data on training set
    for group in test_set:
        for data in test_set[group]:
            #print(data)
            vote=k_nearest_neighnors(train_set,data,k=5)
            if group==vote:
                correct+=1
                #print correct
            total += 1
            #print total
    accuracy=correct/total
    #print('accuracy:',accuracy)
    accuracies.append(accuracy)
print('Accuracy:',sum(accuracies)/len(accuracies))