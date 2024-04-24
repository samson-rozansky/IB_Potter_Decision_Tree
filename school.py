import pickle
import pandas as pd

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import math
import matplotlib as plt

val = False

# def convert_data(arr):
#     dic = {}
#     for i in range(len(l)):
#         dic[l[i]] = arr[i]

#     return pd.DataFrame(dic,index=[0])

df= pd.read_csv("https://raw.githubusercontent.com/Percy-Potter/Covid-Data/main/Covid%20Dataset.csv")



#print(df)

from sklearn import tree
df=df.replace("Yes", 1)
df = df.replace("No", 0)

#print(df)
l ="Breathing Problem,Fever,Dry Cough,Sore throat,Running Nose,Headache,Fatigue ,Gastrointestinal ,Abroad travel,Contact with COVID Patient,Attended Large Gathering,Visited Public Exposed Places,Family working in Public Exposed Places,Wearing Masks".split(',')
X = df[l]
Y = df['COVID-19']
#print(X)




import matplotlib.pyplot as plt
#print(X_test)
depth,value = [],[]
for i in range(1,14):
    X_train, X_test, y_train, y_test = train_test_split(  
        X, Y, test_size=0.25, random_state = 100) 
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    total = 0
    print(total/len(y_pred)*100)
    print(accuracy_score(y_test,y_pred)*100,clf.tree_.max_depth)
    depth.append(clf.tree_.max_depth)
    value.append(accuracy_score(y_test,y_pred)*100)
    tree.plot_tree(clf)
    plt.show()


# plt.plot(depth,value) #X then Y
# plt.ylabel('Accuracy Score')
# plt.ylabel('Depth')


# print(accuracy_score(y_test,y_pred)*100)
# print(clf.tree_.max_depth)
#print("TEST "+str(clf.predict(convert_data([1,0,0,1,1,1,1,0,1,0,0,0,0]))))
