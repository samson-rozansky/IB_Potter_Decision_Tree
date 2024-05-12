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
#,Wearing Masks
l ="Breathing Problem,Fever,Dry Cough,Sore throat,Running Nose,Headache,Fatigue ,Gastrointestinal ,Abroad travel,Contact with COVID Patient,Attended Large Gathering,Visited Public Exposed Places,Family working in Public Exposed Places".split(',')
print(len(l))
X = df[l]
Y = df['COVID-19']
#print(X)


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()

import matplotlib.pyplot as plt
#print(X_test)
depth,value = [],[]
for i in range(13,14):
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
    print(clf.tree_.max_depth,accuracy_score(y_test,y_pred)*100)
    tree.plot_tree(clf)
    plt.show()


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
labels = [0,1]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()
# plt.plot(depth,value) #X then Y
# plt.ylabel('Accuracy Score')
# plt.ylabel('Depth')


# print(accuracy_score(y_test,y_pred)*100)
# print(clf.tree_.max_depth)
#print("TEST "+str(clf.predict(convert_data([1,0,0,1,1,1,1,0,1,0,0,0,0]))))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 100) 
model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)
total = 0
print(total/len(y_pred)*100)
print(accuracy_score(y_test,y_pred)*100)




from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 100) 
model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)
total = 0
print(total/len(y_pred)*100)
print(accuracy_score(y_test,y_pred)*100)
# tree.plot_tree(model)
# plt.show()

from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
print(total/len(y_pred)*100)
print(accuracy_score(y_test,y_pred)*100)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier()

# Perform cross-validation
scores = cross_val_score(clf, X, Y, cv=5)  # cv=5 means 5-fold cross-validation

# Print the accuracy scores for each fold
print("Accuracy scores for each fold:", scores)

# Print the mean accuracy and standard deviation
print("Mean accuracy:", scores.mean())
print("Standard deviation of accuracy:", scores.std())


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Example with RandomForestClassifier
# Assuming you have features X_train and labels y_train
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Extract feature importances
importances = clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
print(df)

plt.figure()
plt.title("Feature importances")
df1 = pd.DataFrame()
print(list(importances))
df1['variables']=X_train.columns
df1['values']=list(importances)

df1 = df1.sort_values(by=['values'], ascending = False)
plt.bar(df1['variables'], df1['values'], color="r", align="center")
#plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.xticks(rotation=90)
plt.show()
print(X_train.columns)