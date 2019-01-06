import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


plt.interactive(False)
pd.set_option('display.max_columns', None)


#imports our data set, removing columns that we don't care about and converting our factors into category
# data types for future analysis
data = pd.read_csv("D:\\breast-cancer-wisconsin-data\data.csv")


data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
labels = data['diagnosis']
data.drop(['diagnosis'], axis=1, inplace=True)


#Standardize the data
scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)

data['diagnosis'] = labels

#Get's a 2D plot of all the correlations in the data
corr = data.corr()
corr = corr.round(2)
plt.figure()
sns.heatmap(corr, cbar = True,  square = True, annot=True,
           cmap= 'coolwarm')

#uncomment to show the correlations
#plt.show()

#25% of the data is put in a test set

#splits into train and test sets
train_X, test_X = train_test_split(data, test_size = 0.25)


train_y = train_X.diagnosis# This is output of our train data

test_y = test_X.diagnosis   #output value of test dat

#removed the output variable column
train_X.drop(['diagnosis'], axis=1, inplace=True)
test_X.drop(['diagnosis'], axis=1, inplace=True)



#Define the model that we want to use
clf = svm.SVC(verbose=False, tol=1e-5, gamma='scale')

# specify hyperparameters to cycle through and find the best one
param_dist = {"C": np.linspace(0.01, 20, 50)}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, verbose= 0, iid= True)

#this applies the model with the best C value
random_search.fit(train_X, train_y)

# uncomment this line if you want to see the report for the best C values
#report(random_search.cv_results_)

#This computers the cross validation score to help determine if the model fits the data well and to spot if it is
#overfitted for the training data.

scores = cross_val_score(random_search, test_X, test_y, cv=5)
print('\n')

scoremean = round(scores.mean(), 2)
scorestd = (round(scores.std(), 2))

print("Accuracy: %0.2f (+/- %0.2f)" % (scoremean, scorestd))
