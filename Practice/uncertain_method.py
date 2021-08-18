#%%
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#%%
data = pd.read_csv("./DataSets/iris.csv", delimiter=',')
print(data.head(10))
#%%
# describe stats summary of data
print("\n\nDescribing data")
print(data.describe())
#%%
# correlation matrix
print("\n\nCorreational matrix:")
print(data.corr(method='pearson'))
#%%
X = data.drop(["Id","Species"], axis=1)
print(X.head())
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
#%%
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
#%%
Xx = np.asarray([[3.5, 2.5, 1.4, 0.5],[5.4, 3.5, 1.4, 2.3]])
print(Xx.shape)
#%%

Y_pred = clf.predict(Xx)
print(Y_pred)