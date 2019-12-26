import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
data = pd.read_csv('seeds_dataset.csv')
# pisakhan data dan label
X = data.iloc[:, :-1].values
y = data.iloc[:, len(data.columns)-1].values

# split trining and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


################################
#### model, ambil salah satu ###
# SVM
from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X_train, y_train)


# KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
### end of model #####
######################


# hasil prediksi
hasil_prediksi = model.predict(X_test)


# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, hasil_prediksi)

