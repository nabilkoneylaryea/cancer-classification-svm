import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer_data = datasets.load_breast_cancer()

#print(cancer_data.feature_names) # features or attributes
#print(cancer_data.target_names) # labels

x = cancer_data.data
y = cancer_data.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.2)

# print(x_train)
# print(y_train)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
