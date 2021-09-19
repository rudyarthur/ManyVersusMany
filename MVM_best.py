from utils import _predict_binary
import numpy as np
from sklearn.base import clone
from MVM import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
	
class MVM_best:
	"""
	Many versus Many best classifier
	Apply N random MVM classifiers on a train test split of the training data.
	Find the split that maximises the balanced accuracy and use that for predictions.
	"""
		
	def __init__(self, estimator, N=10, soft=True):
		self.estimator = estimator;
		self.N = N;
		self.soft=soft
		
	def fit(self, X, y):
		self.class_labels = np.unique(y)
		self.nclass = len(self.class_labels);	
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
		best_clf = None
		best = -np.inf
		for i in range(self.N):
			clf = MVM( self.estimator, self.soft )
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			s = balanced_accuracy_score(y_test, y_pred)
			if s > best:
				best = s;
				best_clf = clf;

		splits = { k:(best_clf.estimators_[k][0], best_clf.estimators_[k][1]) for k in best_clf.estimators_ }
		self.clf = MVM( self.estimator, self.soft )
		self.clf.fit(X, y, use_split=splits )
			
	def predict(self, X):
		return self.clf.predict(X)
			

if __name__ == "__main__":
	from sklearn.datasets import load_iris
	from sklearn.datasets import load_digits
	from sklearn.datasets import load_wine
	
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.svm import SVC
	from sklearn.naive_bayes import GaussianNB
	from sklearn.model_selection import train_test_split

	from sklearn.metrics import matthews_corrcoef
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import balanced_accuracy_score


	def test(X, y, soft=False):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
		
		for ker, clf in { 
			"DecisionTree" : MVM_best( DecisionTreeClassifier(random_state=0), soft=soft ),
			#"Knn         " : MVM_best( KNeighborsClassifier(n_neighbors=3), soft=soft ),
			"GaussianNB  " : MVM_best( GaussianNB(), soft=soft ),
			"LinearSVM   " : MVM_best( SVC(kernel='linear', random_state=0, probability=True), soft=soft ),
			"RadialSVM   " : MVM_best( SVC(kernel='rbf', random_state=0, probability=True), soft=soft )
			}.items():
			
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			print(ker , np.sum(y_test == y_pred), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred) )

	print("Iris::")
	X, y = load_iris(return_X_y=True)
	test(X, y)
	print("Digits::")	
	X, y = load_digits(return_X_y=True)
	test(X, y)
	print("Wine::")	
	X, y = load_wine(return_X_y=True)
	test(X, y)	
