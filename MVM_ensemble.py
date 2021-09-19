from utils import _predict_binary
import numpy as np
from sklearn.base import clone
from MVM import *

class MVM_ensemble:
	"""
	Many versus Many ensemble classifier
	Apply N random MVM classifiers 
	Hard prediction = classifiers vote
	Soft prediction = add confidences, choose max
	"""
		
	def __init__(self, estimator, N=10, soft=True):
		self.estimator = estimator;
		self.N = N;
		self.soft=soft
		
	def fit(self, X, y):
		self.ensemble = []
		self.class_labels = np.unique(y)
		self.nclass = len(self.class_labels);		
		for i in range(self.N):
			clf = MVM( self.estimator, self.soft )
			clf.fit(X, y)
			self.ensemble.append( clf )
			
	def predict(self, X):
		n_samples = len(X);
		if self.soft:
			confidences = np.zeros((n_samples, self.nclass)) 
			for i in range(self.N):
				pred, conf = self.ensemble[i].predict(X, return_conf=True)
				confidences += conf
			return self.class_labels[ confidences.argmax(axis=1) ]
		else:
			votes = np.zeros((n_samples, self.nclass)) 
			for i in range(self.N):
				pred = self.ensemble[i].predict(X)
				for i,p in enumerate(pred): votes[ i, p ] += 1
			return self.class_labels[ votes.argmax(axis=1) ]
			

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


	def test(X, y, soft=True):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
		
		for ker, clf in { 
			"DecisionTree" : MVM_ensemble( DecisionTreeClassifier(random_state=0), soft=soft ),
			#"Knn         " : MVM_ensemble( KNeighborsClassifier(n_neighbors=3),  soft=soft ),
			"GaussianNB  " : MVM_ensemble( GaussianNB(),  soft=soft ),
			"LinearSVM   " : MVM_ensemble( SVC(kernel='linear', random_state=0, probability=True),  soft=soft ),
			"RadialSVM   " : MVM_ensemble( SVC(kernel='rbf', random_state=0, probability=True),  soft=soft )
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
