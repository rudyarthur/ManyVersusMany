from utils import _predict_binary
import numpy as np
from sklearn.base import clone

from OVO import OVO
from OVR import OVR

class AandO: 
	"""
	All and One classifier
		Apply the One versus Rest classifier to get the top 2 classes
		Apply the One versus One classifier on these classes
	Computes all the classifiers up front.
	"""
	def __init__(self, estimator_type):
		self.estimator_type = estimator_type;
		self.ovo_classifier = OVO( estimator_type )
		self.ovr_classifier = OVR( estimator_type )
		
	def fit(self, X, y):
		self.ovo_classifier.fit(X, y)
		self.ovr_classifier.fit(X, y)

	def predict(self, X):
		n_samples = len(X)
		n_classes = len(self.ovr_classifier.estimators_)
		predictions = np.empty( (n_classes, n_samples), dtype=float)

		for i, e in enumerate(self.ovr_classifier.estimators_):
			predictions[i,:] = _predict_binary(e, X); 

		best = np.argmax(predictions, axis=0 )
		for i in range(n_samples): predictions[best[i], i] = -np.inf;
		second = np.argmax(predictions, axis=0 )
		
		final_predictions = np.empty( n_samples, dtype=int )
		for i, c in enumerate(zip(best, second)):
			cs = tuple(sorted(c))
			res = self.ovo_classifier.estimators_[  self.ovo_classifier.idx[cs]  ].predict([X[i]])
			final_predictions[i] = cs[ res[0] ];
		
		return final_predictions
            

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


	def test(X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
		
		for ker, clf in { 
			"DecisionTree" : AandO( DecisionTreeClassifier(random_state=0) ),
			"Knn         " : AandO( KNeighborsClassifier(n_neighbors=3) ),
			"GaussianNB  " : AandO( GaussianNB() ),
			"LinearSVM   " : AandO( SVC(kernel='linear', random_state=0) ),
			"RadialSVM   " : AandO( SVC(kernel='rbf', random_state=0) )
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

