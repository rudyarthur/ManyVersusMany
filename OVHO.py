from utils import _predict_binary
import numpy as np
from sklearn.base import clone

class OVHO:
	"""
	One versus higher order classifier
	Sort classes from most popular to least 1,2,3,4,5 and create the predictors
	1 v 2345 ; 2 v 345, 3 v 45, 4 v 5
	To predict, apply each in turn until a singleton class is chosen.
	This is not the same as the method recommended in the original paper, but is how others implement it.
	"""	
	def __init__(self, estimator_type):
		self.estimator_type = estimator_type;
	def fit(self, X, y):

		self.class_labels, self.class_counts = np.unique(y, return_counts=True)
		self.nclass = len(self.class_labels)
		nsamples = len(X)
		self.estimators_ = []
		self.order = []
		
		Xmod = X;
		ymod = y;
		for i in np.argsort(self.class_counts)[self.nclass-1:0:-1]:
			self.order.append( self.class_labels[i] )
			labels = np.where( ymod==self.class_labels[i], np.ones(len(Xmod),dtype=int), np.zeros(len(Xmod),dtype=int))

			e = clone(self.estimator_type)
			e.fit(Xmod, labels)
			self.estimators_.append(e)

			Xmod = Xmod[ np.nonzero(ymod!=self.class_labels[i]) ] 
			ymod = ymod[ np.nonzero(ymod!=self.class_labels[i]) ] 

		self.order.append( self.class_labels[  np.argmin(self.class_counts)  ] )

				
	def predict(self, X):
		n_samples = len(X)
		classes = np.ones(n_samples, dtype=int)*-1;		
		for i, e in enumerate(self.estimators_):
			prediction = e.predict(X)
			classes[ (prediction == 1) & (classes == -1) ] = self.order[i]
		classes[ classes == -1 ] = self.order[-1]	
		return classes
            

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
			"DecisionTree" : OVHO( DecisionTreeClassifier(random_state=0) ),
			"Knn         " : OVHO( KNeighborsClassifier(n_neighbors=3) ),
			"GaussianNB  " : OVHO( GaussianNB() ),
			"LinearSVM   " : OVHO( SVC(kernel='linear', random_state=0) ),
			"RadialSVM   " : OVHO( SVC(kernel='rbf', random_state=0) )
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

