from utils import _predict_binary
import numpy as np
from sklearn.base import clone
import sys

def log_nCr(n,r):
	#avoid computing big factorials, going to normalise with 2**n anyway
	#log( n!/r!(n-r)! )
	#log(n!) - log(r!) - log( (n-r)! )
	s = 0;
	for i in range( max(r, n-r)+1 , n+1):
		s += np.log(i);
	for i in range( 1, min(r, n-r)+1 ):
		s -= np.log(i);
	return s;		

def nCr(n,r):
	return round( np.exp(log_nCr(n,r)) );		
    
def random_split(labels):
	#choose a split size
	n = len(labels)
	s = 0;
	while s == 0 or s == n:
		s = np.random.choice( np.arange(n+1), size=1, p=[ np.exp( log_nCr(n,r) - n*np.log(2) ) for r in range(n+1)] )[0];
	
	a = np.sort( np.random.choice( labels, size=s, replace=False ) )
	return a, np.sort( np.setdiff1d(labels, a, assume_unique=True) )
	
class MVM:
	"""
	Many versus Many classifier
	Split the classes recursively 
			123|45
			/    \
		  1|23	  4|5
		     \
			 2|3
	This version computes the splits randomly
	Hard prediction = Take only one path
	Soft prediction = Take all the paths, add the confidences, choose max.
	"""
	
	def __init__(self, estimator, soft=True):
		self.estimator = estimator;
		self.estimators_ = {}
		self.soft = soft
		
	def fit(self, X, y, level=0, use_split=None):
		class_labels = np.unique(y)
		nclass = len(class_labels)		
		if level == 0: 
			self.class_labels = tuple(class_labels);
			self.nclass = nclass;
		
		if use_split is not None:
			a, b = use_split[tuple(class_labels)]
		else:		
			a, b = random_split(class_labels)
				
		y_binary = np.zeros(y.shape, int)
		y_binary[np.isin(y, b)] = 1

		e = clone(self.estimator)
		e.fit(X, y_binary)
		self.estimators_[ tuple(class_labels) ] = ( tuple(a), tuple(b), e )
		
		for g in (a,b):
			if len(g) > 1: 
				self.fit( X[np.isin(y, g)], y[np.isin(y, g)], level+1, use_split=use_split )
			

	def sum_at_leaf(self, X, lab, lab_scores, cum_sum, level=0):
		if len(lab) != 1:
			pred = self.estimators_[ lab ][2].predict_proba( [X] )[0]
			if pred[0] == 0:
				p0 = -np.inf
			else:
				p0 = np.log(pred[0])
			if pred[1] == 0:
				p1 = -np.inf
			else:
				p1 = np.log(pred[1])
					
			self.sum_at_leaf(X, self.estimators_[ lab ][ 0 ], lab_scores, cum_sum+p0, level+1 )
			self.sum_at_leaf(X, self.estimators_[ lab ][ 1 ], lab_scores, cum_sum+p1, level+1 )
		else:
			lab_scores[ lab[0] ] += cum_sum;
			
    
	def soft_predict(self, X, return_conf):
		n_samples = len(X);
		classes = np.empty(n_samples, int)
		if return_conf: confs = np.empty( (n_samples, self.nclass), float)
		for s in range(n_samples):
			lab = self.class_labels;
			lab_scores = np.zeros(self.nclass)
			self.sum_at_leaf( X[s], lab, lab_scores, 0)
			if return_conf: confs[s, :] = lab_scores			
			classes[s] =  self.class_labels[ np.argmax(lab_scores) ]
		if return_conf:
			return classes, confs
		return classes

			
	def hard_predict(self, X):
		n_samples = len(X);
		classes = np.empty(n_samples, int)
		for s in range(n_samples):
			lab = self.class_labels;
			while len(lab) > 1:
				pred = self.estimators_[ lab ][2].predict( [X[s]] )[0]
				lab = self.estimators_[ lab ][ pred ]
				if len(lab) == 1:
					classes[s] = lab[0]
					break;
		return classes
		
	def predict(self, X, return_conf=False):
		if return_conf and not self.soft:
			print("return_conf only works with soft prediction");
		if self.soft:
			return self.soft_predict( X, return_conf );
		else:
			return self.hard_predict( X );
			

if __name__ == "__main__":
	from sklearn.datasets import load_iris
	from sklearn.datasets import load_digits
	from sklearn.datasets import load_wine
	
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.svm import SVC
	from sklearn.naive_bayes import GaussianNB
	from sklearn.svm import LinearSVC	
	from sklearn.model_selection import train_test_split

	from sklearn.metrics import matthews_corrcoef
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import balanced_accuracy_score


	def test(X, y, soft=False):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
		
		for ker, clf in { 
			"DecisionTree" : MVM( DecisionTreeClassifier(random_state=0), soft ),
			"Knn         " : MVM( KNeighborsClassifier(n_neighbors=3), soft ),
			"GaussianNB  " : MVM( GaussianNB(), soft ),
			"LinearSVM   " : MVM( SVC(kernel='linear', random_state=0, probability=True), soft ),
			"RadialSVM   " : MVM( SVC(kernel='rbf', random_state=0, probability=True), soft )
			}.items():
			
			np.random.seed(123456789)
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

