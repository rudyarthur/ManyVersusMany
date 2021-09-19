from utils import _predict_binary
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import LabelBinarizer

class OVR:
	"""
	One versus Rest Classification.
	Basically equivalent to SKlearn https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
	"""
	def __init__(self, estimator):
		self.estimator = estimator;

	def fit(self, X, y):
		
		self.estimators_ = []

		self.label_binarizer_ = LabelBinarizer(sparse_output=True)
		Y = self.label_binarizer_.fit_transform(y)
		Y = Y.tocsc()
		self.classes_ = self.label_binarizer_.classes_
		columns = (col.toarray().ravel() for col in Y.T) #<- 1 if class i else 0
		for i, column in enumerate(columns):
			e = clone(self.estimator)
			e.fit(X, column)
			self.estimators_.append(e)

        
	def predict(self, X):

		n_samples = len(X)
		
		maxima = np.empty(n_samples, dtype=float)
		maxima.fill(-np.inf)
		argmaxima = np.zeros(n_samples, dtype=int)
		for i, e in enumerate(self.estimators_):
			pred = _predict_binary(e, X) ##can't use votes because some examples can recieve none!
			np.maximum(maxima, pred, out=maxima)
			argmaxima[maxima == pred] = i
		return self.classes_[argmaxima]
            

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
			"DecisionTree" : OVR( DecisionTreeClassifier(random_state=0) ),
			"Knn         " : OVR( KNeighborsClassifier(n_neighbors=3) ),
			"GaussianNB  " : OVR( GaussianNB() ),
			"LinearSVM   " : OVR( SVC(kernel='linear', random_state=0) ),
			"RadialSVM   " : OVR( SVC(kernel='rbf', random_state=0) )
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

