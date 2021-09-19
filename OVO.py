from utils import _predict_binary
import numpy as np
from sklearn.base import clone

class OVO:
	"""
	One versus One Classification.
	Basically equivalent to SKlearn https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html
	If classic = True only use confidences to resolve ties
	If classic = False always use confidences (same as SKlearn)
	"""
	def __init__(self, estimator_type):
		self.estimator_type = estimator_type;
		
	def fit(self, X, y):
		self.class_labels = np.unique(y)
		self.nclass = len(self.class_labels)
		self.estimators_ = []
		self.idx = {}
		
		k = 0;
		for ii in range(self.nclass):
			i = self.class_labels[ii]
			for jj in range(i+1,self.nclass):		
				j = self.class_labels[jj]						
				self.idx[ (i,j) ] = k
				cond = np.logical_or(y == i, y == j)
				y_ = y[cond]
				y_binary = np.empty(y_.shape, int)
				y_binary[y_ == i] = 0
				y_binary[y_ == j] = 1
				indcond = np.arange(X.shape[0])[cond]				

				e = clone(self.estimator_type)
				e.fit(X[indcond], y_binary)
				self.estimators_.append(e)
				k += 1
				
	def predict(self, X, classic=True):
		Xs = [X] * len(self.estimators_)
		predictions = np.vstack([est.predict(Xi) for est, Xi in zip(self.estimators_, Xs)]).T 
		if not classic:
			confidences = np.vstack([_predict_binary(est, Xi) for est, Xi in zip(self.estimators_, Xs)]).T
		
		
		n_samples = predictions.shape[0]
		votes = np.zeros((n_samples, self.nclass)) 
		if not classic:
			sum_of_confidences = np.zeros((n_samples, self.nclass))
                                     
		k = 0
		for i in range(self.nclass):
			for j in range(i + 1, self.nclass):
				votes[predictions[:, k] == 0, i] += 1
				votes[predictions[:, k] == 1, j] += 1
				if not classic:
					sum_of_confidences[:, i] -= confidences[:, k]
					sum_of_confidences[:, j] += confidences[:, k]
				k += 1

		if not classic:
			transformed_confidences = (sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1)))
			nv = votes + transformed_confidences 
			return self.class_labels[ nv.argmax(axis=1) ]

		classes = np.ones(n_samples, dtype=int)*-1;
		for s in range(n_samples):
			c = self.class_labels[ votes[s,:].argmax() ]
			inds = np.ravel( np.nonzero(votes[s,:] == votes[s,c]));
			if len(inds) > 1:
				
				k = 0
				conf = np.zeros(self.nclass, dtype=float)
				for i in range(self.nclass):
					for j in range(i + 1, self.nclass):
						p = _predict_binary(self.estimators_[k], [X[s,:]])
						conf[i] -= p
						conf[j] += p
						k += 1
				c = self.class_labels[ conf.argmax() ] 
				
				#print("conf", conf, self.class_labels[ conf.argmax() ] )				
				#print("i=", s, self.class_labels[ votes[s,:].argmax() ],self.class_labels[ nv[s,:].argmax() ], c  )
			classes[s] = c;
		
		#return self.class_labels[ votes.argmax(axis=1) ]
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
			"DecisionTree" : OVO( DecisionTreeClassifier(random_state=0) ),
			"Knn         " : OVO( KNeighborsClassifier(n_neighbors=3) ),
			"GaussianNB  " : OVO( GaussianNB() ),
			"LinearSVM   " : OVO( SVC(kernel='linear', random_state=0) ),
			"RadialSVM   " : OVO( SVC(kernel='rbf', random_state=0) )
			}.items():
			
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			print(ker , np.sum(y_test == y_pred), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred) )
		
	#print("Iris::")
	#X, y = load_iris(return_X_y=True)
	#test(X, y)
	#print("Digits::")	
	#X, y = load_digits(return_X_y=True)
	#test(X, y)
	#print("Wine::")	
	#X, y = load_wine(return_X_y=True)
	#test(X, y)	

