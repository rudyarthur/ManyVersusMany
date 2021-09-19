import numpy as np
import sys

def _predict_binary(estimator, X):
	try:
		score = np.ravel(estimator.decision_function(X))
	except (AttributeError, NotImplementedError):
		score = estimator.predict_proba(X)[:, 1]
		
	return score
    
