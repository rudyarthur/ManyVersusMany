import numpy as np
import csv
import sys

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

from OVO import OVO
from OVR import OVR
from OVHO import OVHO
from AandO import AandO
from MVM import MVM
from MVM_ensemble import MVM_ensemble
from MVM_best import MVM_best

def vfactory(vtype, kernel):
	if vtype == "OVO":
		return OVO(kernel);
	elif vtype == "OVR":
		return OVR(kernel);
	elif vtype == "OVHO":
		return OVHO(kernel);
	elif vtype == "AandO":
		return AandO(kernel);
	elif vtype == "MVM":
		return MVM(kernel);
	elif vtype == "MVM10h":
		return MVM_ensemble(kernel, N=10, soft=False);
	elif vtype == "MVM10s":
		return MVM_ensemble(kernel, N=10, soft=True);
	elif vtype == "MVMbest":
		return MVM_best(kernel, N=10, soft=False);
	elif vtype == "DecisionTree":
		return DecisionTreeClassifier(random_state=0)
	elif vtype == "RandomForest":
		return RandomForestClassifier(random_state=0, n_estimators=20)
	elif vtype == "GaussianNB  ":
		return GaussianNB()										
	elif vtype == "LinearSVM   ":
		return SVC(kernel='linear', random_state=0, probability=True)
	elif vtype == "RadialSVM   ":
		return SVC(kernel='rbf', random_state=0, probability=True) 
	elif vtype == "Knn         ":
		return KNeighborsClassifier(n_neighbors=3)
					
def test(X, y, dataset, verbose=False):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
	
	all_obs = {}
	best_kernel = {}
	for k, kernel in { 
		"DecisionTree" : DecisionTreeClassifier(random_state=0),
		"RandomForest" : RandomForestClassifier(random_state=0, n_estimators=20),
		"Knn         " : KNeighborsClassifier(n_neighbors=3),
		"GaussianNB  " : GaussianNB(),
		"LinearSVM   " : SVC(kernel='linear', random_state=0, probability=True) ,
		"RadialSVM   " : SVC(kernel='rbf', random_state=0, probability=True) 
		}.items():
		obs = {}
	
		if verbose: print("\tKernel ::", k);
		for vtype in ["OVO", "OVR", "OVHO", "AandO", "MVM", "MVM10s", "MVM10h", "MVMbest", k]:
			clf = vfactory(vtype, kernel)
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			obs[vtype] = (np.sum(y_test == y_pred), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred))
			all_obs[vtype + "(" + k + ")"] = (np.sum(y_test == y_pred), accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred))
		so = sorted(obs, key=lambda x : obs[x][0], reverse=True )
		if verbose: 
			for o in so:
				print(dataset+"::\t\t" + o+"\t\t", obs[o])	
		best_kernel[ k + "\t::\t" + so[0] ] = obs[ so[0] ]
			
	print("Overall Top 10 (not counting ties)")
	for o in sorted(all_obs, key=lambda x : all_obs[x][0], reverse=True )[:10]:
		print(dataset+"::\t\t" + o+"\t\t", all_obs[o])	
	print("Best for each Kernel (not counting ties)")
	for o in sorted(best_kernel, key=lambda x : best_kernel[x][0], reverse=True ):
		print(dataset+"::\t\t" + o+"\t\t", best_kernel[o])	


X, y = load_iris(return_X_y=True)
test(X, y, "Iris")
X, y = load_digits(return_X_y=True)
test(X, y, "Digits")
X, y = load_wine(return_X_y=True)
test(X, y, "Wine")	

X = np.loadtxt("yeast.data", usecols=range(1,9) )
label_map = {}
labels = []
with open("yeast.data", 'r') as infile:
	csvreader = csv.reader(infile, delimiter=" ")
	for row in csvreader:
		if row[-1] not in label_map:
			label_map[row[-1]] = len(label_map)
		labels.append( label_map[row[-1]] )
y = np.array(labels)
test(X, y, "Yeast")	

X = np.loadtxt("segmentation.data", usecols=range(1,20), delimiter="," )
label_map = {}
labels = []
with open("segmentation.data", 'r') as infile:
	csvreader = csv.reader(infile, delimiter=",")
	for row in csvreader:
		if row[0] not in label_map:
			label_map[row[0]] = len(label_map)
		labels.append( label_map[row[0]] )
y = np.array(labels)
test(X, y, "Segmentation")	


X = np.loadtxt("glass.data", usecols=range(2,10), delimiter="," )
label_map = {}
labels = []
with open("glass.data", 'r') as infile:
	csvreader = csv.reader(infile, delimiter=",")
	for row in csvreader:
		if row[-1] not in label_map:
			label_map[row[-1]] = len(label_map)
		labels.append( label_map[row[-1]] )
y = np.array(labels)
test(X, y, "Glass", verbose=True)	


data = np.loadtxt("seeds_dataset.txt" )
X = data[:,:8]
y = data[:,-1]-1
y = y.astype(int)
print(X.shape, y.shape)
test(X, y, "seeda", verbose=True)	

