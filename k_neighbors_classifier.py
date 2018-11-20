from scipy.spatial import distance
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def euc(a, b): # Euclidean distance of two points by scipy
	return distance.euclidean(a,b)

class KNN(): # KNN from scratch
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []
		for i in x_test:
			label = self.closest(i) # Gaining label name of nearest neighbor data point
			predictions.append(label) # Predictions
		return predictions

	def closest(self, row):
		# best_dist = euc(row, self.x_train[0])
		best_dist = np.sqrt(np.sum((np.asarray(row)-np.asarray(self.x_train[0]))**2)) # Euclidean distance of two points by numpy
		best_index = 0
		for i  in range(1, len(self.x_train)):
			dist = np.sqrt(np.sum((np.asarray(row)-np.asarray(self.x_train[i]))**2))
			# dist = euc(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist # updating distance
				best_index = i # updating index

		return self.y_train[best_index]

def acc(test, pred): # accuracy function
	return np.sum(test == pred)/float(len(test))

from sklearn import datasets
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

iris = datasets.load_iris() # Loading dataset
x, y = iris.data, iris.target # Features and labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) # train-test splitting

clf = {0 : KNeighborsClassifier(), 1 : KNN()}

for i in clf: # iterating over the two classifiers
	print("Result for classifier "+str(i)+":")
	clf[i].fit(x_train, y_train) # train
	preds = clf[i].predict(x_test) # test
	# from sklearn.metrics import accuracy_score
	# print(accuracy_score(y_test, preds)) 
	print(acc(y_test, preds))