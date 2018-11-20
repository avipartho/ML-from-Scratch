import numpy as np
import pandas as pd
from tqdm import tqdm

def sig(x, w): #Sigmoid function
	return 1/(1+np.exp(-np.dot(x, w)))

def cost(w, x, y, m, reg = 0): # Cost and gradient calculation
	wn, h = np.copy(w), sig(x,w)
	wn[0] = 0 # Weight for regularization, setting first element zero 
	J = (1/m)*np.sum(-np.multiply(y,np.log(h))-np.multiply(1-y,np.log(1-h))) 
	+ (reg/(2*m))*np.sum(wn**2) # Cost
	grad = (1/m)*(np.dot(x.T, h-y) +reg*wn) # Gradient
	return J, grad, h

def acc(target, pred, mc = False): # accuracy function
	if(mc):	# for multiclass create prediction vector
		p = np.argmax(pred, axis = 1)
		pred = p.reshape((len(p),1))
	else:
		for j,i in enumerate(pred):
			if i>=0.5 : pred[j] = 1
			else: pred[j] = 0 
	target = target.reshape((len(target),1)) # Reshaping target to suitable numpy size
	return np.sum(target == pred)/float(len(target))

def logistic(x, y, num_steps = 2000000, lr = 5e-5, r = 0, mc = False):
	m, best_acc, class_list = len(y), 0, list(set(y))
	# w = np.exp(-3)*np.random.randn(x.shape[1],1) # Random initialization of weights
	w, class_no = np.zeros((x.shape[1],1)), len(class_list)
	if(mc): #if multiclass, create seperate y(1 if that class, 0 otherwise) for each class
		y_new, w = [], np.zeros((x.shape[1],class_no))
		for i in class_list:
			for j in y:
				if (j==i): y_new.append(1)
				else: y_new.append(0)
		y_new = np.asarray(y_new)
		y_new = y_new.reshape((class_no,m)).T # seperating y into its classes by [no. of examples,class]

		for step in tqdm(range(num_steps)): # Iterating over defined steps
			j, gradient, pred = cost(w, x, y_new, m, reg = r)
			w -= lr * gradient # Gradient update	
			if step % 10000 == 0: # Print after each 10k steps
				accu = acc(y, pred, mc = True)
				# print ("Cost at iteration {:g}: {:g}, acc: {:g}".format(step, j, accu))
				if (accu > best_acc): best_acc,best_w = accu,w # checking for best weights
	else:
		y = y.reshape((m,1)) # Reshaping y to suitable numpy size
		for step in tqdm(range(num_steps)): # Iterating over defined steps
			j, gradient, pred = cost(w, x, y, m, reg = r)
			w -= lr * gradient # Gradient update	
			if step % 10000 == 0: # Print after each 10k steps
				accu = acc(y, pred)
				# print ("Cost at iteration {:g}: {:g}, acc: {:g}".format(step, j, accu))
				if (accu > best_acc): best_acc,best_w = accu,w # checking for best weights

	print ("Last iteration accuracy : {:g}, Best accuracy on training data: {:g}".format(accu, best_acc))						 
	return best_w

from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

#...............loading data ...............
iris = datasets.load_iris() # Loading iris dataset
x, y = iris.data, iris.target # Features and labels

# bc = datasets.load_breast_cancer() # Loading breast cancer dataset
# x, y = bc.data, bc.target # Features and labels

# data = pd.read_csv("diab_data.csv") # Loading sample diabetes data
# feature_column_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'skin_thickness', 
# 'insulin', 'bmi', 'diab_pred', 'age']
# x, y = data[feature_column_names].values, data[['diabetes']].values

# Impute/fill all 0 readings with mean 
fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)
x = fill_0.fit_transform(x)

#...............preprocessing................
# Feature normalization
mu, std, x_normalized = np.mean(x, axis = 0), np.std(x, axis = 0), []
for i,j,k in zip(x.T,mu,std):
	x_normalized.append((i-j)/k)
x = np.asarray(x_normalized).T

# Adding ones in the first column (for bias)
x_tmp = np.ones((x.shape[0],x.shape[1]+1))
x_tmp[:,1:] = x

x = x_tmp
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 25) # 20% as test

#...............Training & evaluation.................
# Using my classifier
w = logistic(x_train, y_train, num_steps = 5000000, mc = True) # Train
pred = sig(x_test,w) # Prediction probability
print("Accuracy on test data from my classifier without regularization: {:g}".format(acc(y_test, pred, mc = True)))

# Using sklearn classifier
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C = 1e15)
log.fit(x_train, y_train.ravel())
pred = log.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy on test data from sklearn classifier without regularization: {:g}".format(accuracy_score(y_test.ravel(), pred.ravel()))) 