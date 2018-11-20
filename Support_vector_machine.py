import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

def acc(target, pred): # accuracy function
	for j,i in enumerate(pred):
		if i>0 : pred[j] = 1
		else: pred[j] = -1 
	target = target.reshape((len(target),1)) # Reshaping target to suitable numpy size
	return np.sum(target == pred)/float(len(target))

def svm(x, y, num_steps = 50000, lr = 20, r = 0):
	best_acc, w = 0, np.zeros((1,x.shape[1]))
	# w = np.exp(-3)*np.random.randn(x.shape[1],1) # Random initialization of weights
	
	for step in (range(1,num_steps)): # Iterating over defined steps
		error = 0
		for i in range(len(x)):
			if (y[i]*np.dot(x[i].reshape((1, len(x[i]))), w.T)) < 1:
				w = w + lr * ( (x[i] * y[i])*(.999)**step + (-2  *(r)* w) ) #misclassified update for ours weights
				error += 1        
			else:
				w = w + lr * (-2  *(r)* w) #correct classification, update our weights
		# if step % 1000 == 0: # Print after each 10k steps
		pred = np.dot(x,w.T)
		accu = acc(y, pred)
		print ("Accuracy at iteration {:g}: {:g}, error : {:g}".format(step, accu, error))
		if (accu > best_acc): best_acc,best_w = accu,w # checking for best weights
		# print(w)	
	print ("Last iteration accuracy : {:g}, Best accuracy on training data: {:g}".format(accu, best_acc))						 
	return w

from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
# from sklearn.datasets.samples_generator import make_blobs

#...............loading data ...............
iris = datasets.load_iris() # Loading iris dataset
x, y = iris.data[:100,:2], iris.target[:100] # Features and labels

# bc = datasets.load_breast_cancer() # Loading breast cancer dataset
# x, y = bc.data, bc.target # Features and labels
# (x,y) =  make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.05,random_state=40)

# data = pd.read_csv("diab_data.csv") # Loading sample diabetes data
# feature_column_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'skin_thickness', 
# 'insulin', 'bmi', 'diab_pred', 'age']
# x, y = data[feature_column_names].values, data[['diabetes']].values

#...............preprocessing................
# changing y labels appropriately
for i in range(len(y)):
	if y[i] == 0: y[i] = -1

# Impute/fill all 0 readings with mean 
fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)
x = fill_0.fit_transform(x)

# Feature normalization
mu, std, x_normalized = np.mean(x, axis = 0), np.std(x, axis = 0), []
for i,j,k in zip(x.T,mu,std):
	x_normalized.append((i-j)/k)
x = np.asarray(x_normalized).T

# Adding ones in the last column (for bias) 
x_tmp = np.ones((x.shape[0],x.shape[1]+1))
x_tmp[:,:-1] = x
x = x_tmp

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) # 20% as test
# x_train, x_test, y_train, y_test = np.vstack((x[:40],x[50:90])), np.vstack((x[40:50],x[-10:])), np.append(y[:40],y[50:90]), np.append(y[40:50],y[-10:])

#...............Training & evaluation.................
# Using my classifier
w = svm(x_train, y_train, num_steps = 500) # Train
pred = np.dot(x_test,w.T) # Prediction probability
print("Accuracy on test data from my classifier without regularization: {:g}".format(acc(y_test, pred)))

# Using sklearn classifier
from sklearn import svm
clf_svm = svm.SVC(decision_function_shape='ovo')
clf_svm.fit(x_train, y_train.ravel())
pred = clf_svm.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy on test data from sklearn classifier without regularization: {:g}".format(accuracy_score(y_test.ravel(), pred.ravel()))) 

# ............plotting..............
def line(w, a):
	return -(w[0][0]*a+w[0][2])/w[0][1]

plt.scatter(x[:50,0],x[:50,1],marker='+',color='green')
plt.scatter(x[50:100,0],x[50:100,1],marker='_',color='blue')
plt.plot([-2,2.5],[line(w,-2),line(w,2.5)],'k-')
plt.show()