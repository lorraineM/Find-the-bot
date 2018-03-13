import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import math
import pandas as pd
import matplotlib.pyplot as plt

#load observations and labels
observations = np.genfromtxt('Observations.csv', delimiter=',')
labels = np.genfromtxt('Label', delimiter=',')

#make plots
#ob1_x = np.arange(901,1001)
#ob1_y = observations[0][900:1000]
#ob1 = pd.DataFrame({'ob1_x':ob1_x,'ob1_y':ob1_y})
#ob1.plot(x='ob1_x',y='ob1_y',title='angles change with last 100 steps in the first run')

#just use 100 runs to train two SVRs
#the training data is angles, the target data is the corresponding x-coordinates in the first SVR which is used to predict the x-coordinate of the 1001th location
#the training data is angles, the target data is the corresponding y-coordinates in the second SVR which is used to predict the y-coordinate of the 10001th location
x_coordinate = []
y_coordinate = []
angles = []
for i in range(10000):
	step = int(labels[i][1])
	run = int(labels[i][0])
	angles.append(observations[run-1][step-21:step-1])
	x_coordinate.append(labels[i][2])
	y_coordinate.append(labels[i][3])
x_coordinate = np.array(x_coordinate)
y_coordinate = np.array(y_coordinate)
angles = np.array(angles)

#cross validation to find the best C = 400 and gamma=0.15
#C_list = [100,200,300,400,500,600,700,800,900,1000]
#scores_list = []
#for c in C_list:
#	test_svr_for_C = svm.SVR(kernel='rbf',C=c , gamma=0.1)
#	avg_score = cross_val_score(test_svr_for_C, angles, x_coordinate).mean()
#	scores_list.append(avg_score)
#score_C = pd.DataFrame({'score':scores_list,'C':C_list})
#score_C.plot(x='C', y='score',title='accuracy for different Cs using cross validation')
#gamma_list = [0.05,0.1,0.15,0.2]
#scores_list = []
#for g in gamma_list:
#	test_svr_for_gamma = svm.SVR(kernel='rbf',C=400 , gamma=g)
#	avg_score = cross_val_score(test_svr_for_gamma, angles, x_coordinate).mean()
#	scores_list.append(avg_score)
#score_gamma = pd.DataFrame({'score':scores_list,'gamma':gamma_list})
#score_gamma.plot(x='gamma', y='score',title='accuracy for different gammas using cross validation')
#plt.show()

x_svr = svm.SVR(kernel='rbf',C=400 , gamma=0.15).fit(angles,x_coordinate)
y_svr = svm.SVR(kernel='rbf',C=400 , gamma=0.15).fit(angles,y_coordinate)

#we use angles of the last 20 steps for each run as the test data
#so 4000 samples with 20 features each sample
test_data = observations[6000:,-20:]
x_prediction = x_svr.predict(test_data)
y_prediction = y_svr.predict(test_data)

#output the prediction results
f = open('predictionUsingN20.csv', 'w')
f.writelines('\"Id\",\"Value\"\n')
for i in range(6001,10001):
	f.writelines(str(i)+'x,'+str(x_prediction[i-6001])+'\n')
	f.writelines(str(i)+'y,'+str(y_prediction[i-6001])+'\n')
f.close()
