import pandas as pd
from sklearn.preprocessing import Imputer
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import pylab as pl
import numpy as np


def RandomForest(data, i):

	df = pd.read_csv(data)
	#df2 = pd.read_csv(data2)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	
	horizon = int(raw_input("What is the horizon we want to predict in months? "))
	ret_str = 'Ret' #'RET_F12M_OP'	
	f = open("Output.txt","w")
	#if horizon == '6':
	#	ret_str = 'RET_F6M_OP'
	count = 0
	itr = int(raw_input("How many times should we iterate? "))
	while count < itr: 
		tr_len = int(i)*52+count*(horizon*4);	

		X = df.drop(ret_str, axis=1)
		X = X.drop('FACTORS', axis=1)
		X = X.drop('PERIOD', axis=1)
		X_trans = imp.fit_transform(X)
		y = df[ret_str]


		X_train = X_trans[:tr_len]
		y_train = y[:tr_len]


		X_test = X_trans[tr_len+(horizon*4):tr_len+2*(horizon*4)]
		y_test = y[tr_len+(horizon*4):tr_len+2*(horizon*4)]

		#np.savetxt('X_train'+str(count)+'.csv',X_train, delimiter=",")
		#np.savetxt('y_train'+str(count)+'.csv',y_train, delimiter=",")
		#np.savetxt('X_test'+str(count)+'.csv',X_test, delimiter=",")
		#np.savetxt('y_test'+str(count)+'.csv',y_test, delimiter=",")

		rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=12, random_state=1)

		rf.fit(X_train, y_train)

		y_predict = rf.predict(X_test)

		y_prob = rf.predict_proba(X_test)


		precision, recall, thresholds= precision_recall_curve(y_test, y_predict)
		pl.clf()
		pl.plot(recall, precision)
		pl.xlabel('Recall')
		pl.ylabel('Precision')
		pl.savefig('figure'+(str(count))+'.png')


		f.write("\n" + str(count) + " iteration" + "\n")
		f.write(str(y_prob) + "\n")
		f.write(str(y_predict) +"\n")
		acc_score = accuracy_score(y_test, y_predict)
		f.write(str(acc_score)+"\n")
		#cm = confusion_matrix(y_test, y_predict)
		f.write(str(pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Predicted'],margins=True))+"\n")
		count += 1
	f.close()

def main():
	data = raw_input("Name the data .csv file: ")
	#data2 = raw_input("Name the data file that want to predict: ")
	i = raw_input("How much test data should we use (in years): ")
	model = raw_input("What kind of model would you like to use?" + "\n" + "(1) Random Forest" + "\n")
	if model=='1':
		RandomForest(data, i)

if __name__ == "__main__":
	main()
