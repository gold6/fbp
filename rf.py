import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import explained_variance_score
import pylab as pl
import numpy as np

def RandomForest(data, i, horizon, itr, default):
		
	f = open("RF_Results.txt","w")
	best_acc = 0
	best_mtry = 5
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)
		
		if default == '1':
		
			for j in range (1, 50):
				rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=j, random_state=1)

				rf.fit(X_train, y_train)

				y_predict = rf.predict(X_test)

				y_prob = rf.predict_proba(X_test)
		
				acc_score = accuracy_score(y_test, y_predict)
				if acc_score > best_acc:
					best_acc = acc_score
					best_mtry = j
				#f.write("\n" + "M_try: " + str(j) + "\n")
				#computestats(y_predict, y_prob, count, y_test, f)

			f.write("\n" + "Iteration: " + str(count) + " Best M_Try: " + str(best_mtry)+ "\n")
			
			rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=best_mtry, random_state=1)

			rf.fit(X_train, y_train)

			y_predict = rf.predict(X_test)

			y_prob = rf.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:
			rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=12, random_state=1)

			rf.fit(X_train, y_train)

			y_predict = rf.predict(X_test)

			y_prob = rf.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1

	f.close()

def GBM(data, i, horizon, itr, default):
		
	f = open("GBM_Results.txt","w")
	best_acc = 0
	best_mtry = 5
	best_learn = .1
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)
		
		if default == '1':
			for k in range (1, 8):
				for j in range (1, 50):
					gbm = GradientBoostingClassifier(n_estimators=30, max_depth=15, max_features=j, learning_rate = (k*.1), random_state=1)

					gbm.fit(X_train, y_train)

					y_predict = gbm.predict(X_test)

					y_prob = gbm.predict_proba(X_test)
		
					acc_score = accuracy_score(y_test, y_predict)
					if acc_score > best_acc:
						best_acc = acc_score
						best_mtry = j
						best_learn = k
					#f.write("\n" + "M_try: " + str(j) + "\n")
					#f.write("\n" + "Learning Rate: " + str((k*.1)) + "\n")
					#f.write("\n" + "Accuracy Score: " + str(acc_score) + "\n")
					#computestats(y_predict, y_prob, count, y_test, f)

			f.write("\n" + "Iteration: " + str(count) + " Best M_Try: " + str(best_mtry)+ "\n")
			f.write("\n" + "Iteration: " + str(count) + " Best Learning Rate: " + str(best_learn*.1)+ "\n")			
			
			gbm = GradientBoostingClassifier(n_estimators=30, max_depth=15, max_features=best_mtry, learning_rate = (best_learn*.1), random_state=1)

			gbm.fit(X_train, y_train)

			y_predict = gbm.predict(X_test)

			y_prob = gbm.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)

			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:

			gbm = GradientBoostingClassifier(n_estimators=30, max_depth=15, max_features=12, learning_rate = 0.1, random_state=1)

			gbm.fit(X_train, y_train)

			y_predict = gbm.predict(X_test)

			y_prob = gbm.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)

			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1

	f.close()

def SVM(data, i, horizon, itr, default):#So far will only predict 1, haven't found a proper config yet
		
	f = open("SVM_Results.txt","w")
	best_acc = 0
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)
		
		if default == '1':
			
			#for j in range (1, 50):
			#	svm_c = svm.NuSVC(probability = True, random_state=1)

			#	svm_c.fit(X_train, y_train)

			#	y_predict = svm_c.predict(X_test)

			#	y_prob = svm_c.predict_proba(X_test)
	
			#	acc_score = accuracy_score(y_test, y_predict)
			#	if acc_score > best_acc:
			#		best_acc = acc_score			
			
			svm_c = svm.NuSVC(probability = True, random_state=1)

			svm_c.fit(X_train, y_train)

			y_predict = svm_c.predict(X_test)

			y_prob = svm_c.predict_proba(X_test)

			acc_score = accuracy_score(y_test, y_predict)

			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:

			svm_c = svm.NuSVC(probability = True, random_state=1)

			svm_c.fit(X_train, y_train)

			y_predict = svm_c.predict(X_test)

			y_prob = svm_c.predict_proba(X_test)

			acc_score = accuracy_score(y_test, y_predict)

			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1

	f.close()

def LinReg(data, i, horizon, itr, default):
		
	f = open("LR_Results.txt","w")
	best_acc = 0
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)

		lr = LogisticRegression(random_state=1)

		lr.fit(X_train, y_train)

		y_predict = lr.predict(X_test)

		y_prob = lr.predict_proba(X_test)

		acc_score = accuracy_score(y_test, y_predict)

		computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1

	f.close()

def handlecsv(data, count, horizon, i):
	df = pd.read_csv(data)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	ret_str = 'Ret'
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
	return (X_train, y_train, X_test, y_test)

def computestats(y_predict, y_prob, count, y_test, acc_score, f):
	precision, recall, thresholds= precision_recall_curve(y_test, y_predict)
	pl.clf()
	pl.plot(recall, precision)
	pl.xlabel('Recall')
	pl.ylabel('Precision')
	pl.savefig('PR'+(str(count))+'.png')

	#false_pos_rate, true_pos_rate, thresholds = roc_curve(y_test, y_predict)
	#pl.clf()
	#pl.plot(false_pos_rate, true_pos_rate)
	#pl.xlabel('False Positives')
	#pl.ylabel('True Positives')
	#pl.savefig('ROC'+(str(count))+'.png')
	#auc_score = roc_auc_score(y_test, y_predict)

	f.write("\n" + " Iteration " + str(count) + "\n")
	f.write(str(y_prob) + "\n")
	f.write(str(y_predict) +"\n")
	f.write(str(acc_score)+"\n")
	f.write(str(pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Predicted'],margins=True))+"\n")
	
	#f.write(str(auc_score)+"\n")

def main():
	data = raw_input("Name the data .csv file: ")
	i = raw_input("How much test data should we use (in years): ")
	model = raw_input("What kind of model would you like to use?" + "\n" + "(1) Random Forest" + "\n" + "(2) GBM" + "\n" + "(3) SVM" + "\n"+ "(4) Linear Regression" + "\n"+ "(5) Compare all methods" + "\n")
	default = raw_input("Should we finethebest parameters or use default?" + "\n" + "(1) Best" + "\n" + "(0) Default" + "\n")
	horizon = int(raw_input("What is the horizon we want to predict in months? "))
	itr = int(raw_input("How many times should we iterate? "))
	if model=='1':
		RandomForest(data, i, horizon, itr, default)
	if model=='2':
		GBM(data, i, horizon, itr, default)
	if model=='3':
		SVM(data, i, horizon, itr, default)
	if model=='4':
		LinReg(data, i, horizon, itr, default)
	if model=='5':
		RandomForest(data, i, horizon, itr, default)
		GBM(data, i, horizon, itr, default)
		SVM(data, i, horizon, itr, default)
		LinReg(data, i, horizon, itr, default)

if __name__ == "__main__":
	main()
