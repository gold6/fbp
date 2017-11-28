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
from rr_forest import RRForestClassifier
from sklearn.grid_search import GridSearchCV
import pylab as pl
import numpy as np

def RandomForest(data, i, horizon, itr, default, factor):
		
	f = open("RF_Results"+factor+"_"+(str(itr))+".txt","w")
	best_acc = 0
	best_mtry = 5
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)
		
		if default == '1':
			param_test = {'max_features':range(2,26,1)}
			gsearch = GridSearchCV(estimator = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=1), param_grid = param_test, scoring='roc_auc', n_jobs=4, cv=10)
			gsearch.fit(X_train, y_train)
			#f.write("\n" + "Scores: " + str(gsearch.grid_scores_) + "\n" + "Best Params: " + "\n" + str(gsearch.best_params_) + "\n" + "Best Score: " + str(gsearch.best_score_) + "\n")
			
			rf = gsearch.best_estimator_
			f.write(str(rf))
			#for j in range (1, 25):
			#	rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_features=j, random_state=1)

			#	rf.fit(X_train, y_train)

			#	y_predict = rf.predict(X_test)

			#	y_prob = rf.predict_proba(X_test)
		
			#	acc_score = accuracy_score(y_test, y_predict)
			#	if acc_score > best_acc:
			#		best_acc = acc_score
			#		best_mtry = j
				#f.write("\n" + "M_try: " + str(j) + "\n")
				#computestats(y_predict, y_prob, count, y_test, f)

			#f.write("\n" + "Iteration: " + str(count) + " Best M_Try: " + str(best_mtry)+ "\n")
			
			#rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_features=best_mtry, random_state=1)

			rf.fit(X_train, y_train)

			y_predict = rf.predict(X_test)

			y_prob = rf.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:
			rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_features=12, random_state=1)#class_weight={0:.9, 1:.1}--this attribute changes weighting which affects decisions, can change as we wish

			rf.fit(X_train, y_train)

			y_predict = rf.predict(X_test)

			y_prob = rf.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1
		print (float((count/itr)*100) + "% Done")
	f.close()
def RotationForest(data, i, horizon, itr, default, factor):
		
	f = open("RotF_Results_"+factor+"_"+(str(itr))+".txt","w")
	best_acc = 0
	best_mtry = 5
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)
		if default == '1':
			param_test = {'max_features':range(2,26,1)}
			gsearch = GridSearchCV(estimator = RRForestClassifier(n_estimators=1000, criterion='entropy', random_state=1), param_grid = param_test, scoring='roc_auc', n_jobs=4, cv=10)
			gsearch.fit(X_train, y_train)
			#f.write("\n" + "Scores: " + str(gsearch.grid_scores_) + "\n" + "Best Params: " + "\n" + str(gsearch.best_params_) + "\n" + "Best Score: " + str(gsearch.best_score_) + "\n")
			
			rotf = gsearch.best_estimator_
			f.write(str(rf))
		
		
		#	for j in range (1, 50):
		#		rotf = RRForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=j, random_state=1)

		#		rotf.fit(X_train, y_train)

		#		y_predict = rotf.predict(X_test)

		#		y_prob = rotf.predict_proba(X_test)
		
		#		acc_score = accuracy_score(y_test, y_predict)
		#		if acc_score > best_acc:
		#			best_acc = acc_score
		#			best_mtry = j
				#f.write("\n" + "M_try: " + str(j) + "\n")
				#computestats(y_predict, y_prob, count, y_test, f)

		#	f.write("\n" + "Iteration: " + str(count) + " Best M_Try: " + str(best_mtry)+ "\n")
			
		#	rotf = RRForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=best_mtry, random_state=1)

			rotf.fit(X_train, y_train)

			y_predict = rotf.predict(X_test)

			y_prob = rotf.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:
			rotf = RRForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=12, random_state=1)

			rotf.fit(X_train, y_train)

			y_predict = rotf.predict(X_test)

			y_prob = rotf.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1
		print (float((count/itr)*100) + "% Done")
	f.close()
	print ("Finished All Predictions for " + factor)

def GBM(data, i, horizon, itr, default, factor):
		
	f = open("GBM_Results_"+factor+"_"+(str(itr))+".txt","w")
	best_acc = 0
	best_depth = 2
	best_learn = .01
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)
		
		if default == '1':
			param_test = {'n_estimators':range(100,1501,100), 'max_depth':range(2,8,1)}
			gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=.01, random_state=1), param_grid = param_test, scoring='roc_auc', n_jobs=4, cv=10)
			gsearch.fit(X_train, y_train)
			#f.write("\n" + "Scores: " + str(gsearch.grid_scores_) + "\n" + "Best Params: " + "\n" + str(gsearch.best_params_) + "\n" + "Best Score: " + str(gsearch.best_score_) + "\n")
			
			gbm = gsearch.best_estimator_ #GradientBoostingClassifier(n_estimators=n_est, max_depth=max_d, learning_rate = .05, random_state=1)
			f.write(str(gbm))
			#print gbm
			gbm.fit(X_train, y_train)

			y_predict = gbm.predict(X_test)

			y_prob = gbm.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)

			computestats(y_predict, y_prob, count, y_test, acc_score, f)
			#for k in range (2, 8):
			#	for j in [.01, .05, .1, .15]:
			#		gbm = GradientBoostingClassifier(n_estimators=1500, max_depth=k, learning_rate = j, random_state=1)

			#		gbm.fit(X_train, y_train)

			#		y_predict = gbm.predict(X_test)

			#		y_prob = gbm.predict_proba(X_test)
		
			#		acc_score = accuracy_score(y_test, y_predict)
			#		if acc_score > best_acc:
			#			best_acc = acc_score
			#			best_depth = k
			#			best_learn = j
					#f.write("\n" + "M_try: " + str(j) + "\n")
					#f.write("\n" + "Learning Rate: " + str((k*.1)) + "\n")
					#f.write("\n" + "Accuracy Score: " + str(acc_score) + "\n")
					#computestats(y_predict, y_prob, count, y_test, f)

			#f.write("\n" + "Iteration: " + str(count) + " Best Depth: " + str(best_depth)+ "\n")
			#f.write("\n" + "Iteration: " + str(count) + " Best Learning Rate: " + str(best_learn)+ "\n")			
			
			#gbm = GradientBoostingClassifier(n_estimators=1500, max_depth=best_depth, learning_rate = best_learn, random_state=1)

			#gbm.fit(X_train, y_train)

			#y_predict = gbm.predict(X_test)

			#y_prob = gbm.predict_proba(X_test)
			#acc_score = accuracy_score(y_test, y_predict)

			#computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:

			gbm = GradientBoostingClassifier(n_estimators=1500, max_depth=5, learning_rate = 0.1, random_state=1)

			gbm.fit(X_train, y_train)

			y_predict = gbm.predict(X_test)

			y_prob = gbm.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)

			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1
		print (str(float((count/itr)*100)) + "% Done")
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
	data_in = raw_input("Name the data(s) .csv file: ")
	data_in_arr = ['' for x in range(0,2)]
	data_arr = ['' for x in range(0,2)]
	data_hor = ['' for x in range(0,2)]
	horizon = int(raw_input("What is the horizon we want to predict in months? "))
	count = 0
	while (data_in != 'End'):
		data = data_in + ".csv"
		data_in_arr[count] = data_in
		data_arr[count] = data
		data_hor[count] = horizon
		count += 1
		data_in = raw_input("Name another factor should we use:")
		horizon = raw_input("What is the horizon for this factor?: ")
	
	i = raw_input("How much test data should we use (in years): ")
	model = raw_input("What kind of model would you like to use?" + "\n" + "(1) Random Forest" + "\n" + "(2) GBM" + "\n" + "(3) SVM" + "\n"+ "(4) Linear Regression" + "\n"+ "(5) Rotational Forest" + "\n"+ "(6) Compare all methods" + "\n")
	default = raw_input("Should we find the best parameters or use default?" + "\n" + "(1) Best" + "\n" + "(0) Default" + "\n")
	
	itr = int(raw_input("How many times should we iterate? "))
	if model=='1':
		for y in range(0,2):
			RandomForest(data_arr[y], i, data_hor[y], itr, default, data_in_arr[y])
			#print data_arr[y], data_hor[y], data_in_arr[y]
	if model=='2':
		for y in range(0,2):
			GBM(data_arr[y], i, data_hor[y], itr, default, data_in_arr[y])
	if model=='3':
		SVM(data, i, horizon, itr, default)
	if model=='4':
		LinReg(data, i, horizon, itr, default)
	if model=='5':
		for y in range(0,2):			
			RotationForest(data_arr[y], i, data_hor[y], itr, default, data_in_arr[y])
	if model=='6':
		for y in range(0,2):
			RandomForest(data_arr[y], i, data_hor[y], itr, default, data_in_arr[y])
			GBM(data_arr[y], i, data_hor[y], itr, default, data_in_arr[y])
			RotationForest(data_arr[y], i, data_hor[y], itr, default,data_in_arr[y])

if __name__ == "__main__":
	main()
