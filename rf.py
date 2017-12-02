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

def RandomForest(data, i, horizon, itr, default, factor, rf_big_csv):
		
	f = open("RF_Results"+factor+"_"+(str(itr))+".txt","w")
	best_acc = 0
	best_mtry = 5
	count = 0
	all_probs = pd.Series()
	all_dates = pd.Series()
	while count < itr: 
		
		X_train, y_train, X_test, y_test, all_dates = handlecsv(data, count, horizon, i, all_dates)
		
		if default == '1':
			param_test = {'max_features':range(2,26,1)}
			gsearch = GridSearchCV(estimator = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=1), param_grid = param_test, scoring='roc_auc', n_jobs=4, cv=10)
			gsearch.fit(X_train, y_train)
			
			rf = gsearch.best_estimator_
			f.write(str(rf))
			
			rf.fit(X_train, y_train)

			y_predict = rf.predict(X_test)

			y_prob = rf.predict_proba(X_test)
			all_probs = pd.concat([all_probs,pd.Series(y_prob[:,1])], ignore_index=True)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:
			rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_features=12, random_state=1)#class_weight={0:.9, 1:.1}--this attribute changes weighting which affects decisions, can change as we wish

			rf.fit(X_train, y_train)

			y_predict = rf.predict(X_test)

			y_prob = rf.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
			all_probs = pd.concat([all_probs,pd.Series(y_prob[:,1])], ignore_index=True)

		count += 1
		print (factor + " Iteration " + str(count) + " Done")

	rf_big_csv[factor] = all_probs
	rf_big_csv['PERIOD'] = all_dates
	rf_big_csv.to_csv('RandForrestProbs.csv')
	f.close()
def RotationForest(data, i, horizon, itr, default, factor, rrf_big_csv):
		
	f = open("RotF_Results_"+factor+"_"+(str(itr))+".txt","w")
	best_acc = 0
	best_mtry = 5
	count = 0
	all_probs = pd.Series()
	all_dates = pd.Series()
	while count < itr: 
		
		X_train, y_train, X_test, y_test, all_dates = handlecsv(data, count, horizon, i, all_dates)
		if default == '1':
			param_test = {'max_features':range(2,26,1)}
			gsearch = GridSearchCV(estimator = RRForestClassifier(n_estimators=1000, criterion='entropy', random_state=1), param_grid = param_test, scoring='roc_auc', n_jobs=4, cv=10)
			gsearch.fit(X_train, y_train)
			#f.write("\n" + "Scores: " + str(gsearch.grid_scores_) + "\n" + "Best Params: " + "\n" + str(gsearch.best_params_) + "\n" + "Best Score: " + str(gsearch.best_score_) + "\n")
			
			rotf = gsearch.best_estimator_
			f.write(str(rf))

			rotf.fit(X_train, y_train)

			y_predict = rotf.predict(X_test)

			y_prob = rotf.predict_proba(X_test)
			all_probs = pd.concat([all_probs,pd.Series(y_prob[:,1])], ignore_index=True)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		else:
			rotf = RRForestClassifier(n_estimators=30, criterion='entropy', max_depth=15, max_features=12, random_state=1)

			rotf.fit(X_train, y_train)

			y_predict = rotf.predict(X_test)

			y_prob = rotf.predict_proba(X_test)
			all_probs = pd.concat([all_probs,pd.Series(y_prob[:,1])], ignore_index=True)
			acc_score = accuracy_score(y_test, y_predict)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		count += 1
		print (factor + " Iteration " + str(count) + " Done")
	
	rrf_big_csv[factor] = all_probs
	rrf_big_csv['PERIOD'] = all_dates
	rrf_big_csv.to_csv('RotForrestProbs.csv')
	f.close()

def GBM(data, i, horizon, itr, default, factor, gbm_big_csv):
		
	f = open("GBM_Results_"+factor+"_"+(str(itr))+".txt","w")
	best_acc = 0
	best_depth = 2
	best_learn = .01
	count = 0
	all_probs = pd.Series()
	all_dates = pd.Series()
	while count < itr: 
		
		X_train, y_train, X_test, y_test, all_dates = handlecsv(data, count, horizon, i, all_dates)
		
		if default == '1':
			param_test = {'n_estimators':range(100,1501,100), 'max_depth':range(2,8,1)}
			gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=.01, random_state=1), param_grid = param_test, scoring='roc_auc', n_jobs=4, cv=10)
			gsearch.fit(X_train, y_train)
			
			gbm = gsearch.best_estimator_ #GradientBoostingClassifier(n_estimators=n_est, max_depth=max_d, learning_rate = .05, random_state=1)
			f.write(str(gbm))
			gbm.fit(X_train, y_train)

			y_predict = gbm.predict(X_test)

			y_prob = gbm.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			all_probs = pd.concat([all_probs,pd.Series(y_prob[:,1])], ignore_index=True)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)

		else:

			gbm = GradientBoostingClassifier(n_estimators=1500, max_depth=5, learning_rate = 0.1, random_state=1)

			gbm.fit(X_train, y_train)

			y_predict = gbm.predict(X_test)

			y_prob = gbm.predict_proba(X_test)
			acc_score = accuracy_score(y_test, y_predict)
			all_probs = pd.concat([all_probs,pd.Series(y_prob[:,1])], ignore_index=True)
			computestats(y_predict, y_prob, count, y_test, acc_score, f)
		
		count += 1
		print (factor + " Iteration " + str(count) + " Done")
	gbm_big_csv[factor] = all_probs
	gbm_big_csv['PERIOD'] = all_dates
	gbm_big_csv.to_csv('GBMProbs.csv')
	f.close()

def SVM(data, i, horizon, itr, default):#So far will only predict 1, haven't found a proper config yet
		
	f = open("SVM_Results.txt","w")
	best_acc = 0
	count = 0
	
	while count < itr: 
		
		X_train, y_train, X_test, y_test = handlecsv(data, count, horizon, i)
		
		if default == '1':	
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

def handlecsv(data, count, horizon, i, all_dates):
	df = pd.read_csv(data)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	ret_str = 'Ret'
	tr_len = int(i)*52+count*(horizon*4);	

	X = df.drop(ret_str, axis=1)
	X = X.drop('FACTORS', axis=1)
	X = X.drop('PERIOD', axis=1)
	X_trans = imp.fit_transform(X)
	y = df[ret_str]
	
	dates = df['PERIOD']
	test_dates = dates[tr_len+(horizon*4):tr_len+2*(horizon*4)]
	#print test_dates
	all_dates = pd.concat([all_dates,pd.Series(test_dates)], ignore_index=True)

	X_train = X_trans[:tr_len]
	y_train = y[:tr_len]


	X_test = X_trans[tr_len+(horizon*4):tr_len+2*(horizon*4)]
	y_test = y[tr_len+(horizon*4):tr_len+2*(horizon*4)]

	return (X_train, y_train, X_test, y_test, all_dates)

def computestats(y_predict, y_prob, count, y_test, acc_score, f):
	precision, recall, thresholds= precision_recall_curve(y_test, y_predict)
	pl.clf()
	pl.plot(recall, precision)
	pl.xlabel('Recall')
	pl.ylabel('Precision')
	pl.savefig('PR'+(str(count))+'.png')

	f.write("\n" + " Iteration " + str(count) + "\n")
	f.write(str(y_prob) + "\n")
	f.write(str(y_predict) +"\n")
	f.write(str(acc_score)+"\n")
	f.write(str(pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Predicted'],margins=True))+"\n")
	try:
		auc_score = roc_auc_score(y_test, y_prob[:,1])
		f.write("ROC_AUC: " + str(auc_score)+"\n")
	except ValueError:
		pass

def main():
	factor_name = raw_input("Name the data(s) .csv file: ")
	factor_name_arr = ['' for x in range(0,2)]
	csv_name = ['' for x in range(0,2)]
	#data_hor = ['' for x in range(0,2)]
	horizon = int(raw_input("What is the horizon we want to predict in months? "))
	count = 0
	while (factor_name != 'End'):
		data = factor_name + "_6.csv"#currently hardcoded for 6 month horizon, needs to change if adapting to 12 month
		factor_name_arr[count] = factor_name
		csv_name[count] = data
		#data_hor[count] = horizon
		count += 1
		factor_name = raw_input("Name another factor should we use:")
		#horizon = raw_input("What is the horizon for this factor?: ")
	
	i = raw_input("How much test data should we use (in years): ")
	model = raw_input("What kind of model would you like to use?" + "\n" + "(1) Random Forest" + "\n" + "(2) GBM" + "\n" + "(3) SVM" + "\n"+ "(4) Linear Regression" + "\n"+ "(5) Rotational Forest" + "\n"+ "(6) Compare all methods" + "\n")
	default = raw_input("Should we find the best parameters or use default?" + "\n" + "(1) Best" + "\n" + "(0) Default" + "\n")
	
	itr = int(raw_input("How many times should we iterate? "))


	big_csv = pd.DataFrame(columns=['PERIOD'])
	for z in range(0,2):
		#print factor_name_arr[z]
		big_csv[factor_name_arr[z]] = z

	#print (str(big_csv))

	if model=='1':
		rf_big_csv = big_csv
		for y in range(0,2):
			RandomForest(csv_name[y], i, horizon, itr, default, factor_name_arr[y], rf_big_csv)
	if model=='2':
		gbm_big_csv = big_csv
		for y in range(0,2):
			GBM(csv_name[y], i, horizon, itr, default, factor_name_arr[y], gbm_big_csv)
	if model=='3':
		SVM(data, i, horizon, itr, default)
	if model=='4':
		LinReg(data, i, horizon, itr, default)
	if model=='5':
		rrf_big_csv = big_csv
		for y in range(0,2):			
			RotationForest(csv_name[y], i, horizon, itr, default, factor_name_arr[y], rrf_big_csv)
	if model=='6':
		rf_big_csv = big_csv
		gbm_big_csv = big_csv
		rrf_big_csv = big_csv
		for y in range(0,2):
			RandomForest(csv_name[y], i, horizon, itr, default, factor_name_arr[y], rf_big_csv)
			GBM(csv_name[y], i, horizon, itr, default, factor_name_arr[y], gbm_big_csv)
			RotationForest(csv_name[y], i, horizon, itr, default,factor_name_arr[y], rrf_big_csv)

if __name__ == "__main__":
	main()
