# coding:utf-8

import pandas as pd
import gensim, logging
import numpy as np
from collections import defaultdict
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import IncrementalPCA
import time
import re
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import xgboost as xgb 
from xgboost import XGBClassifier
from scipy.sparse import hstack
from matplotlib import pyplot
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import *
from keras.models import Sequential
from keras.optimizers import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy import sparse
from data_preprocess import *
from matplotlib import pyplot

#-- xgboost cross validation model frame
def xgbModel( x_train, x_test, y_train, y_test, config, params, test_file, method='test'):
    # used for evaluating model
    
	print(np.unique(y_train))
    
	#--Set parameter: scale_pos_weight-- 
	if config['multiClass'] == True:
		params['num_class'] = len(np.unique(y_train))
		print(params['num_class'])
	else:
		params['scale_pos_weight'] = (float)(len(y_train[y_train == 0]))/len(y_train[y_train == 1])
		print(params['scale_pos_weight'])


	#--Get User-define DMatrix: dtrain--
	#print trainQid[0]
	dtrain = xgb.DMatrix(x_train, label = y_train)
  
	rounds = config['rounds']
	folds = config['folds']    

	#--Run CrossValidation--
    
# 	print('run cv: ' + 'round: ' + str(rounds) + ' folds: ' + str(folds))
# 	res = xgb.cv(params, dtrain, rounds,early_stopping_rounds=10, nfold = folds, verbose_eval = 2)
# 	print(res)
# 	watchlist = [(dtrain, 'train')]

	if method == 'test':       
		dtest = xgb.DMatrix(x_test, label = y_test) 
		evals = [(dtrain,'train'),(dtest,'test')]
		results = {}
		model = xgb.train(params, dtrain, rounds, evals=evals, evals_result=results, early_stopping_rounds=100, verbose_eval = 5)
		model.save_model('../model/test/xgb01.m')
		ptest = model.predict(dtest,ntree_limit=model.best_iteration)
		y_pred = [round(value) for value in ptest]
		print('testing...')
		accuracy = accuracy_score(y_test, y_pred)
		print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
		epochs = len(results['train']['auc'])
		x_axis = range(0, epochs)
# plot log loss
		fig, ax = pyplot.subplots()
		ax.plot(x_axis, results['train']['logloss'], label='Train')
		ax.plot(x_axis, results['test']['logloss'], label='Test')
		ax.legend()
		pyplot.ylabel('Log Loss')
		pyplot.title('XGBoost Log Loss')
		pyplot.show()
# # plot classification error
# 		fig, ax = pyplot.subplots()
# 		ax.plot(x_axis, results['train']['error'], label='Train')
# 		ax.plot(x_axis, results['test']['error'], label='Test')
# 		ax.legend()
# 		pyplot.ylabel('Classification Error')
# 		pyplot.title('XGBoost Classification Error')
# 		pyplot.show()
# plot auc
		fig, ax = pyplot.subplots()
		ax.plot(x_axis, results['train']['auc'], label='Train')
		ax.plot(x_axis, results['test']['auc'], label='Test')
		ax.legend()
		pyplot.ylabel('AUC')
		pyplot.title('XGBoost AUC')
		pyplot.show()     
        
	if method == 'predict':  
		dtest = xgb.DMatrix(x_test)
		evals = [(dtrain,'train')]  
		results = {}
		model = xgb.train(params, dtrain, rounds, evals=evals, evals_result=results, early_stopping_rounds=50, verbose_eval = 5)
		model.save_model('../model/predict/xgb02.m')
		ptest = model.predict(dtest,ntree_limit=model.best_iteration)
		y_pred = [round(value) for value in ptest]
		print('predicting...')
		res = pd.read_excel(test_file)
		res['prob'] = ptest
		res['type'] = y_pred
    
		print('result:')
		print(res.shape,res.head())
		save_path = '../res/res02.xls'
		print('save result: '+save_path)
		res.to_excel(save_path,index=0)
        
        
def xgb_predict(x_test):
    model = xgb.Booster({'n_thread':24})
    model.load_model('../model/xgb02.m')
    dtest = xgb.DMatrix(x_test)
    ypred = model.predict(dtest)
#     print(ypred)
    return ypred
    

def predict_strings(data):
    # used for predicting test strings list
    # input: strings list
    data = pd.Series(data)
    corpus,label = load_data()
   
    alldata = pd.concat([data,corpus]).reset_index()
    del alldata['index']
    alldata.rename(columns={0:'content'},inplace=True)
    print(alldata[:15])
    alldata = preprocess_data(alldata.content)
    allfeatures = generate_features(alldata)
    x_test = allfeatures[:len(data)]
    y_test = None
    y_train = label
    x_train = allfeatures[len(data):]
#     print(x_test)
    xgbModel(x_train,x_test,y_train,y_test,config,params)
    ypred = xgb_predict(x_test)
    for i,string in enumerate(data):
        print(string)
        print('Probability: ',ypred[i])
        
#-- xgb parameters
params={
	'booster':'gbtree',
# 	'objective': 'multi:softmax',
	'objective': 'binary:logistic',
	#'objective': 'rank:pairwise',
#     'eval_metric': 'merror',
    'eval_metric':['logloss','auc'],
	'stratified':True,
    
	'max_depth':8,
	'min_child_weight':0.01,
	'gamma':0.1,
	'subsample':0.6,
	'colsample_bytree':0.5,
	#'max_delta_step':1,
	#'colsample_bylevel':0.5,
	#'rate_drop':0.3,
	'scale_pos_weight':1,
	'lambda':0.0001,   #550
	'alpha':10,
	#'lambda_bias':0,
# 	'num_class':2,# 二分类的时候不需要这个参数！！
	'eta': 0.02,
	'seed':12,
	'nthread':24,
	'silent':1
}

config={
    'rounds':500,
    'folds':3,
    'multiClass':False
}

if __name__=='__main__':
    data,label = load_data()
    
#     data = preprocess_data(data)
    

#     x_train, x_test, y_train, y_test = get_allData()
#     xgbModel(x_train, x_test, y_train, y_test, config, params)

#     string=[
#         '为什么不能用了',
#         '你傻啊',
#         '很重要的聊天记录丢失',
#         '无法支付',
#         '出问题了',
#         '店的环境非常好，价格也便宜，值得推荐',
#         '质量很差',
#         '我要去北京天安门',
#         '手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了！'
#     ]
#     predict(string)
  
