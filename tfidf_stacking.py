# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
import re
import jieba
# import data_preprocess 

# SKlearn classification models
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier

#cross validation
from scipy import sparse
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

#creat stacking features
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from data_preprocess import *

#Naive Bayes models
mnb_clf = MultinomialNB()
gnb_clf = GaussianNB()
bnb_clf = BernoulliNB()

#SVM_based models
sgd_clf = SGDClassifier(loss = 'hinge',penalty = 'l2', alpha = 0.0001,n_iter = 500, random_state = 42, verbose=1, n_jobs=256)

svc_clf = svm.LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.000001, C=0.5, multi_class='ovr', 
                         fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=1, random_state=None, max_iter=5000)

svm_clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, 
                  tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', random_state=None)

#Logistic Regression
lr_clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                            class_weight=None, random_state=None, solver='liblinear', max_iter=5000, 
                            multi_class='ovr', verbose=1, warm_start=False, n_jobs=256)


#-- Log function
def LogInfo(stri):
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' '+stri)
    
def tfidf_test():
    print('preprocess data...')
    data,label = load_data()
    data = preprocess_data(data)
    print('tfidf transform...')
    tf_vec = TfidfVectorizer(min_df=1, norm='l2',use_idf=True, sublinear_tf=True, smooth_idf=True)
    tf_vec.fit(data)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=42)

    tfidf = tf_vec.transform(data)
    X_train_tf = tf_vec.transform(x_train)
    X_test_tf = tf_vec.transform(x_test)

    # words = tf_vec.get_feature_names()
    # for i in range(len(data)):
    #     print('---Document %d ---' %(i))
    #     for j in range(len(words)):
    #         if tfidf[i,j]>1e-5:
    #             print(words[j],tfidf[i,j])
   
    ch2 = SelectKBest(chi2, k=1000)
    x_train = ch2.fit_transform(X_train_tf, y_train)
    x_test = ch2.transform(X_test_tf)
    method = 'test'
    generate_clf_features(lr_clf,'lr',x_train,x_test, y_train, method)
    generate_clf_features(svc_clf,'svc',x_train,x_test, y_train, method)
    generate_clf_features(mnb_clf,'mnb',x_train,x_test, y_train, method)
    generate_clf_features(bnb_clf,'bnb',x_train,x_test, y_train, method)

def tfidf_predict():
    print('preprocess data...')
    x_train, y_train = load_data()
    print(set(y_train))
    x_test,test_id = load_testdata()
    print(x_train.shape,x_test.shape,len(y_train))   
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)
    data = pd.concat([x_train,x_test])
    print('tfidf transform...')
    tf_vec = TfidfVectorizer(min_df=1, norm='l2',use_idf=True, sublinear_tf=True, smooth_idf=True)
    tf_vec.fit(data)
  
#     tfidf = tf_vec.transform(data)
    X_train_tf = tf_vec.transform(x_train)
    X_test_tf = tf_vec.transform(x_test)

    # words = tf_vec.get_feature_names()
    # for i in range(len(data)):
    #     print('---Document %d ---' %(i))
    #     for j in range(len(words)):
    #         if tfidf[i,j]>1e-5:
    #             print(words[j],tfidf[i,j])
   
    ch2 = SelectKBest(chi2, k=1000)
    x_train = ch2.fit_transform(X_train_tf, y_train)
    x_test = ch2.transform(X_test_tf)
    
    print('generate clf features...')
    method = 'predict'
    generate_clf_features(lr_clf,'lr',x_train,x_test, y_train, method)
    generate_clf_features(svc_clf,'svc',x_train,x_test, y_train, method)
    generate_clf_features(mnb_clf,'mnb',x_train,x_test, y_train, method)
    generate_clf_features(bnb_clf,'bnb',x_train,x_test, y_train, method)
    
def generate_clf_features(clf, clf_name, x_train, x_test, y_train, method):
    print('Creat '+clf_name+' features...')
    random_seed = 2016
    class_num = 2
    x = x_train
    y = [1 if i == 1 else 0 for i in y_train]
    skf = StratifiedKFold(y, n_folds=5, shuffle=True)

    new_train = np.zeros((x_train.shape[0],1))
    new_test = np.zeros((x_test.shape[0],1))

    for i,(trainid,valid) in enumerate(skf):
        print('fold ' + str(i))
        train_x = x_train[trainid]
        train_y = y_train[trainid]
        val_x = x_train[valid]
        clf.fit(train_x, train_y)
        if clf_name == 'svc':
            new_train[valid] = clf.decision_function(val_x).reshape(-1,1)
            new_test += clf.decision_function(x_test).reshape(-1,1)
        else:
            new_train[valid] = clf.predict_proba(val_x)[:,0].reshape(-1,1)
            new_test += clf.predict_proba(x_test)[:,0].reshape(-1,1)

    new_test /= 5
    stacks = []
    stacks_name = []
    print(len(new_train),len(new_test))
    stack = np.vstack([new_train,new_test])
    stacks.append(stack)
#     stacks_name += ['%s_%d'%(clf_name+'_',i) for i in range(class_num)]
    stacks_name = [clf_name]
    stacks = np.hstack(stacks)
    clf_stacks = pd.DataFrame(data=stacks,columns=stacks_name)
#     path = '../feature/stack/'+clf_name+'_prob.csv'
#     path = '../feature/predict/'+clf_name+'_prob1.csv'
   
    path = '../feature/'+method+'/'+clf_name+'_prob1.csv'
    clf_stacks.to_csv(path, index=0)

def sentiment_word_count_test():
    with open('../data/NTUSD_positive_simplified.txt','rb') as file:
        pos_words = file.read().decode('utf-16').split('\r\n')  
    with open('../data/NTUSD_negative_simplified.txt','rb') as file:
        neg_words = file.read().decode('utf-16').split('\r\n')    
        
    pos_words_count = []
    neg_words_count = []
    
    data, label = load_data()   
    data = preprocess_data(data)
    documents = data.astype(str).values
    texts = [[word for word in document.split(' ')] for document in documents]
    
    for document in texts:
        pos_cnt = 0
        neg_cnt = 0
        for word in document:
            if word in set(pos_words):
                pos_cnt += 1
            if word in set(neg_words):
                neg_cnt += 1
        pos_words_count.append(pos_cnt)
        neg_words_count.append(neg_cnt)
    sentiment = pd.DataFrame(columns=['pos_words_count','neg_words_count'])
    sentiment.pos_words_count = pos_words_count
    sentiment.neg_words_count = neg_words_count
    sentiment_x_train, sentiment_x_test, y_train, y_test = train_test_split(sentiment, label, test_size=0.2,random_state=42)
    print(sentiment.head())
    sentiment =pd.concat([sentiment_x_train,sentiment_x_test],axis=0).reset_index()
    del sentiment['index']
    print(sentiment.shape,sentiment.head())
    sentiment.to_csv('../feature/test/sentiment_words_count.csv',index=0)

def sentiment_word_count_predict():
    print('prepare sentiment dictionary...')
    with open('../data/NTUSD_positive_simplified.txt','rb') as file:
        pos_words = file.read().decode('utf-16').split('\r\n')  
    with open('../data/NTUSD_negative_simplified.txt','rb') as file:
        neg_words = file.read().decode('utf-16').split('\r\n')    
    
    
    
    print('preprocess data...')
    x_train, y_train = load_data()
    x_test,test_id = load_testdata()
#     print(x_train.shape,x_test.shape)   
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)
    data = pd.concat([x_train,x_test],axis=0)      
    documents = data.astype(str).values
    texts = [[word for word in document.split(' ')] for document in documents]
    
    print('calculate sentiment words count...')
    pos_words_count = []
    neg_words_count = []
    i = 0
    for document in texts:
        pos_cnt = 0
        neg_cnt = 0
        if i%100 == 0:
            print(i)
        i = i+1
        for word in document:
            if word in set(pos_words):
                pos_cnt += 1
            if word in set(neg_words):
                neg_cnt += 1
        pos_words_count.append(pos_cnt)
        neg_words_count.append(neg_cnt)
    sentiment = pd.DataFrame(columns=['pos_words_count','neg_words_count'])
    sentiment.pos_words_count = pos_words_count
    sentiment.neg_words_count = neg_words_count
     
#     sentiment_x_train, sentiment_x_test, y_train, y_test = train_test_split(sentiment, label, test_size=0.2,random_state=42)
    
    print(sentiment.shape,sentiment.head())
    print('save result...')
    sentiment.to_csv('../feature/predict/sentiment_words_count1.csv',index=0)
    
def words_count_test():
    print('preprocess data...')
    data, label = load_data()   
    data = preprocess_data(data)
    documents = data.astype(str).values
    texts = [[word for word in document.split(' ')] for document in documents]
    print('calculate words count...')
    words_count = [len(document) for document in texts]
    words_x_train, words_x_test, y_train, y_test = train_test_split(words_count, label, test_size=0.2,random_state=42)
    words_count = words_x_train+words_x_test
    words_count = pd.DataFrame(data=words_count,columns=['words_count'])
    print('save result...')
    words_count.to_csv('../feature/test/words_count1.csv',index=0)

def words_count_predict():
    print('preprocess data...')
    x_train, y_train = load_data()
    x_test,test_id = load_testdata()
#     print(x_train.shape,x_test.shape)   
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)
    data = pd.concat([x_train,x_test],axis=0)
    documents = data.astype(str).values
    texts = [[word for word in document.split(' ')] for document in documents]
    print('calculate words count...')
    words_count = [len(document) for document in texts]
#     words_x_train, words_x_test, y_train, y_test = train_test_split(words_count, label, test_size=0.2,random_state=42)

    words_count = pd.DataFrame(data=words_count,columns=['words_count'])
    print('save result...')
    words_count.to_csv('../feature/predict/words_count1.csv',index=0)
    
def predict():
#     sentiment_word_count_predict()
    words_count_predict()
    tfidf_predict()

def test():
    words_count_test()
    tfidf_test()
    
if __name__ == '__main__':
#     test()
    predict()

