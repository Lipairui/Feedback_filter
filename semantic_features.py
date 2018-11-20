# coding:utf-8

import time
import re
import jieba
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
from scipy.sparse import hstack
from matplotlib import pyplot
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from data_preprocess import *

#-- Log function
def LogInfo(stri):
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' '+stri)

#-- numpy array to pandas dataframe add columns name
def getColName(colNum, stri):
	LogInfo(str(colNum)+','+stri)
	colName = []
	for i in range(colNum):
		colName.append(stri + str(i))
	return colName


#-- Get pretrained word2vec model
def get_pretrained_w2vmodel():
    with open('../model/sgns.weibo.bigram-char') as file:
        lines = file.read().split('\n')
#     print(lines[0])
        w2vmodel = {}
        for line in lines[1:]:
            vec = line.split()       
            key = ''.join(vec[:-300])  
            w2vmodel[key]=list(map(float,vec[-300:]))
    return w2vmodel

#-- Get selftrained word2vec model
def get_selftrained_w2vmodel(documents,topicNum):
    # reconstruct corpus according to word frequency    
	min_word_freq = 20    
	texts = [[word for word in document.split(' ')] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
	    for token in text:
	        frequency[token] += 1
	texts = [[token for token in text if frequency[token] >= min_word_freq] for text in texts]
    
    # train word2vec model according to the corpus
	LogInfo('train word2vec Model...')
	w2vmodel = Word2Vec(texts, size=topicNum, window=5, iter = 15, min_count=min_word_freq, workers=12, seed = 12)#, sample = 1e-5, iter = 10,seed = 1)
# 	path = '../feature/'+str(topicNum)+'w2vModel.m'
# 	model.save(path)
	return w2vmodel

# Generate pretrained word2vec features
def get_pretrained_w2vfeatures(documents):
    # reconstruct corpus according to word frequency 
	documents = documents.astype(str).values
	texts = [[word for word in document.split(' ')] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1

	texts = [[token for token in text if frequency[token] >= 20] for text in texts]
    
    # get pretrained word2vec model
	print('get pretrained w2vmodel...')
	with open('../model/sgns.weibo.bigram-char') as file:
		lines = file.read().split('\n')
		w2vmodel = {}
		for line in lines[1:]:
			vec = line.split()       
			key = ''.join(vec[:-300])  
			w2vmodel[key]=list(map(float,vec[-300:]))
    
    # generate w2vFeatures
	topicNum = 300
	w2vFeature = np.zeros((len(texts), topicNum))
	w2vFeatureAvg = np.zeros((len(texts), topicNum))
	i = 0
	for line in texts:
		num = 0
		for word in line:
			num += 1
			
			if word in w2vmodel.keys():
				vec = w2vmodel[word]
			else:
				vec = np.zeros(topicNum)
			try:    
				w2vFeature[i, :] += vec
			except:
				w2vFeature[i, :] += np.zeros(topicNum)
		w2vFeatureAvg[i,:] = w2vFeature[i,:]/num
		i += 1        
	colName = getColName(topicNum, "vecT")
	w2vFeature = pd.DataFrame(w2vFeatureAvg, columns = colName)
	print(w2vFeature.shape)
	return w2vFeature



# Generate selftrained word2vec features
def get_selftrained_w2vfeatures(documents,topicNum):
    # reconstruct corpus according to word frequency    
	documents = documents.astype(str).values
	min_word_freq = 20    
	texts = [[word for word in document.split(' ')] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
	    for token in text:
	        frequency[token] += 1
	texts = [[token for token in text if frequency[token] >= min_word_freq] for text in texts]
    
    # train word2vec model according to the corpus
	LogInfo('train word2vec Model...')
	w2vmodel = Word2Vec(texts, size=topicNum, window=5, iter = 15, min_count=min_word_freq, workers=12, seed = 12)#, sample = 1e-5, iter = 10,seed = 1)
	path = '../feature/predict/'+str(topicNum)+'w2vModel.m'
	w2vmodel.save(path)
    
    # generate w2vFeatures
	w2vFeature = np.zeros((len(texts), topicNum))
	w2vFeatureAvg = np.zeros((len(texts), topicNum))
	i = 0
	for line in texts:
		num = 0
		for word in line:
			num += 1
			vec = w2vmodel[word]
			w2vFeature[i, :] += vec
		w2vFeatureAvg[i,:] = w2vFeature[i,:]/num
		i += 1 
	colName = getColName(topicNum, "vecT")
	w2vFeature = pd.DataFrame(w2vFeatureAvg, columns = colName)
	print(w2vFeature.shape)
	return w2vFeature


#-- doc2vec model function
def doc2vec(documents, topicNum):
	texts = []
	for i, document in enumerate(documents):
		word_list = document.split(' ')
		TaggededDocument = gensim.models.doc2vec.TaggedDocument
		document = TaggededDocument(word_list, tags=[i])
		texts.append(document)

	print(len(texts))

	model = Doc2Vec(texts, vector_size=topicNum, window=8,min_count=18, seed = 1)#, sample = 1e-5, iter = 10,seed = 1)
	LogInfo('d2v model finished!')
	doc2vecFeature = np.zeros((len(texts), topicNum))

	for i in range(len(texts)):
		
		vec = model.docvecs[i] 
		doc2vecFeature[i, :] = vec
	
	return doc2vecFeature

#-- get doc2vec feature
def getDoc2vecFeature(data,dim):

	vecFeature = doc2vec(data.astype(str).values, dim)
	colName = getColName(dim, "qvec")
	vecFeature = pd.DataFrame(vecFeature, columns = colName)	
	
	
# 	vecFeature['uid'] = data['uid'].values.T
	print(vecFeature.shape)
# 	name = '../feature/predict/d2vFeature'+str(dim)+'.csv'
# 	vecFeature.to_csv(name, index = False)
	return vecFeature

    #-- lsi model function
def lsi(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(len(texts)))
	dictionary = corpora.Dictionary(texts)
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' get corpus..')
	corpusD = [dictionary.doc2bow(text) for text in texts]
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' tfidf Model...')
	tfidf = TfidfModel(corpusD)
	corpus_tfidf = tfidf[corpusD]

	model = LsiModel(corpusD, num_topics=topicNum, chunksize=8000, extra_samples = 100)#, distributed=True)#, sample = 1e-5, iter = 10,seed = 1)

	lsiFeature = np.zeros((len(texts), topicNum))
	print('translate...')
	i = 0

	for doc in corpusD:
		topic = model[doc]
		
		for t in topic:
			 lsiFeature[i, t[0]] = round(t[1],5)
		i = i + 1
		if i%1000 == 1:
			print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(i))

	return lsiFeature

#-- get lsi feature
def getLsiFeature(data,dim):

	lsiFeature = lsi(data.astype(str).values, dim)
	colName = getColName(dim, "qlsi")
	lsiFeature = pd.DataFrame(lsiFeature, columns = colName)	
# 	lsiFeature['uid'] = data['uid'].values.T
	print(lsiFeature.shape)
# 	name = '../feature/lsiFeature'+str(dim)+'.csv'
# 	lsiFeature.to_csv(name, index = False)	
	return lsiFeature

#-- lda model function
def lda(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(len(texts)))
	dictionary = corpora.Dictionary(texts)
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' get corpus..')
	corpusD = [dictionary.doc2bow(text) for text in texts]

	#id2word = dictionary.id2word
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' tfidf Model...')
	tfidf = TfidfModel(corpusD)
	corpus_tfidf = tfidf[corpusD]
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' train lda Model...')
	ldaModel = gensim.models.ldamulticore.LdaMulticore(corpus_tfidf, workers = 8, num_topics=topicNum, chunksize=8000, passes=10, random_state = 12)
	#ldaModel = gensim.models.ldamodel.LdaModel(corpus=corpusD, num_topics=topicNum, update_every=1, chunksize=8000, passes=10)
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' get lda feature...')
	ldaFeature = np.zeros((len(texts), topicNum))
	i = 0

	for doc in corpus_tfidf:
		topic = ldaModel.get_document_topics(doc, minimum_probability = 0.01)
		
		for t in topic:
			 ldaFeature[i, t[0]] = round(t[1],5)
		i = i + 1
		if i%1000 == 1:
			print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(i))

	return ldaFeature

#-- get lda feature function
def getLdaFeature(data,dim):

	ldaFeature = lda(data.astype(str).values, dim)
	colName = getColName(dim, "qlda")
	ldaFeature = pd.DataFrame(ldaFeature, columns = colName)	
# 	ldaFeature['uid'] = data['uid'].values.T
	print(ldaFeature.shape)
# 	name = '../feature/ldaFeature'+str(dim)+'.csv'
# 	ldaFeature.to_csv(name, index = False)
	return ldaFeature

def generate_features(data,method):
    LogInfo('w2v starts!')
#     w2vFeature100 = getWord2vecFeature(data,300) #100
    w2vFeature100 = get_selftrained_w2vfeatures(data,100)
    LogInfo('w2v finishes!')

    LogInfo('d2v starts!')
    d2vFeature100 = getDoc2vecFeature(data,100)
    LogInfo('d2v finishes!')


    LogInfo('lsi starts!')
    lsiFeature100 = getLsiFeature(data,100)
    LogInfo('lsi finishes!')
    
    LogInfo('lda starts!')
    ldaFeature10 = getLdaFeature(data,10)
    LogInfo('lda finishes!')
    
    features = pd.concat([w2vFeature100,d2vFeature100,lsiFeature100,ldaFeature10],axis=1)
    features.to_csv('../feature/'+method+'/semantic_features1.csv',index=0)
    return features

def get_semantic_features_test():
    
    w2vFeature100 = pd.read_csv('../feature/test/w2vFeature100.csv')
    w2vFeatureAvg100 = pd.read_csv('../feature/test/w2vFeatureAvg100.csv')
    lsiFeature100 = pd.read_csv('../feature/test/lsiFeature100.csv')
    ldaFeature10 = pd.read_csv('../feature/test/ldaFeature10.csv')
    data = pd.concat([w2vFeature100,w2vFeatureAvg100,lsiFeature100,ldaFeature10],axis=1)
#     print(data.shape)
#     x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=42)
#     return x_train, x_test, y_train, y_test
    return data
    
    
def get_semantic_features_predict():
    w2vFeature100 = pd.read_csv('../feature/predict/pretrained_w2vFeature100.csv')
    w2vFeatureAvg100 = pd.read_csv('../feature/predict/pretrained_w2vFeatureAvg100.csv')
    d2vFeature100 = pd.read_csv('../feature/predict/d2vFeature100.csv')
    lsiFeature100 = pd.read_csv('../feature/predict/lsiFeature100.csv')
    ldaFeature10 = pd.read_csv('../feature/predict/ldaFeature10.csv')
    data = pd.concat([w2vFeature100,w2vFeatureAvg100,lsiFeature100,ldaFeature10],axis=1)
    return data

def predict():
    print('preprocess data...')
    x_train, y_train = load_data()
    x_test,test_id = load_testdata()
#     print(x_train.shape,x_test.shape)   
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)
    data = pd.concat([x_train,x_test],axis=0)  
    
    print('generate semantic features...')
    generate_features(data,'predict')  
    print('finish!')

def test():
    print('preprocess data...')
    data, label = load_data()
    data = preprocess_data(data)
    print('generate semantic features...')
    generate_features(data,'test')  
    print('finish!')
    
if __name__ == '__main__':
    predict()
