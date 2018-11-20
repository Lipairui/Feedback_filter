# coding:utf-8

import pandas as pd
import gensim, logging
import numpy as np
from collections import defaultdict
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
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
import numpy as np 
import xgboost as xgb 
import yaml
from scipy.sparse import hstack
from matplotlib import pyplot
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
from data_preprocess import *

#-- Log function
def LogInfo(stri):
	print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' '+stri)


def word2vec(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
	    for token in text:
	    	#token = int(token)
	    	#print token
	        frequency[token] += 1

	texts = [[token for token in text if frequency[token] >= 20] for text in texts]
    
	print('train Model...')
	model = Word2Vec(texts, size=topicNum, window=5, iter = 15, min_count=20, workers=12, seed = 12)#, sample = 1e-5, iter = 10,seed = 1)
	path = '../feature/'+str(topicNum)+'w2vModel.m'
	model.save(path)
	w2vFeature = np.zeros((len(texts), topicNum))
	w2vFeatureAvg = np.zeros((len(texts), topicNum))
	
	i = 0
	for line in texts:
		num = 0
		for word in line:
			num += 1
			vec = model[word]
			w2vFeature[i, :] += vec
		w2vFeatureAvg[i,:] = w2vFeature[i,:]/num
		i += 1
		if i%5000 == 0:
			print(i) 
	
	return w2vFeature, w2vFeatureAvg


def get_pretrained_w2vmodel():
    with open('../feature/sgns.weibo.bigram-char') as file:
        lines = file.read().split('\n')
#     print(lines[0])
        w2vmodel = {}
        for line in lines[1:]:
            vec = line.split()       
            key = ''.join(vec[:-300])  
            w2vmodel[key]=list(map(float,vec[-300:]))
        print(list(w2vmodel.keys())[:100])
    return w2vmodel

def lstm_test():
   
    dim = 100
    # preprocess for lstm
    print('preprocess data...')
    x_train, y_train = load_data()
#     print(x_train.shape,x_test.shape)   
    x_train = preprocess_data(x_train)
    texts = [[word for word in document.split(' ')] for document in x_train.astype(str).values]
#     texts, label = select_data(texts,label)
           
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            #token = int(token)
            #print token
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] >= 20] for text in texts]
    vocab = set([word for doc in texts for word in doc])
    vocab_size = len(vocab)
    
    print('generate embedding_matrix...')
    embedding_matrix = np.zeros((vocab_size,dim))
    word2index = {}
    model = Word2Vec.load('../feature/test/100w2vModel.m')
#     model = get_pretrained_w2vmodel()
    for i, word in enumerate(vocab):
        word2index[word] = i
        # if model is selftrained
        embedding_matrix[i] = model[word]
        # if model is pretrained
#         if word in model.keys():
#             try:
#                 embedding_matrix[i] = model[word]
#             except:
#                 print('error: ',word)
#         else:
#             embedding_matrix[i] = np.zeros(dim)
        
    print('generate encoded_texts...')
    encoded_texts = []
    for doc in texts:
        encoded_doc = []
        for word in doc:
            encoded_doc.append(word2index[word])
        encoded_texts.append(encoded_doc)
#     print(encoded_texts[:100])
#     max_length = max([len(doc) for doc in texts])
    max_length=config['max_length']
    print('generate padded_texts...')
    padded_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')
   
#     x_train = padded_texts[:len(x_train)]
#     x_test = padded_texts[len(x_train):]
    x_train, x_test, y_train, y_test = train_test_split(padded_texts, y_train, test_size=0.2,random_state=42)
    print(len(x_train),len(x_test),len(y_train))
    
    # lstm model structure
    print('Construct lstm model...')
    model = Sequential()
    embedding = Embedding(
        input_dim=vocab_size, 
        output_dim=dim, 
        mask_zero=True,
        weights=[embedding_matrix], 
        input_length=max_length,
        trainable=False)
    model.add(embedding)
    model.add(LSTM(units=50, activation='sigmoid', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile and train the model
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    skf = StratifiedKFold(y_train, n_folds=3, shuffle=True)
    new_train = np.zeros((len(x_train),1))
    new_test = np.zeros((len(x_test),1))

    for i,(trainid,valid) in enumerate(skf):
        print('fold' + str(i))
        train_x = x_train[trainid]
        train_y = y_train[trainid]
        val_x = x_train[valid]
        model.fit(train_x, train_y,
              batch_size=config['batch_size'],
              epochs=config['n_epoch'],verbose=1)
        new_train[valid] = model.predict_proba(val_x)
        new_test += model.predict_proba(x_test)

    new_test /= 3
    stacks = []
    stacks_name = []
    stack = np.vstack([new_train,new_test])
    stacks.append(stack)
    stacks = np.hstack(stacks)
    clf_stacks = pd.DataFrame(data=stacks,columns=['lstm'])
    clf_stacks.to_csv('../feature/test/lstm_prob1.csv',index=0)

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(document):
    data,label = load_data()
    data = preprocess_data(data)   
    texts = [[word for word in document.split(' ')] for document in data.values]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            #token = int(token)
            #print token
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] >= 20] for text in texts]
    vocab = set([word for doc in texts for word in doc])
    vocab_size = len(vocab)
    
    embedding_matrix = np.zeros((vocab_size,300))
    word2index = {}
    model = Word2Vec.load('../feature/100w2vModel.m')
    for i, word in enumerate(vocab):
        embedding_matrix[i] = model[word]
        word2index[word] = i
 
    
    encoded_texts = []
    for doc in document:
        encoded_doc = []
        for word in doc:
            if word in vocab:
                encoded_doc.append(word2index[word])
            else:
                encoded_doc.append(0)
        encoded_texts.append(encoded_doc)  
        
#     print(encoded_texts[:100])
#     max_length = max([len(doc) for doc in texts])
    max_length = config['max_length']
    
    padded_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')
    return padded_texts

def input_transform(strings):
    texts = []
    for string in strings:
        string = re.sub(u"[^\u4E00-\u9FFF]", "", string)
        punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
        string = re.sub(punctuation, "",string)
        string =' '.join(jieba.cut(string))
        string = re.sub('\s{2,}',' ',string)
        string = string.split()
        texts.append(string)
#     print(texts)
    texts = create_dictionaries(texts)
     
#     print(texts)
    return texts


def lstm_predict(strings):
    print('loading model......')
    max_length=config['max_length']
    path1 = '../lstm_data/lstm_len'+str(max_length)+'_epoch'+str(config['n_epoch'])+'.yml'
    path2 = '../lstm_data/lstm_len'+str(max_length)+'_epoch'+str(config['n_epoch'])+'.h5'
    with open(path1, 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights(path2)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(strings)
    for i,string in enumerate(strings):
        predict_data = data[i].reshape(1,-1)
        pro = model.predict(predict_data)[0][0]
        result=model.predict_classes(predict_data)
        print(string)
        if result[0][0]==1:
            print('probability:',pro,' positive')
        else:
            print('probability:',pro,' negative')
    
        

def lstm_predict():
    dim = 100
    # preprocess for lstm
    print('preprocess data...')
    x_train, y_train = load_data()
    x_test,test_id = load_testdata()
#     print(x_train.shape,x_test.shape)   
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)
    data = pd.concat([x_train,x_test],axis=0).astype(str)    
    texts = [[word for word in document.split(' ')] for document in data.values]
#     texts, label = select_data(texts,label)
           
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            #token = int(token)
            #print token
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] >= 20] for text in texts]
    vocab = set([word for doc in texts for word in doc])
    vocab_size = len(vocab)
    
    print('generate embedding_matrix...')
    embedding_matrix = np.zeros((vocab_size,dim))
    word2index = {}
    model = Word2Vec.load('../feature/predict/100w2vModel.m')
#     model = get_pretrained_w2vmodel()
    for i, word in enumerate(vocab):
        word2index[word] = i
        # if model is selftrained
        embedding_matrix[i] = model[word]
        # if model is pretrained
#         if word in model.keys():
#             try:
#                 embedding_matrix[i] = model[word]
#             except:
#                 print('error: ',word)
#         else:
#             embedding_matrix[i] = np.zeros(dim)
        
    print('generate encoded_texts...')
    encoded_texts = []
    for doc in texts:
        encoded_doc = []
        for word in doc:
            encoded_doc.append(word2index[word])
        encoded_texts.append(encoded_doc)
#     print(encoded_texts[:100])
#     max_length = max([len(doc) for doc in texts])
    max_length=config['max_length']
    print('generate padded_texts...')
    padded_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')
   
    x_train = padded_texts[:len(x_train)]
    x_test = padded_texts[len(x_train):]
#     x_train, x_test, y_train, y_test = train_test_split(padded_texts, label, test_size=0.2,random_state=42)
    print(len(x_train),len(x_test),len(y_train))
    
    # lstm model structure
    print('Construct lstm model...')
    model = Sequential()
    embedding = Embedding(
        input_dim=vocab_size, 
        output_dim=dim, 
        mask_zero=True,
        weights=[embedding_matrix], 
        input_length=max_length,
        trainable=False)
    model.add(embedding)
    model.add(LSTM(units=50, activation='sigmoid', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile and train the model
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    skf = StratifiedKFold(y_train, n_folds=3, shuffle=True)
    new_train = np.zeros((len(x_train),1))
    new_test = np.zeros((len(x_test),1))

    for i,(trainid,valid) in enumerate(skf):
        print('fold' + str(i))
        train_x = x_train[trainid]
        train_y = y_train[trainid]
        val_x = x_train[valid]
        model.fit(train_x, train_y,
              batch_size=config['batch_size'],
              epochs=config['n_epoch'],verbose=1)
        new_train[valid] = model.predict_proba(val_x)
        new_test += model.predict_proba(x_test)

    new_test /= 3
    stacks = []
    stacks_name = []
    stack = np.vstack([new_train,new_test])
    stacks.append(stack)
    stacks = np.hstack(stacks)
    clf_stacks = pd.DataFrame(data=stacks,columns=['lstm'])
    clf_stacks.to_csv('../feature/predict/lstm_prob2.csv',index=0)

config={
    'max_length':50,
    'batch_size':32,
    'n_epoch':10
}
if __name__=='__main__':
    
#     data,label = load_data()
#     data = preprocess_data(data)
    
#     LogInfo('w2v starts!')
#     getWord2vecFeature(data,100)
#     LogInfo('w2v finishes!')

#     LogInfo('d2v starts!')
#     getDoc2vecFeature(data,100)
#     LogInfo('d2v finishes!')


#     LogInfo('lsi starts!')
#     getLsiFeature(data,100)
#     LogInfo('lsi finishes!')
    
#     LogInfo('lda starts!')
#     getLdaFeature(data,10)
#     LogInfo('lda finishes!')

#     get_allData()
#     lstm_test()
    lstm_predict()

   
#     lstm_predict(string)
