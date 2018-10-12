# Feedback_filter
Filter noise feedback, text classification combining LSTM, TFIDF stacking, XGBoost, word2vec, LDA, LSI...

# Code description  
## Layer1    
data_preprocess.py:  Preprocess data       
tfidf_stacking.py:  Generate traditional models (lr, svc, mnb, bnb) stacking features based on Tfidf vectors       
semantic_features.py:  Generate semantic features based on Word2vec, Doc2vec, LDA, LSI      
my_lstm.py:  Generate LSTM stacking features     
## Layer2         
my_xgboost.py:  Train XGBoost model based on Layer1 features        
excute.py:  Excute project
