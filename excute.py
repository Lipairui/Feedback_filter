from my_xgboost import *
from data_preprocess import *
% matplotlib inline

def test():
    
    bnb_feature = pd.read_csv('../feature/test/bnb_prob1.csv')
    svc_feature = pd.read_csv('../feature/test/svc_prob1.csv')
    mnb_feature = pd.read_csv('../feature/test/mnb_prob1.csv')
    lr_feature = pd.read_csv('../feature/test/lr_prob1.csv')
    lstm_feature = pd.read_csv('../feature/test/lstm_prob1.csv')
#     sentiment_feature = pd.read_csv('../feature/sentiment_words_count.csv')
    words_count_feature = pd.read_csv('../feature/test/words_count1.csv')
    # cnn_feature = pd.read_csv('../feature/stack/cnn_prob.csv')
    basic_features = pd.concat([bnb_feature,svc_feature,mnb_feature,lr_feature,lstm_feature,words_count_feature],axis=1)

    _, label = load_data()
    semantic_features = pd.read_csv('../feature/test/semantic_features1.csv')
    semantic_x_train, semantic_x_test, y_train, y_test = train_test_split(semantic_features, label, test_size=0.2,random_state=42)
    semantic_x_train = semantic_x_train.reset_index()
    del semantic_x_train['index']
    semantic_x_test = semantic_x_test.reset_index()
    del semantic_x_test['index']
    basic_x_train = basic_features[:len(y_train)]
    basic_x_test = basic_features[len(y_train):]
    basic_x_test = basic_x_test.reset_index()
    del basic_x_test['index']

    x_train = pd.concat([semantic_x_train,basic_x_train],axis=1)
    x_test = pd.concat([semantic_x_test,basic_x_test],axis=1)
    print(len(x_train),len(basic_x_train),len(semantic_x_train))
    xgbModel(basic_x_train, basic_x_test, y_train, y_test, config, params,_,'test')

def predict():
    
    test_file = '../data/top_stories.xlsx'
    _, label = load_data()  
    bnb_feature = pd.read_csv('../feature/predict/bnb_prob1.csv')
    svc_feature = pd.read_csv('../feature/predict/svc_prob1.csv')
    mnb_feature = pd.read_csv('../feature/predict/mnb_prob1.csv')
    lr_feature = pd.read_csv('../feature/predict/lr_prob1.csv')
    lstm_feature = pd.read_csv('../feature/predict/lstm_prob2.csv')
#     sentiment_feature = pd.read_csv('../feature/predict/sentiment_words_count1.csv')
    words_count_feature = pd.read_csv('../feature/predict/words_count1.csv')
    semantic_features = pd.read_csv('../feature/predict/semantic_features2.csv')
    # cnn_feature = pd.read_csv('../feature/stack/cnn_prob.csv')
    features = pd.concat([bnb_feature,svc_feature, mnb_feature,lr_feature,words_count_feature,lstm_feature,semantic_features],axis=1)
    print(features)
    features.to_excel('../feature/predict/features.xls',index=0)
    x_train = features[:len(label)]
    x_test = features[len(label):].reset_index()
    del x_test['index']
    print('Features dimensions: ',len(features.columns))
    print('Train data: ',len(x_train))
    print('Test data: ',len(x_test))
    print('XGboost starts...')
    xgbModel(x_train, x_test, label, _, config, params,test_file,'predict')
#     print('predicting...')
#     model = xgb.Booster({'n_thread':24})
#     model.load_model('../model/predict/xgb06.m')
#     dtest = xgb.DMatrix(x_test)
#     ptest = model.predict(dtest)
#     y_pred = [1 if i > 0.5 else 0 for i in ptest]
#     print('finish predicting!')
#     data = pd.read_csv('../data/wechat_original_test.csv')
#     res = pd.DataFrame(columns=['id','REF','REF_sentiment_prob','REF_sentiment','HYP','HYP_sentiment_prob','HYP_sentiment'])
#     res.id = data.id
#     res.REF = data.REF 
#     res.REF_sentiment_prob = ptest[:len(data)]
#     res.REF_sentiment = y_pred[:len(data)]
#     res.HYP = data.HYP
#     res.HYP_sentiment_prob = ptest[len(data):]
#     res.HYP_sentiment = y_pred[len(data):]
#     print('result:')
#     print(res.shape,res.head())
#     save_path = '../res/res01.csv'
#     print('save result: '+save_path)
#     res.to_csv(save_path,index=0)

if __name__  == '__main__':
    predict()
    
