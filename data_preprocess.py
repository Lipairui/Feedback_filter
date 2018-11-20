import json
import pandas as pd
import re
import numpy as np
import jieba

def transform_data_style(path):
    with open(path) as file:
        lines = file.read().split('\n')
        lines.pop() # 去掉最后一个空格
        # 每个测试用例共八个元素
        # 总共41815个测试用例
    #     print(lines[:8])
    #     print(len(lines),(len(lines))/8)
    test_id = []
    ref = []
    hyp = []
    errors_count = []
    words_count = []
    for i,line in enumerate(lines):
        # id
        if i % 8 == 0:
            line = line.split()[1]
            test_id.append(line)
        # error count & words count
        if i % 8 == 1:
            line = list(map(int,line.split()[-4:]))
            words_count.append(sum(line))
            errors_count.append(sum(line)-line[0])    
        # REF text
        if i % 8 == 2:
            line = ''.join(line.split()[1:])
            ref.append(line)      
        # HYP text
        if i % 8 == 3:
            line = ''.join(line.split()[1:])
            hyp.append(line)

    original_data = pd.DataFrame(columns=['id','REF','HYP','words_count','errors_count','WER'])
    original_data.id = test_id
    original_data.REF = ref
    original_data.HYP = hyp
    original_data.words_count = words_count
    original_data.errors_count = errors_count
    original_data.WER = original_data.errors_count/original_data.words_count
    print(original_data.head())
    original_data.to_csv('../data/wechat_original_test2.csv',index=0)
    return original_data


# def load_testdata():
#     data = pd.read_csv('../data/wechat_original_test2.csv')
#     data_id = pd.concat([data.id,data.id],axis=0)
#     data = pd.concat([data.REF,data.HYP],axis=0)
#     return data,data_id

def load_testdata():
#     data = pd.read_excel('../data/wechat_team_test.xlsx',index=None)
    data = pd.read_excel('../data/top_stories.xlsx',index=None)
#     data['text_length'] = data.comment.apply(len)
#     print(data.text_length.mean())
    return data.comment,data.uin
    
    
# def load_data():
#     neg=pd.read_excel('../data/neg.xls',header=None,index=None)
#     pos=pd.read_excel('../data/pos.xls',header=None,index=None)
# #     print(neg.head(),pos.head())
#     data = pd.concat([neg[0],pos[0]])
# #     print(data.head())
#     print(data.apply(len).mean())
#     y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))  
# #     print(y)

#      # add extra training data
#     train = pd.read_excel('../data/wechat_team_train.xlsx')
#     train_data = train[:399][train.Label!= 2].comment
#     train_label = train[:399][train.Label!=2].Label.values
#     data = pd.concat([data,train_data])
#     y = np.concatenate((y,train_label))
#     print('train data: ',data.shape,len(y))  
  
#     return data,y

def load_data():
    noise = get_noise() # 0
    clean = get_clean() # 1
    data = pd.concat([noise,clean])
    y = np.concatenate((np.zeros(len(noise),dtype=int), np.ones(len(clean),dtype=int)))  
    
     # add extra training data
    train = pd.read_excel('../data/wechat_team_train.xlsx')
    train[:1500].label = train[:1500].label.fillna(1)

    data = pd.concat([data,train[:1500].comment])
    y = np.concatenate((y,train[:1500].label.values))
    print('train data: ',data.shape,len(y))  
    
    return data,y
# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    words = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
#         if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
        if i.word not in stopkey and i.flag in pos:
            words.append(i.word)
    return ' '.join(words)

def preprocess_data(data):
    # clean data
    data = data.apply(lambda x:re.sub(u"[^\u4E00-\u9FFF]", "", x))
    punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    data = data.apply(lambda x:re.sub( punctuation, "",x)) #清除所有标点符号
    # tokenize 
    data = data.apply(lambda x:' '.join(jieba.cut(x)))
    data = data.apply(lambda x: re.sub('\s{2,}',' ',x))
 
    # delete stop words
    file = open('../data/chinese_stopwords.txt','r',encoding='utf-8')
    stopwords = file.read().split('\n')

    stopwords = set(stopwords) # set比list查找速度更快！！
    data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
#     save data
    data.to_csv('../data/preprocess_data.csv',index=False)
#     data = pd.read_csv('../data/preprocess_data.csv',header=0, sep = ",",encoding='utf-8')
   
    return data
    

# noise data
def get_noise():
    data = open('../data/wechat_pay_mp.json').read()
    data = json.loads(data)
    train_data = []
    for item in data:
        train_data.append(item['_source']['comment'])
    # print(len(train_data)) 15374
    noise = pd.DataFrame({'comment':train_data})
    noise.comment.to_csv('../data/noise_data.csv',index=0)
    return noise.comment

# clean data
def get_clean():
    data = pd.read_excel('../data/wechat_pay.xlsx')
    clean = pd.DataFrame({'comment':data.comment.values})
    clean.comment.to_csv('../data/clean_data.csv',index=0)
    return clean.comment

if __name__ == '__main__':
    load_data()
