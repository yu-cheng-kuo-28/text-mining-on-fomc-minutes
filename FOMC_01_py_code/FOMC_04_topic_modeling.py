# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 01:00:46 2020

@author: Morton
"""

#%% (1) Open it from the pickle-type files
import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_04_pickle_data')
os.getcwd()

import pickle
FOMC_pickle = list(range(0,222))
for i in range(0,222):
    name = str(i+1)+'.txt'
    with open(name, 'rb') as file:
        FOMC_pickle[i] = pickle.load(file)

len(FOMC_pickle[221])
print((FOMC_pickle[221])[0:30])
type(FOMC_pickle[221])
len(FOMC_pickle)

print(FOMC_pickle[0:20])


#%% (2) Functions

# https://bit.ly/36DJAWh
# https://www.kaggle.com/hj5992/nlp-topic-modelling-basics
# https://www.kaggle.com/miguelniblock/predict-the-author-unsupervised-nlp-lsa-and-bow


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_feature_matrix(documents, feature_type='frequency'):

    feature_type = feature_type.lower().strip()  
    print(feature_type)
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, 
                                     ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


from scipy.sparse.linalg import svds


# build_feature_matrix(corpus_nmf_02, feature_type='frequency')

    

def low_rank_svd(matrix, singular_count=2):
    
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt


import gensim
from gensim import corpora, models
import numpy as np

## pre-processing

def preprocessing_FOMC (word_lemma):
    
    #1. Dictionary of copora
    global dictionary
    dictionary = corpora.Dictionary(word_lemma)
    print(dictionary.token2id); print('\n')
    
    # 2. Convert the tokenized documents into bag of words vector
    corpus = [dictionary.doc2bow(text) for text in word_lemma]
    print(corpus)
    return corpus 

corpus = preprocessing_FOMC(FOMC_pickle); 
# print(corpus)



#%% (3) LSI

# 1. Build a tf-idf feature vectors
print(corpus)
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
print(corpus_tfidf)

# 2. Fix the number of topics
total_topics = 3

# 3. Build an LSI topic model

lsi = models.LsiModel(corpus_tfidf, id2word= dictionary, num_topics= total_topics)

for index, topic in lsi.print_topics(total_topics):
    print('Tpoic #'+str(index+1))
    print(topic); print('\n')


# 4. Build an LSI topic model with threshold
def print_topics_gensim(topic_model, total_topics=1, weight_threshold=0.00001,
                        display_weights = False, num_terms= 100):
    list_topics = []
    
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic= [(word, round(wt, 2)) for word, wt in topic if abs(wt) >= weight_threshold]
        
        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term,wt in topic]
            print(tw[:num_terms if num_terms else tw])
        print('\n')
        
        tw = [term for term,wt in topic]
        list_topics.append(tw[:num_terms if num_terms else tw])
        
    return list_topics
        

print_topics_gensim(topic_model = lsi, total_topics= total_topics, weight_threshold =0.00001,
                        display_weights = False, num_terms= 12)

print_topics_gensim(topic_model = lsi, total_topics= total_topics,
                        display_weights = True, num_terms= 12)


import pandas as pd
def get_lsi_topics(model, total_topics = 1, num_terms = 5 , weight_threshold = 0.001):
    word_dict = {};
    
    rank_topic = print_topics_gensim(topic_model = lsi, total_topics= total_topics,
                        display_weights = False, num_terms= num_terms,
                        weight_threshold= weight_threshold)
    
    for i in range(total_topics):
        words = rank_topic[i];
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [k for k in words];
    return pd.DataFrame(word_dict, index=['Term '+ str(i) for i in range(1,num_terms+1)])

get_lsi_topics(lsi, total_topics= 3, num_terms=10, weight_threshold=0.0001)



def get_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 60);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [k[0] for k in words];
    return pd.DataFrame(word_dict)

get_topics(lsi, num_topics=3)



#### Scikit-Learn

def preprocessing_test_2(toy_corpus):
    # 1. Lowering & sent_tokenize 
    from nltk.tokenize import word_tokenize
    toy_corpus_lower = [str(sent).lower() for sent in toy_corpus]
    norm_tokenized_corpus = [word_tokenize(text) for text in toy_corpus_lower]
    
    # 2. Deleting stopwords
    import nltk
    def remove_stopwords(tokens):
        stopword_list = nltk.corpus.stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return filtered_tokens
    
    word_stopwords = [remove_stopwords(sent) for sent in norm_tokenized_corpus]
    # print(word_stopwords); print('\n')
    
    # 3. Lemmatization 
    from nltk import WordNetLemmatizer
    def lemmatization(text):
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word in text]
        return lemmas
    
    word_lemma = [lemmatization(sent) for sent in word_stopwords]
    # print(word_lemma); print('\n')
    
    return word_lemma


# print(preprocessing_test_2(FOMC_pickle))
corpus_nmf = preprocessing_test_2(FOMC_pickle)



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF;
from sklearn.decomposition import TruncatedSVD

#the count vectorizer module needs string inputs, not array, so I join them with a space. This is a very quick operation.
corpus_nmf_02 = [' '.join(text) for text in corpus_nmf]
print(corpus_nmf_02[0:10])

vectorizer = CountVectorizer(analyzer='word', max_features=5000, max_df=1.0, min_df= 0.0);
print(vectorizer)
x_counts = vectorizer.fit_transform(corpus_nmf_02);
print(x_counts)

transformer = TfidfTransformer(smooth_idf=False);
print(transformer)
x_tfidf = transformer.fit_transform(x_counts);
print(x_tfidf)
type(x_tfidf)
x_tfidf.shape


xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
print(xtfidf_norm)
#obtain a LSI / SVD / LSA model
lsi_model = TruncatedSVD (n_components = 100, n_iter=5, random_state=42)
# model = NMF(n_components= 2, init='nndsvd', max_iter=10000, random_state=42, alpha=.1, l1_ratio=.5);
#fit the model
lsi_model.fit_transform(xtfidf_norm)
lsi_model.explained_variance_ratio_[0:100].sum()
lsi_model.explained_variance_ratio_[0:46].sum()


def explained_variance_ratio_threshold(lsi_model, threshold = 0.5):
    for i in range(0,100):
        global n
        n = i
        
        if lsi_model.explained_variance_ratio_[0:i].sum() > threshold:
            break
    return n

explained_variance_ratio_threshold(lsi_model, threshold = 0.7)


def get_topics_2(model, num_topics=1, n_top_words=10):
    
    lsi_model = TruncatedSVD (n_components = 2, n_iter=5, random_state=42)
    #fit the model
    lsi_model.fit_transform(xtfidf_norm)

    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
        
    print(word_dict)
    return pd.DataFrame(word_dict, index=['Term '+ str(i) for i in range(1,n_top_words+1)])

LSA = get_topics_2(lsi_model, num_topics=3, n_top_words=50); LSA

import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_06_output_excel')
os.getcwd()
LSA.to_csv("LSA_3topics.csv", index=True, header=True)



from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)  
TruncatedSVD(algorithm='randomized', n_components=5, n_iter=7,
        random_state=42, tol=0.0)
print(svd.explained_variance_ratio_)  
# [0.0606... 0.0584... 0.0497... 0.0434... 0.0372...]
print(svd.explained_variance_ratio_.sum())  
# 0.249...
print(svd.singular_values_)  
# [2.5841... 2.5245... 2.3201... 2.1753... 2.0443...]




#%% (4) LDA

corpus = preprocessing_FOMC(FOMC_pickle); 
print(corpus)

def train_lda_model_gensim(corpus, total_topics=2):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda= models.LdaModel(corpus_tfidf, id2word= dictionary, 
                         iterations =1000, num_topics= total_topics)
    return lda


lda_gensim = train_lda_model_gensim(corpus, total_topics = 3)

'''
print_topics_gensim(topic_model = lda_gensim, total_topics= total_topics,
                        display_weights = False, num_terms= 5)

print_topics_gensim(topic_model = lda_gensim, total_topics= total_topics,
                        display_weights = True, num_terms= 5)


print_topics_gensim(topic_model = lda_gensim, total_topics= total_topics,
                        display_weights = False, num_terms= 5,
                        weight_threshold= 0.01)

rank_topic = print_topics_gensim(topic_model = lda_gensim, total_topics= total_topics,
                        display_weights = False, num_terms= 5,
                        weight_threshold= 0.01)

rank_topic
'''

import pandas as pd
def get_lda_topics(model, total_topics = 1, num_terms = 5 , weight_threshold = 0.001):
    word_dict = {};
    
    rank_topic = print_topics_gensim(topic_model = lda_gensim, total_topics= total_topics,
                        display_weights = False, num_terms= num_terms,
                        weight_threshold= weight_threshold)
    
    for i in range(total_topics):
        words = rank_topic[i];
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [j for j in words];
    return pd.DataFrame(word_dict, index=['Term '+ str(i) for i in range(1,num_terms+1)])

get_lda_topics(lda_gensim, total_topics=3, num_terms=8, weight_threshold=0.001)



def train_lda_model_gensim(corpus, total_topics=2):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda= models.LdaModel(corpus_tfidf, id2word= dictionary, 
                         iterations =1000, num_topics= total_topics)
    return lda

lda_0 = train_lda_model_gensim(corpus, 3)

id2word = gensim.corpora.Dictionary(train_)
corpus = [id2word.doc2bow(text) for text in train_]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

def get_lda_topics_2(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 8);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict)



get_lda_topics_2(lda_gensim, num_topics=3)

lda_gensim = train_lda_model_gensim(corpus, total_topics = 3)

get_topics_2(lda_gensim, num_topics=3, n_top_words=100)

###########################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.

import scipy as sp
import sklearn
import sys
from nltk.corpus import stopwords
import nltk
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
import pickle
import re

data_text.iloc[1]['Text']
'I am used to the excellent ketchup popcorn seasoning from Canada. I tried this instead for a change. Big mistake. One use and it was in the trash. Tastes like burnt seasoning, no ketchup flavor at all.'

from nltk import word_tokenize
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
stop.update(['href','br'])
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

for idx in range(len(data_text)):
    data_text.iloc[idx]['Text'] = [word for word in tokenizer.tokenize(data_text.iloc[idx]['Text'].lower()) if word not in stop]
    
train_ = [value[0] for value in data_text.iloc[0:].values]
num_topics = 8


corpus = preprocessing_FOMC(FOMC_pickle); 

id2word = gensim.corpora.Dictionary(corpus)
corpus = [id2word.doc2bow(text) for text in train_]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

def get_lda_topics(model, num_topics, topn=200):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = topn);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict)

LDA = get_lda_topics(lda_0, num_topics=3, topn = 50); LDA

import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_06_output_excel')
os.getcwd()
LDA.to_csv("LDA_3topics.csv", index=True, header=True)



#%% (5) NMF

def preprocessing_test_2(toy_corpus):
    # 1. Lowering & sent_tokenize 
    from nltk.tokenize import word_tokenize
    toy_corpus_lower = [sent[0].lower() for sent in toy_corpus]
    norm_tokenized_corpus = [word_tokenize(text) for text in toy_corpus_lower]
    
    # 2. Deleting stopwords
    import nltk
    def remove_stopwords(tokens):
        stopword_list = nltk.corpus.stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return filtered_tokens
    
    word_stopwords = [remove_stopwords(sent) for sent in norm_tokenized_corpus]
    # print(word_stopwords); print('\n')
    
    # 3. Lemmatization 
    from nltk import WordNetLemmatizer
    def lemmatization(text):
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word in text]
        return lemmas
    
    word_lemma = [lemmatization(sent) for sent in word_stopwords]
    # print(word_lemma); print('\n')
    
    return word_lemma


# print(preprocessing_test_2(toy_corpus))
corpus_nmf = preprocessing_test_2(FOMC_pickle)
    

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF;

#the count vectorizer module needs string inputs, not array, so I join them with a space. This is a very quick operation.
corpus_nmf_02 = [' '.join(text) for text in corpus_nmf]
print(corpus_nmf_02[0:10])

vectorizer = CountVectorizer(analyzer='word', max_features=5000);
print(vectorizer)
x_counts = vectorizer.fit_transform(corpus_nmf_02);
print(x_counts)

transformer = TfidfTransformer(smooth_idf=False);
print(transformer)
x_tfidf = transformer.fit_transform(x_counts);
print(x_tfidf)

xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
print(xtfidf_norm)
#obtain a NMF model.
model = NMF(n_components= 2, init='nndsvd', max_iter=10000, random_state=42, alpha=.1, l1_ratio=.5);
#fit the model
model.fit_transform(xtfidf_norm)


def get_nmf_topics(model, num_topics=1, n_top_words=10):
    
    model = NMF(n_components= num_topics, init='nndsvd', max_iter=10000, random_state=42, alpha=.1, l1_ratio=.5);
    #fit the model
    model.fit_transform(xtfidf_norm)

    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
        
    print(word_dict)
    return pd.DataFrame(word_dict, index=['Term '+ str(i) for i in range(1,n_top_words+1)])

get_nmf_topics(model, num_topics=2, n_top_words=8)





