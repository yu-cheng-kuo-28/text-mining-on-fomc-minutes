# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 01:35:01 2021

@author: Morton
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 05:21:05 2021

@author: Morton
"""

#%% (1) Data Input and a Helper Function

import os
os.chdir(r'D:\G03_1\FOMC\FOMC_04_pickle_data')
os.getcwd()

import pickle
FOMC_pickle = list(range(0,222))
for i in range(0,222):
    name = str(i+1)+'.txt'    
    FOMC_pickle[i] = pickle.load(open(name, 'rb'))

import numpy as np
FOMC_words = np.zeros(222)
FOMC_vocabularies = np.zeros(222) 

for i in range(0,222):
    FOMC_words[i] = len(FOMC_pickle[i])
    FOMC_vocabularies[i] = len(set(FOMC_pickle[i]))


def cal(text):
    words = len(text)
    vocabularies = len(set(text))
    vocabularies_ratio = vocabularies / words
    
    print("Words:{}".format(words))
    print("Vocabularies:{}".format(vocabularies))
    print("TTR:{}".format(vocabularies_ratio))



import os
os.chdir(r'D:\G03_1\FOMC\FOMC_02_excel_data')
os.getcwd()

import pandas as pd
# df = pd.read_excel (r'Path where the Excel file is stored\File name.xlsx')
df = pd.read_excel(r'Fed_fund_rate_change.xlsx')
df.shape
df.columns
print (df.iloc[:,0])
df.info()

print (df)
print (df.iloc[:,0])
print (df.iloc[:,1])
print (df.iloc[:,5])
print (list(df.iloc[:,5]))

fund_rate_change = list(df.iloc[:,5])


up_index = []
down_index = []
unchanged_index = []
for i in range(24,246):
    if fund_rate_change[i] == 1:
        up_index.append(i-24)
    elif fund_rate_change[i] == -1:
        down_index.append(i-24)
    elif fund_rate_change[i] == 0:
        unchanged_index.append(i-24)
    else:
        print('Wrong !')

print(up_index)
print(len(up_index)); print(len(down_index)); print(len(unchanged_index))
print(len(up_index) + len(down_index) + len(unchanged_index))



#%% (2) TF (Term Frequenct)

# TF of different period

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd


#### 2-1 TF (ALL) of 1993 ~ 2020

All_words = []
for i in range(0,222):
    All_words += FOMC_pickle[i] 

cal(All_words)

All_words[0:10]
type(All_words)
type(All_words[0])


corpus_df = pd.DataFrame({'Document': FOMC_pickle, 
                          'Category': fund_rate_change[24:246]})
corpus_df = corpus_df[['Document', 'Category']]
print(corpus_df)


import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=20, ngram_range=(1,1))
sf = cvec.fit_transform(All_words)
transformer = TfidfTransformer(use_idf=False, smooth_idf=False, sublinear_tf=False)
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()

weights_2 = sorted(weights, reverse=True)
len(weights_2)
print(weights_2[400])

weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_All = weights_df.sort_values(by='weight', ascending=False)
print(TF_All[:20])
TF_All_2 = TF_All.iloc[:,0][:200]

TF_test = TF_All.iloc[:,0]

type(TF_All_2)
print(TF_All_2.shape)

print(TF_All_2.tolist()) # .tolist() / .toarray() / .tostring()
print(len(TF_All_2.tolist()))
# type(TF_All)
# print(TF_All.iloc[1,0])
# print(TF_All.iloc[1,1])
'''
import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_06_output_excel')
os.getcwd()
TF_All.to_csv("TF_All.csv", index=False, header=True)
'''




'''
heads = ['apple', 'banana', 'orange']
bodies = ['Science is good', 'Math is good', 'Engineering is good']
doc = heads + bodies
type(doc)
doc

All_words = heads + bodies
print(All_words)

cvec = CountVectorizer(stop_words='english', min_df=0, ngram_range=(1,1))
sf = cvec.fit_transform(All_words)
transformer = TfidfTransformer(use_idf=False, smooth_idf=True, sublinear_tf=False)
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()

weights_2 = sorted(weights, reverse=True)
print(weights_2[:20])

weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_All = weights_df.sort_values(by='weight', ascending=False)
print(TF_All[:20])
TF_All_2 = TF_All.iloc[:,0][:200]
type(TF_All_2)
print(TF_All_2.shape)

print(TF_All_2.tolist()) # .tolist() / .toarray() / .tostring()
print(len(TF_All_2.tolist()))
'''







#### 2-2 TF (Up) of 1993 ~ 2020

Up_words = []
for i in up_index:
    Up_words += FOMC_pickle[i]

cal(Up_words)


import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=20, ngram_range=(1,1))
sf = cvec.fit_transform(Up_words)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_Up = weights_df.sort_values(by='weight', ascending=False)
# print(TF_Up)


'''
import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_06_output_excel')
os.getcwd()
TF_All.to_csv("TF_All.csv", index=False, header=True)
'''


#### 2-3 TF (Down) of 1993 ~ 2020
Down_words = []
for i in down_index:
    Down_words += FOMC_pickle[i]

cal(Down_words)


import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=20, ngram_range=(1,1))
sf = cvec.fit_transform(Down_words)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_Down = weights_df.sort_values(by='weight', ascending=False)
# print(Down_words)
'''
import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_06_output_excel')
os.getcwd()
TF_All.to_csv("TF_All.csv", index=False, header=True)
'''



#### 2-4 TF (Unchanged) of 1993 ~ 2020
Unchanged_words = []
for i in unchanged_index:
    Unchanged_words += FOMC_pickle[i]

cal(Unchanged_words)

import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=20, ngram_range=(1,1))
sf = cvec.fit_transform(All_words)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_Unchanged = weights_df.sort_values(by='weight', ascending=False)
print(TF_Unchanged.iloc[:,0])

len(TF_Unchanged.iloc[:,0])

#### 2-5 TF (Down + Changed) of 1993 ~ 2020
Down_Unchanged_words = Down_words + Unchanged_words
cal(Down_Unchanged_words)

import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=20, ngram_range=(1,1))
sf = cvec.fit_transform(Down_Unchanged_words)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_Down_Unchanged = weights_df.sort_values(by='weight', ascending=False)
print(len(TF_Down_Unchanged))
# print(TF_Down_Unchanged)




#### 2-6 TF (Up + Changed) of 1993 ~ 2020
import os
os.chdir(r'D:\G03_1\FOMC\FOMC_06_output_excel')
os.getcwd()

Up_Unchanged_words = Up_words + Unchanged_words
cal(Up_Unchanged_words)


import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=20, ngram_range=(1,1))
sf = cvec.fit_transform(Up_Unchanged_words)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_Up_Unchanged = weights_df.sort_values(by='weight', ascending=False)
# print(TF_All)
'''
import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_06_output_excel')
os.getcwd()
TF_All.to_csv("TF_All.csv", index=False, header=True)
'''



#%% (3) Filtering by 1.5, 2, 3 Times

#### 3-1 Up / Down + Unchanged

cal(Up_words)
print(TF_Up.iloc[0:10,])
Up_threshold = 20/98018 # 0.00020404415515517559

print(TF_Up.iloc[0:10,1])

'''
up_count = 0
while TF_Up.iloc[:,1].iloc[up_count] > Up_threshold:
    up_count += 1
print(up_count) # 715
'''
print(len(TF_Up)) # 715


cal(Down_Unchanged_words)
print(TF_Down_Unchanged.iloc[0:10,])
Down_Unchanged_threshold = 20/535259 #  3.73650886766967e-05

535259 / 98018

down_change_count = 0
while TF_Down_Unchanged.iloc[:,1].iloc[down_change_count] > Down_Unchanged_threshold:
    down_change_count += 1
print(down_change_count) # 1419

print(len(TF_Down_Unchanged)) # 1448 ## ??



print(TF_Up.iloc[:,1].iloc[714])
TF_Up_2 = TF_Up.iloc[0:714,]

print(TF_Down_Unchanged.iloc[:,1].iloc[1418])
TF_Down_Unchanged_2 = TF_Down_Unchanged.iloc[0:1418,]



Up_words = []
Down_Unchanged_words = []
for i in range(0,715):
    for j in range(0,1419):
        if TF_Up.iloc[:,0].iloc[i] == TF_Down_Unchanged.iloc[:,0].iloc[j]:
            if TF_Up.iloc[:,1].iloc[i] >= (5) * TF_Down_Unchanged.iloc[:,1].iloc[j]:
                Up_words.append(TF_Up.iloc[:,0].iloc[i])
            elif TF_Up.iloc[:,1].iloc[i] <=(1/5) * TF_Down_Unchanged.iloc[:,1].iloc[j]:
                Down_Unchanged_words.append(TF_Up.iloc[:,0].iloc[i])
            else:
                pass

len(Up_words) # 27
print(Up_words)
'''
['hurrican', 'contain', 'upsid', 'remov', 'pass', 'owe', 'momentum', 'grew', 'save', 'buildup', 'evolv', 'fundament', 'curv', 'shortag', 'ga', 'flatten', 'neutral', 'remark', 'increasingli', 'constraint', 'discount', 'rebuild', 'unseason', 'warm', 'buoy', 'equal', 'workweek']
'''

len(Down_Unchanged_words) # 17
print(Down_Unchanged_words)
'''
['bank', 'credit', 'loan', 'purchas', 'weak', 'commerci', 'asset', 'decreas', 'euro', 'contract', 'eas', 'recoveri', 'weaken', 'weaker', 'estat', 'lend', 'institut']
'''


True or False
False or False




#### 3-2 Down / Up + Unchanged
cal(Down_words)
print(TF_Down.iloc[0:10,])
Down_threshold = 20/87416 # 0.0002287910679967054

print(TF_Down.iloc[0:10,1])

'''
down_count = 0
while TF_Down.iloc[:,1].iloc[down_count] > Down_threshold:
    down_count += 1
print(down_count) # 669
'''
print(len(TF_Down)) # 689 #??


cal(Up_Unchanged_words)
print(TF_Up_Unchanged.iloc[0:10,])
Up_Unchanged_threshold = 20/545861 #  3.6639364233751815e-05

545861 / 87416
'''
up_change_count = 0
while TF_Up_Unchanged.iloc[:,1].iloc[up_change_count] > Up_Unchanged_threshold:
    up_change_count += 1
print(up_change_count) # 1432
'''
print(len(TF_Up_Unchanged)) # 1432



TF_Down.iloc[:,1].iloc[668] > Down_threshold


print(TF_Down.iloc[:,1].iloc[668])
TF_Down_2 = TF_Down.iloc[0:668,]

print(TF_Up_Unchanged.iloc[:,1].iloc[1431])
TF_Up_Unchanged_2 = TF_Up_Unchanged.iloc[0:1431,]



Down_words = []
Up_Unchanged_words = []
for i in range(0,669):
    for j in range(0,1432):
        if TF_Down.iloc[:,0].iloc[i] == TF_Up_Unchanged.iloc[:,0].iloc[j]:
            if TF_Down.iloc[:,1].iloc[i] >= (5) * TF_Up_Unchanged.iloc[:,1].iloc[j]:
                Down_words.append(TF_Down.iloc[:,0].iloc[i])
            elif TF_Down.iloc[:,1].iloc[i] <= (1/5) * TF_Up_Unchanged.iloc[:,1].iloc[j]:
                Up_Unchanged_words.append(TF_Down.iloc[:,0].iloc[i])
            else:
                pass
   
len(Down_words) # 47
print(Down_words)
'''
['eas', 'weak', 'septemb', 'basi', 'action', 'liquid', 'reduct', 'weaken', 'deterior', 'deceler', 'strain', 'weaker', 'promot', 'soften', 'stimulu', 'facil', 'tech', 'stress', 'summer', 'excess', 'paper', 'tighter', 'function', 'intensifi', 'stem', 'board', 'coronaviru', 'downturn', 'act', 'turmoil', 'heighten', 'outbreak', 'taken', 'tension', 'correct', 'discount', 'pronounc', 'worsen', 'repo', 'intend', 'combin', 'disappoint', 'cutback', 'strike', 'approv', 'overhang', 'ampl']
'''

len(Up_Unchanged_words) # 8
print(Up_Unchanged_words)
'''
['purchas', 'expand', 'april', 'gradual', 'slack', 'program', 'hire', 'medium']
'''



#### 3-3  Down / Up
cal(Down_words)
print(TF_Down.iloc[0:10,])
Down_threshold = 20/87416 # 0.0002287910679967054

print(TF_Down.iloc[0:10,1])

down_count = 0
while TF_Down.iloc[:,1].iloc[down_count] > Down_threshold:
    down_count += 1
print(down_count) # 669

print(len(TF_Down)) # 689



cal(Up_words)
print(TF_Up.iloc[0:10,])
Up_threshold = 20/98018 # 0.00020404415515517559

87416 / 98018

print(TF_Up.iloc[0:10,1])

'''
up_count = 0
while TF_Up.iloc[:,1].iloc[up_count] > Up_threshold:
    up_count += 1
print(up_count) # 715
'''
print(len(TF_Up)) # 715



print(TF_Down.iloc[:,1].iloc[668])
TF_Down_2 = TF_Down.iloc[0:668,]

print(TF_Up.iloc[:,1].iloc[714])
TF_Up_2 = TF_Up.iloc[0:714,]


Down_words = []
Up_words = []
for i in range(0,669):
    for j in range(0,715):
        if TF_Down.iloc[:,0].iloc[i] == TF_Up.iloc[:,0].iloc[j]:
            if TF_Down.iloc[:,1].iloc[i] >= (5) * TF_Up.iloc[:,1].iloc[j]:
                Down_words.append(TF_Down.iloc[:,0].iloc[i])
            elif TF_Down.iloc[:,1].iloc[i] <= (1/5) * TF_Up.iloc[:,1].iloc[j]:
                Up_words.append(TF_Down.iloc[:,0].iloc[i])
            else:
                pass

len(Down_words) # 29
print(Down_words)
'''
['eas', 'credit', 'weak', 'bank', 'septemb', 'august', 'reduc', 'fell', 'drop', 'liquid', 'sharpli', 'reduct', 'weaken', 'contract', 'global', 'decreas', 'weaker', 'institut', 'advers', 'lend', 'fall', 'soften', 'stimulu', 'grade', 'sluggish', 'neg', 'insur', 'prefer', 'overnight']
'''

len(Up_words) # 16
print(Up_words)
'''
['rise', 'higher', 'tighten', 'expand', 'april', 'rais', 'gradual', 'close', 'underli', 'acceler', 'roughli', 'weather', 'upsid', 'margin', 'brisk', 'medium']
'''

40000/12

#%% (4) Top 30 of TF of All 

# 1993 ~ 1995
All_words = []
    
for i in range(0,24):
    All_words += FOMC_pickle[i] 

cal(All_words)
len(All_words)

cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,1))
sf = cvec.fit_transform(All_words)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TF_top30 = weights_df.sort_values(by='weight', ascending=False).head(30)
print(TF_top30)

import os
os.chdir('D:\\G03_1\\FOMC\\FOMC_06_output_excel')
os.getcwd()
TF_top30.to_csv("TF_top30_1993_1995.csv", index=False, header=True)




#%% (5) TF-IDF


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd


All_words = []
for i in range(0,222):
    All_words += FOMC_pickle[i] 

cal(All_words)

All_words[0:10]
type(All_words)
type(All_words[0])


type(' '.join(FOMC_pickle[0]))


corpus_df = pd.DataFrame({'Document': FOMC_pickle, 
                          'Category': fund_rate_change[24:246]})
corpus_df = corpus_df[['Document', 'Category']]
print(corpus_df.shape)
print(corpus_df.iloc[:,0])


FOMC_corpus = []
for k in FOMC_pickle:
    FOMC_corpus.append(' '.join(k))

print(len(FOMC_corpus))
print(len(FOMC_corpus[0]))



import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=2, max_df=0.99, ngram_range=(1,1))
sf = cvec.fit_transform(FOMC_corpus)
transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_dfidf = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})

TFIDF_All = weights_dfidf.sort_values(by='weight', ascending=False)
print(TFIDF_All[:20])
print(len(TFIDF_All))
TFIDF_All_2 = TFIDF_All.iloc[:,0][:200]

TFIDF_test = TFIDF_All.iloc[:,0]

type(TFIDF_All_2)
print(TFIDF_All_2.shape)

print(TFIDF_All_2.tolist()) # .tolist() / .toarray() / .tostring()
print(len(TFIDF_All_2.tolist()))




'''
# version 1
tfIdfVectorizer=TfidfVectorizer(min_df=3, max_df=0.9, norm='l2', use_idf=True, smooth_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(corpus_TFIDF_2)
df = pd.DataFrame(tfIdf[2].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(30))

# version 2
tfIdfTransformer = TfidfTransformer(use_idf=True, smooth_idf=True)
countVectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.9, ngram_range=(1,1))
wordCount = countVectorizer.fit_transform(corpus_TFIDF_2)
newTfIdf = tfIdfTransformer.fit_transform(wordCount)
'''


#%% Plottting of Document Classification


