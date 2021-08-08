# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:49:09 2021

@author: Morton
"""

#%% (1) Inputting Fed Funds Rate Data Change Made by FOMC

import os
os.chdir(r'D:\G03_1\FOMC\FOMC_02_excel_data')
os.getcwd()

import pandas as pd
df = pd.read_excel(r'Fed_fund_rate_change.xlsx')
print(df.shape)
print(df.columns)
print (df.iloc[:,0])

print(df.info())
print(df.head(n=3))

import datetime
df.index = df.Date

print(df.info())
print(df.head(n=3))

print (df)
print (df.iloc[:,0])
print (df.iloc[:,1])
print (df.iloc[:,5])
print (list(df.iloc[:,5]))

fund_rate_change = list(df.iloc[:,5])
print(df.iloc[:,5].head(n=4))



#%% (2) Inputting the Real Fed Funds Rate Data Change

import os
os.chdir(r'D:\G03_1\FOMC\FOMC_02_excel_data')
os.getcwd()

import pandas as pd
real_rate = pd.read_csv("FEDFUNDS_19930101_20201001.csv", sep=",")

real_rate.info()

# import datetime
real_rate.index = real_rate.DATE

real_rate.info()
real_rate.head(n=3)



#%% (3) Inputting Corpus of FOMC Minutes

import os
os.chdir(r'D:\G03_1\FOMC\FOMC_04_pickle_data')
os.getcwd()

import pickle
FOMC_pickle = list(range(0,222))
for i in range(0,222):
    name = str(i+1)+'.txt'
    with open(name, 'rb') as file:
        FOMC_pickle[i] = pickle.load(file)

import numpy as np
FOMC_words = np.zeros(222)
FOMC_vocabularies = np.zeros(222) 

for i in range(0,222):
    FOMC_words[i] = len(FOMC_pickle[i])
    FOMC_vocabularies[i] = len(set(FOMC_pickle[i]))

print(len(FOMC_pickle))

import numpy as np
FOMC_words = np.array(FOMC_words)
FOMC_vocabularies = np.array(FOMC_vocabularies)




#%% (4) Preparing the Indices of Up, Down & Unchanged

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


import numpy as np
up_index = np.array(up_index)
down_index = np.array(down_index)
unchanged_index = np.array(unchanged_index)




#%% (5) Time Series Analysis of Real Fed Funds Rate Change 

#### 5-1 Line Chart

import os
os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.figure(figsize=(22,6))
plt.grid()

mpl.rc('xtick', labelsize=13.5) 
mpl.rc('ytick', labelsize=15) 
plt.xticks(range(0, 333, 25)) 
plt.plot(real_rate.FEDFUNDS, c='blue')

plt.ylabel('Percent', fontsize = 18)
plt.title('Effective Federal funds rate (1993/01/01 ~ 2020/10/01)', fontsize = 18)
plt.axis([-2, 336, -0.2, 6.7]) 

plt.savefig('01_Real_Rate_Change.png', dpi= 1000, bbox_inches='tight')
plt.close()


#### 5-2 Stationarity & Differencing

from pmdarima.arima import ndiffs as ndiffs

# test =  (‘kpss’, ‘adf’, ‘pp’)

print('KPSS: d =', ndiffs(real_rate.FEDFUNDS, alpha=0.05, test='kpss', max_d=2)) # d = 1. Indicating non-stationary sequence
print('ADF: d =', ndiffs(real_rate.FEDFUNDS, alpha=0.05, test='adf', max_d=2)) # d = 0. Indicating stationary sequence
print('PP: d =', ndiffs(real_rate.FEDFUNDS, alpha=0.05, test='pp', max_d=2)) # d = 0. Indicating stationary sequence


#### 5-3 ACF

# ACF of statsmodels
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib as mpl

mpl.rc("figure", figsize=(18,4))
plt.rc('font', size=15) 
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 

plot_acf(real_rate.FEDFUNDS, lags = 100)
plt.savefig('02_Real_Rate_ACF.png', dpi= 1000, bbox_inches='tight')
plt.close()


#### 5-4 PACF

from statsmodels.graphics.tsaplots import plot_pacf

mpl.rc("figure", figsize=(18,4))
plt.rc('font', size=15) 
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 

plot_pacf(real_rate.FEDFUNDS, lags=100)
plt.savefig('03_Real_Rate_PACF.png', dpi= 1000, bbox_inches='tight')
plt.close()


#### 5-5 Automatic ARIMA

import pmdarima as pm
model = pm.auto_arima(real_rate.FEDFUNDS, seasonal=True, m=12, suppress_warnings=True, trace=True, information_criterion='aic')
# Best ARIMA model: ARIMA(1,1,2)(0,0,0)[12]
    

#### 5-6 Checking the Best ARIMA Model: ARIMA(1,1,2)(0,0,0)[12]

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(real_rate.FEDFUNDS, exog=None, order=(1,1,2), seansonal_order = (0,0,0,12),
              enforce_stationary=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())


#### 5-7 Ljung-Box Test (Lack of Fit Test)
import statsmodels.api as sm

type(model_fit.resid)
model_fit.resid.plot()
sm.stats.acorr_ljungbox(model_fit.resid, lags=[12], return_df=True)

#      lb_stat  lb_pvalue
# 12  3.089372   0.994879

sm.stats.acorr_ljungbox(model_fit.resid, return_df=True)

'''
      lb_stat  lb_pvalue
1    0.003823   0.950701
2    0.035098   0.982604
3    0.333857   0.953540
4    0.339116   0.987151
5    0.339459   0.996834
6    0.977963   0.986434
7    0.980891   0.995131
8    2.237250   0.972861
9    2.239501   0.987089
10   2.249309   0.994048
11   2.834815   0.992734
12   3.089372   0.994879
13   3.123956   0.997449
14   3.278362   0.998470
15   3.285012   0.999298
16   4.259695   0.998379
17   4.570498   0.998748
18   5.182909   0.998546
19   5.242705   0.999198
20   5.252948   0.999594
21   5.643275   0.999647
22   9.376259   0.991133
23  10.077286   0.990789
24  10.206255   0.993643
'''


#%% (6) EDA of Fed Funds Rate 

#### 6-1 Pie chart of Fed fund rate

## Import Chinese font
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()


fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["Down (13.51%) (30/222)",
          "Up (17.57%) (39/222)",
          "Unchanged (68.92%) (153/222)"]

data = [30, 39, 153]
myexplode = [0.05, 0.05, 0.15]

wedges, texts = ax.pie(data, wedgeprops=dict(width=1.0), startangle=-40, explode = myexplode)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), fontsize="x-large",
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Proportions of Fed Funds Rate Changes", fontsize="xx-large")

plt.savefig('04_Rate_Pie_Chart_0.png', dpi= 1000, bbox_inches='tight')
plt.close()


## Chinese
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["降息樣本 (13.51%) (30/222)",
          "升息樣本 (17.57%) (39/222)",
          "利率不變樣本 (68.92%) (153/222)"]

data = [30, 39, 153]
myexplode = [0.05, 0.05, 0.15]

wedges, texts = ax.pie(data, wedgeprops=dict(width=1.0), startangle=-40, explode = myexplode)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), fontsize="x-large",
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("FOMC Minutes 中升息、降息、利率不變樣本的占比", fontsize="xx-large")

plt.savefig('04_Rate_Pie_Chart_1_Chinese.png', dpi= 1000, bbox_inches='tight')
plt.close()

#%% (8) Line Chart of Word, Voc, TTR of FOMC Minutes Combining with Fed Funds Rate 

#### 8-1 Word counts line chart

import os
os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()
import matplotlib.pyplot as plt


plt.figure(figsize=(15, 5))
plt.grid()

plt.plot(up_index+1 ,  FOMC_words[up_index], 'o', color= 'blue', label="Up")  # up_index 
plt.plot(down_index+1,  FOMC_words[down_index], 's', color= 'crimson', label="Down")  # down_index 
plt.plot(unchanged_index+1,  FOMC_words[unchanged_index], '*', color= 'darkgreen', label="Unchanged")  # unchanged_index 
# Draw 3 kinds of markers and change their colors.
# 'o' is circle marker;'s' is square marker; '*' star marker. 
# Add labels and plt.legend() will catch them later
plt.plot(list(range(1,223)), FOMC_words, '--k')


# plt.xlabel('1993/01/01 ~ 2020/10/01', fontsize = 15)
plt.ylabel('Word Counts', fontsize = 'large')
plt.title('Word Counts of FOMC Minutes (1993/01/01 ~ 2020/10/01)', fontsize = 'large')
plt.axis([-1, 224, 0, 4800]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'])  
plt.yticks(range(0, 5001, 500))

plt.axvspan(66, 71, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(120, 132, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(217, 222, color='lightblue', alpha=0.5, lw=0)

crisis_data = [
    (66, '2001/03: Peak of Dot-Com Bubble'),
    (120, '2007/12: Peak of Financial Crisis'),
    (217, '2020/02: Peak of COVID-19')]
x, label = crisis_data[0]
plt.annotate(label, xy=(x, FOMC_words[x] - 550),
             xytext= (x, FOMC_words[x] - 1150),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 'large')
x, label = crisis_data[1]
plt.annotate(label, xy=(x, FOMC_words[x] - 1150),
             xytext= (x, FOMC_words[x] - 1750),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='left', verticalalignment= 'top', fontsize = 'large')
x, label = crisis_data[2]
plt.annotate(label, xy=(x, FOMC_words[x] - 1200),
             xytext= (x, FOMC_words[x] - 1700),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 'large')
plt.legend(fontsize = 14, loc = 'lower right') # Indicate the labed markers
'''
1. fontsize: Either an relative value of 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large' 
or an absolute font size, e.g., 12.
'''
plt.savefig('05_FOMC_Word_Counts_0.png', dpi= 1000, bbox_inches= 'tight') 
plt.close()



## Chinese

import os
os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 5))
plt.grid()

plt.plot(up_index+1 ,  FOMC_words[up_index], 'o', color= 'blue', label="升息")  # up_index 
plt.plot(down_index+1,  FOMC_words[down_index], 's', color= 'crimson', label="降息")  # down_index 
plt.plot(unchanged_index+1,  FOMC_words[unchanged_index], '*', color= 'darkgreen', label="利率不變")  # unchanged_index 
# Draw 3 kinds of markers and change their colors.
# 'o' is circle marker;'s' is square marker; '*' star marker. 
# Add labels and plt.legend() will catch them later
plt.plot(list(range(1,223)), FOMC_words, '--k')


# plt.xlabel('1993/01/01 ~ 2020/10/01', fontsize = 15)
plt.ylabel('字詞數 (word counts)', fontsize = 16)
plt.title('FOMC Minutes 的字詞數 (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-1, 224, 0, 4800]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'], fontsize = 14.5)  
plt.yticks(range(0, 5001, 500), fontsize = 14.5)

plt.axvspan(66, 71, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(120, 132, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(217, 222, color='lightblue', alpha=0.5, lw=0)

crisis_data = [
    (66, '2001/03: Peak of Dot-Com Bubble'),
    (120, '2007/12: Peak of Financial Crisis'),
    (217, '2020/02: Peak of COVID-19')]
x, label = crisis_data[0]
plt.annotate(label, xy=(x, FOMC_words[x] - 550),
             xytext= (x, FOMC_words[x] - 1150),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16)
x, label = crisis_data[1]
plt.annotate(label, xy=(x, FOMC_words[x] - 1150),
             xytext= (x, FOMC_words[x] - 1750),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='left', verticalalignment= 'top', fontsize = 16)
x, label = crisis_data[2]
plt.annotate(label, xy=(x, FOMC_words[x] - 1200),
             xytext= (x, FOMC_words[x] - 1700),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 16)
plt.legend(fontsize = 15, loc = 'lower right') # Indicate the labed markers
'''
1. fontsize: Either an relative value of 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large' 
or an absolute font size, e.g., 12.
'''
plt.savefig('05_FOMC_Word_Counts_1.png', dpi= 1000, bbox_inches= 'tight') 
plt.close()






#### 8-2 Vocabulary counts line chart

plt.figure(figsize=(15, 5))
plt.grid()

plt.plot(up_index+1 ,  FOMC_vocabularies[up_index], 'o', color= 'blue', label="Up")  # up_index 
plt.plot(down_index+1,  FOMC_vocabularies[down_index], 's', color= 'crimson', label="Down")  # down_index 
plt.plot(unchanged_index+1,  FOMC_vocabularies[unchanged_index], '*', color= 'darkgreen', label="Unchanged")  # unchanged_index 
# Draw 3 kinds of markers and change their colors.
# 'o' is circle marker;'s' is square marker; '*' star marker. 
# Add labels and plt.legend() will catch them later
plt.plot(list(range(1,223)), FOMC_vocabularies, '--k')

# plt.xlabel('1993/01/01 ~ 2020/10/01', fontsize = 15)
plt.ylabel('Vocabulary Counts', fontsize = 'large')
plt.title('Vocabulary Counts of FOMC Minutes (1993/01/01 ~ 2020/10/01)', fontsize = 'large')
plt.axis([-1, 224, 0, 1020]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'])  
plt.yticks(range(0, 1020, 100))

plt.axvspan(66, 71, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(120, 132, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(217, 222, color='lightblue', alpha=0.5, lw=0)


crisis_data = [
    (66, '2001/03. Peak of Dot-Com Bubble.'),
    (120, '2007/12. Peak of Financial Crisis.'),
    (217, '2020/02. Peak of COVID-19.')]
x, label = crisis_data[0]
plt.annotate(label, xy=(x, FOMC_vocabularies[x] - 247),
             xytext= (x, FOMC_vocabularies[x] - 347),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16)
x, label = crisis_data[1]
plt.annotate(label, xy=(x, FOMC_vocabularies[x] - 240),
             xytext= (x, FOMC_vocabularies[x] - 330),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16)
x, label = crisis_data[2]
plt.annotate(label, xy=(x, FOMC_vocabularies[x] - 237),
             xytext= (x, FOMC_vocabularies[x] - 307),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 16)

plt.legend(fontsize = 14, loc = 'lower right') # Indicate the labed markers

plt.savefig('06_FOMC_Voc_Counts.png', dpi = 1000, bbox_inches= 'tight')
plt.close()


#### 8-3 TTR (Type/Token Ratio) 

# 8-3-1 TTR_1
FOMC_vocabularies_ratio = FOMC_vocabularies / FOMC_words

plt.figure(figsize=(15, 5))
plt.grid()

plt.plot(list(range(1,223)), FOMC_vocabularies_ratio, 'o', color= 'royalblue')
plt.plot(list(range(1,223)), FOMC_vocabularies_ratio, '--k') # label='Line'

plt.ylabel('TTR', fontsize = 'large')
plt.title('TTR of FOMC Minutes (1993/01/01 ~ 2020/10/01)', fontsize = 'large')
plt.axis([-2, 224, 0.18, 0.56]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'])  
# plt.yticks(range(0, 1020, 100))

plt.savefig('07_FOMC_TTR_1.png', dpi = 1000, bbox_inches= 'tight')
plt.close()


# 8-3-2 TTR_2
type(FOMC_vocabularies_ratio)
FOMC_vocabularies_ratio[6]
len(FOMC_vocabularies_ratio)

FOMC_vocabularies_ratio_2 = np.delete(FOMC_vocabularies_ratio, np.s_[6], 0)

FOMC_vocabularies_ratio_2[6]
len(FOMC_vocabularies_ratio_2)


plt.figure(figsize=(15, 5))
plt.grid()

plt.plot(list(range(1,222)), FOMC_vocabularies_ratio_2, 'o', color= 'royalblue')
plt.plot(list(range(1,222)), FOMC_vocabularies_ratio_2, '--k') # label='Line'

plt.ylabel('TTR', fontsize = 'large')
plt.title('TTR of FOMC Minutes (1993/01/01 ~ 2020/10/01)', fontsize = 'large')
plt.axis([-2, 224, 0.18, 0.40]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'])  
# plt.yticks(range(0, 1020, 100))

plt.savefig('07_FOMC_TTR_2.png', dpi = 1000, bbox_inches= 'tight')
plt.close()


# 8-3-3 TTR_3
import numpy as np

plt.figure(figsize=(15, 5)) # Change figure size
plt.grid() # Simply add grid by default

plt.plot(list(range(1,222)), FOMC_vocabularies_ratio_2, 'o', color= 'blue')
plt.plot(list(range(1,222)), FOMC_vocabularies_ratio_2, '--k') # label='Line'

plt.ylabel('TTR', fontsize = 13)
plt.title('TTR of FOMC Minutes  (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-2, 223, 0.18, 0.40]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 12.5)  
plt.yticks(np.arange(0.18, 0.40, 0.05), fontsize = 12.5)

# Shadow
plt.axvspan(65, 70, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(119, 131, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(216, 221, color='lightblue', alpha=0.5, lw=0)


# Annotation 
crisis_data = [
    (65, '2001/03. Peak of Dot-Com Bubble.'),
    (119, '2007/12. Peak of Financial Crisis.'),
    (216, '2020/02. Peak of COVID-19.')]

x, label = crisis_data[0]
plt.annotate(label, xy=(x, 0.58),
             xytext= (x, 0.57),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1
    
x, label = crisis_data[1]
plt.annotate(label, xy=(x, 0.567),
             xytext= (x, 0.5557),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1

x, label = crisis_data[2]
plt.annotate(label, xy=(x, 0.56),
             xytext= (x, 0.55),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 16) # rotation= 1

# plt.tight_layout() # tight layout for plt.show()
plt.savefig('07_FOMC_TTR_3.png', dpi= 1000, bbox_inches= 'tight') 
plt.close() # Close the current figure



## Chinese

import numpy as np

plt.figure(figsize=(11, 5)) # Change figure size
plt.grid() # Simply add grid by default

plt.plot(list(range(1,222)), FOMC_vocabularies_ratio_2, 'o', color= 'blue')
plt.plot(list(range(1,222)), FOMC_vocabularies_ratio_2, '--k') # label='Line'

plt.ylabel('TTR (type-token ratio)', fontsize = 15)
plt.title('FOMC Minutes 的 TTR (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-2, 223, 0.18, 0.40]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 12.5)  
plt.yticks(np.arange(0.18, 0.40, 0.05), fontsize = 13)

# Shadow
plt.axvspan(65, 70, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(119, 131, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(216, 221, color='lightblue', alpha=0.5, lw=0)


# Annotation 
crisis_data = [
    (65, '2001/03: Peak of Dot-Com Bubble'),
    (119, '2007/12: Peak of Financial Crisis'),
    (216, '2020/02: Peak of COVID-19')]

x, label = crisis_data[0]
plt.annotate(label, xy=(x, 0.23),
             xytext= (x, 0.21),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 15) # rotation= 1
    
x, label = crisis_data[1]
plt.annotate(label, xy=(x, 0.363),
             xytext= (x, 0.395),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 15) # rotation= 1

x, label = crisis_data[2]
plt.annotate(label, xy=(x, 0.28),
             xytext= (x, 0.312),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 15) # rotation= 1

# plt.tight_layout() # tight layout for plt.show()
plt.savefig('07_FOMC_TTR_4_Chinese.png', dpi= 1000, bbox_inches= 'tight') 
plt.close() # Close the current figure



#%% (9) Line Chart of STTR of FOMC Minutes Combining with Fed Funds Rate 


#### 9-1 Transforming FOMC Minute by dorpping the outlier

FOMC_pickle_2 = FOMC_pickle.copy()
len(FOMC_pickle_2[6])
del FOMC_pickle_2[6]
len(FOMC_pickle_2[6])



#### 9-2 STTR calculation

# It takes around 261 s 

import random
random.seed(20210202)

STTR_all = [0] * 3
STTR_sample_size = [100, 300, 500]

import time as t
t1 = t.perf_counter()

for k in range(3):
    
    STTR = []
    
    for i in range(0,221):
    
        STTR_j = []
        STTR_bootstrap_j = []
        for j in range(0,1000):
            STTR_bootstrap_j = random.sample(FOMC_pickle_2[i], STTR_sample_size[k])
            STTR_j.append(len(set(STTR_bootstrap_j)) / len(STTR_bootstrap_j))
        STTR.append((np.sum(STTR_j)/1000).tolist())
            
    STTR_all[k] = STTR

t2 = t.perf_counter()
print(t2-t1) # It takes around 260 s on my NB

print(len(STTR_all)); print(len(STTR_all[0]))
print(STTR_all[0][:10])



#### 9-3 Saving STTR calculation as pickle

# Serialization: Note that all data structures of Python need to be open in binary form
import os
os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()
import matplotlib.pyplot as plt

import pickle

# 9-3-1 Saving
with open('STTR_all', mode = 'wb') as file:
    pickle.dump(STTR_all, file, pickle.HIGHEST_PROTOCOL)

# 9-3-2 Loading
with open('STTR_all', 'rb') as file:
    STTR_all = pickle.load(file)
    


#### 9-4 Adjsuting indices to drop the outlier's index

up_index_2 = []
down_index_2 = []
unchanged_index_2 = []

for k in up_index:
    if k < 6:
        up_index_2.append(k)
    else:
        up_index_2.append(k-1)

for k in down_index:
    if k < 6:
        down_index_2.append(k)
    else:
        down_index_2.append(k-1)

for k in unchanged_index:
    if k == 6:
        continue
    if k < 6:
        unchanged_index_2.append(k)
    else:
        unchanged_index_2.append(k-1)

print(len(up_index)); print(len(up_index_2))
print(len(down_index)); print(len(down_index_2))
print(len(unchanged_index)); print(len(unchanged_index_2))
# print(unchanged_index_2)
FOMC_index_2 = sorted(up_index_2 + down_index_2 + unchanged_index_2)
print(FOMC_index_2[0:10])
print(len(FOMC_index_2))

print(up_index_2[0:10]); print(down_index_2[0:10]); print(unchanged_index_2[0:10])


up_index_3 = np.array(up_index_2)
down_index_3 = np.array(down_index_2)
unchanged_index_3 = np.array(unchanged_index_2)



#### 9-5 STTR (sampling 100 words) w/o the outlier
import numpy as np

plt.figure(figsize=(15, 5)) # Change figure size
plt.grid() # Simply add grid by default

STTR_100 = np.array(STTR_all[0])

# plt.plot(list(range(0,221)), STTR, 'bo')
plt.plot(up_index_3 + 1,  STTR_100[up_index_3], 'bo', label="Up")  # up_index 
plt.plot(down_index_3 + 1,  STTR_100[down_index_3], 'rs', label="Down")  # down_index 
plt.plot(unchanged_index_3 + 1,  STTR_100[unchanged_index_3], 'g*', label="Unchanged")  # unchanged_index 

plt.plot(list(range(1,222)), STTR_100, 'k', linewidth = 1.2) # label='Line'
# Draw a line. '--' is dashed line style. 'k' is black.

plt.ylabel('STTR', fontsize = 13)
plt.title('STTR (sampling 100 words) of FOMC Minutes  (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-2, 223, 0.820, 0.887]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 12.5)  
plt.yticks(np.arange(0.82, 0.888, 0.01), fontsize = 12.5)

# Shadow
plt.axvspan(65, 70, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(119, 131, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(216, 221, color='lightblue', alpha=0.5, lw=0)


# Annotation 
crisis_data = [
    (65, '2001/03. Peak of Dot-Com Bubble.'),
    (119, '2007/12. Peak of Financial Crisis.'),
    (216, '2020/02. Peak of COVID-19.')]

x, label = crisis_data[0]
plt.annotate(label, xy=(x, 0.851),
             xytext= (x, 0.841),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1
    
x, label = crisis_data[1]
plt.annotate(label, xy=(x, 0.835),
             xytext= (x, 0.825),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1

x, label = crisis_data[2]
plt.annotate(label, xy=(x, 0.841),
             xytext= (x, 0.831),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 16) # rotation= 1

# plt.tight_layout() # tight layout for plt.show()

plt.legend(fontsize = 12.5, loc = 'lower left') # Indicate the labed markers
plt.savefig('08_FOMC_STTR_Sampling_100.png', dpi= 1000, bbox_inches= 'tight') 
plt.close() # Close the current figure




#### 9-6 STTR (sampling 300 words) w/o the outlier
import numpy as np

plt.figure(figsize=(15, 5)) # Change figure size
plt.grid() # Simply add grid by default

STTR_300 = np.array(STTR_all[1])

# plt.plot(list(range(0,221)), STTR, 'bo')
plt.plot(up_index_3 + 1,  STTR_300[up_index_3], 'bo', label="Up")  # up_index 
plt.plot(down_index_3 + 1,  STTR_300[down_index_3], 'rs', label="Down")  # down_index 
plt.plot(unchanged_index_3 + 1,  STTR_300[unchanged_index_3], 'g*', label="Unchanged")  # unchanged_index 

plt.plot(list(range(1,222)), STTR_300, 'k', linewidth = 1.2) # label='Line'
# Draw a line. '--' is dashed line style. 'k' is black.

plt.ylabel('STTR', fontsize = 13)
plt.title('STTR (sampling 300 words) of FOMC Minutes  (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-2, 223, 0.639, 0.734]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 12.5)  
plt.yticks(np.arange(0.63, 0.75, 0.01), fontsize = 12.5)

# Shadow
plt.axvspan(65, 70, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(119, 131, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(216, 221, color='lightblue', alpha=0.5, lw=0)


# Annotation 
crisis_data = [
    (65, '2001/03. Peak of Dot-Com Bubble.'),
    (119, '2007/12. Peak of Financial Crisis.'),
    (216, '2020/02. Peak of COVID-19.')]

x, label = crisis_data[0]
plt.annotate(label, xy=(x, 0.682),
             xytext= (x, 0.672),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1
    
x, label = crisis_data[1]
plt.annotate(label, xy=(x, 0.668),
             xytext= (x, 0.658),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1

x, label = crisis_data[2]
plt.annotate(label, xy=(x, 0.662),
             xytext= (x, 0.652),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 16) # rotation= 1

# plt.tight_layout() # tight layout for plt.show()

plt.legend(fontsize = 12.5, loc = 'lower left') # Indicate the labed markers
plt.savefig('09_FOMC_STTR_Sampling_300.png', dpi= 1000, bbox_inches= 'tight') 
plt.close() # Close the current figure



#### 9-7 STTR (sampling 500 words) w/o the outlier
import numpy as np

plt.figure(figsize=(15, 5)) # Change figure size
plt.grid() # Simply add grid by default

STTR_500 = np.array(STTR_all[2])

# plt.plot(list(range(0,221)), STTR, 'bo')
plt.plot(up_index_3 + 1,  STTR_500[up_index_3], 'bo', label="Up")  # up_index 
plt.plot(down_index_3 + 1,  STTR_500[down_index_3], 'rs', label="Down")  # down_index 
plt.plot(unchanged_index_3 + 1,  STTR_500[unchanged_index_3], 'g*', label="Unchanged")  # unchanged_index 

plt.plot(list(range(1,222)), STTR_500, 'k', linewidth = 1.2) # label='Line'
# Draw a line. '--' is dashed line style. 'k' is black.

plt.ylabel('STTR', fontsize = 13)
plt.title('STTR (sampling 500 words) of FOMC Minutes  (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-2, 223, 0.545, 0.635]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 12.5)  
plt.yticks(np.arange(0.54, 0.64, 0.01), fontsize = 12.5)

# Shadow
plt.axvspan(65, 70, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(119, 131, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(216, 221, color='lightblue', alpha=0.5, lw=0)


# Annotation 
crisis_data = [
    (65, '2001/03. Peak of Dot-Com Bubble.'),
    (119, '2007/12. Peak of Financial Crisis.'),
    (216, '2020/02. Peak of COVID-19.')]

x, label = crisis_data[0]
plt.annotate(label, xy=(x, 0.58),
             xytext= (x, 0.57),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1
    
x, label = crisis_data[1]
plt.annotate(label, xy=(x, 0.567),
             xytext= (x, 0.5557),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1

x, label = crisis_data[2]
plt.annotate(label, xy=(x, 0.56),
             xytext= (x, 0.55),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 16) # rotation= 1

# plt.tight_layout() # tight layout for plt.show()
plt.legend(fontsize = 12.5, loc = 'lower left') # Indicate the labed markers
plt.savefig('10_FOMC_STTR_Sampling_500.png', dpi= 1000, bbox_inches= 'tight') 
plt.close() # Close the current figure



## Chinese version
import numpy as np

plt.figure(figsize=(11.6, 5)) # Change figure size
plt.grid() # Simply add grid by default

STTR_500 = np.array(STTR_all[2])

# plt.plot(list(range(0,221)), STTR, 'bo')
plt.plot(up_index_3 + 1,  STTR_500[up_index_3], 'bo', label="升息")  # up_index 
plt.plot(down_index_3 + 1,  STTR_500[down_index_3], 'rs', label="降息")  # down_index 
plt.plot(unchanged_index_3 + 1,  STTR_500[unchanged_index_3], 'g*', label="利率不變")  # unchanged_index 

plt.plot(list(range(1,222)), STTR_500, 'k', linewidth = 1.2) # label='Line'
# Draw a line. '--' is dashed line style. 'k' is black.

plt.ylabel('STTR', fontsize = 13)
plt.title('FOMC Minutes 的 STTR (拔靴法抽樣 1000 次，每次抽取 500 個單字) (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-2, 223, 0.545, 0.635]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 12.5)  
plt.yticks(np.arange(0.54, 0.64, 0.01), fontsize = 12.5)

# Shadow
plt.axvspan(65, 70, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(119, 131, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(216, 221, color='lightblue', alpha=0.5, lw=0)


# Annotation 
crisis_data = [
    (65, '2001/03: Peak of Dot-Com Bubble'),
    (119, '2007/12: Peak of Financial Crisis'),
    (216, '2020/02: Peak of COVID-19')]

x, label = crisis_data[0]
plt.annotate(label, xy=(x, 0.58),
             xytext= (x, 0.57),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 15) # rotation= 1
    
x, label = crisis_data[1]
plt.annotate(label, xy=(x, 0.567),
             xytext= (x, 0.5557),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 15) # rotation= 1

x, label = crisis_data[2]
plt.annotate(label, xy=(x, 0.56),
             xytext= (x, 0.55),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 15) # rotation= 1

# plt.tight_layout() # tight layout for plt.show()
plt.legend(fontsize = 12.5, loc = 'lower left') # Indicate the labed markers
plt.savefig('10_FOMC_STTR_Sampling_500_Chinese.png', dpi= 1000, bbox_inches= 'tight') 
plt.close() # Close the current figure




#%% (10) Probing into STTR: Time Series

#### 10-1 Max, min, (max-min)/2

(STTR_100.max() - STTR_100.min()) * 2 / (STTR_100.max() + STTR_100.min())
# 0.0592

(STTR_300.max() - STTR_300.min()) * 2 / (STTR_300.max() + STTR_300.min())
# 0.1063

(STTR_500.max() - STTR_500.min()) * 2 / (STTR_500.max() + STTR_500.min())
# 0.1217

# We reckon STTR_500 is the best in terms of discrimination.
# So we choose STTR_500 to do time series analysis


#### 10-2 Stationarity & Differencing
import numpy as np
STTR_500 = np.array(STTR_all[2])

from pmdarima.arima import ndiffs as ndiffs

# test =  (‘kpss’, ‘adf’, ‘pp’)

print('KPSS: d =', ndiffs(STTR_500, alpha=0.05, test='kpss', max_d=2)) # d = 1. Indicating non-stationary TS
print('ADF: d =', ndiffs(STTR_500, alpha=0.05, test='adf', max_d=2)) # d = 1. Indicating non-stationary TS
print('PP: d =', ndiffs(STTR_500, alpha=0.05, test='pp', max_d=2)) # d = 0. Indicating stationary TS


#### 10-3 ACF

# ACF of statsmodels
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib as mpl

mpl.rc("figure", figsize=(18,4))
plt.rc('font', size=15) 
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 

plot_acf(STTR_500, lags = 100)
plt.savefig('11_STTR_ACF.png', dpi= 1000, bbox_inches='tight')
plt.close()


#### 10-4 PACF

from statsmodels.graphics.tsaplots import plot_pacf

mpl.rc("figure", figsize=(18,4))
plt.rc('font', size=15) 
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 

plot_pacf(STTR_500, lags=100)
plt.savefig('12_STTR_PACF.png', dpi= 1000, bbox_inches='tight')
plt.close()


#### 10-5 Automatic ARIMA

import pmdarima as pm
model = pm.auto_arima(STTR_500, seasonal=True, m=8, suppress_warnings=True, trace=True, information_criterion='aic')
# Best ARIMA model: ARIMA(0,1,1)(0,0,0)[8] 
    

#### 10-6 Checking the Best ARIMA Model: ARIMA(0,1,1)(0,0,0)[8]

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(STTR_500, exog=None, order=(0,1,1), seansonal_order = (0,0,0,8),
              enforce_stationary=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())


#### 10-7 Ljung-Box Test (Lack of Fit Test)
import statsmodels.api as sm

type(model_fit)
type(model_fit.resid)
plt.plot(model_fit.resid)
plt.show()

sm.stats.acorr_ljungbox(model_fit.resid, lags=[8], return_df=True)
'''
    lb_stat  lb_pvalue
8  0.500105   0.999867
'''

sm.stats.acorr_ljungbox(model_fit.resid, return_df=True)

'''
     lb_stat  lb_pvalue
1   0.014491   0.904184
2   0.056663   0.972066
3   0.110132   0.990594
4   0.219719   0.994390
5   0.220002   0.998883
6   0.344616   0.999250
7   0.494846   0.999465
8   0.500105   0.999867
9   0.511359   0.999966
10  0.590017   0.999985
11  0.613912   0.999996
12  0.635377   0.999999
13  0.689286   1.000000
14  0.736851   1.000000
15  0.798144   1.000000
16  0.826578   1.000000
17  0.826956   1.000000
18  0.841224   1.000000
19  0.914473   1.000000
20  0.916941   1.000000
21  0.918962   1.000000
22  0.919229   1.000000
23  1.211235   1.000000
24  1.213826   1.000000
'''



#%% (11) Boxplot 

import numpy as np
len(FOMC_words)
type(FOMC_words)
len(FOMC_vocabularies)
type(FOMC_vocabularies)

print(FOMC_words[6])
FOMC_words_2 = np.delete(FOMC_words, np.s_[6])
print(FOMC_words_2[6])
len(FOMC_words_2)

print(FOMC_vocabularies[6])
FOMC_vocabularies_2 = np.delete(FOMC_vocabularies, np.s_[6])
print(FOMC_vocabularies_2[6])


#### 11-1 Word counts
import os
os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()

# [1] Boxplot_01_words
# datafile = u'D:\\pythondata\\learn\\matplotlib.xlsx'
# data = pd.read_excel(datafile)
box_1, box_2, box_3 = FOMC_words_2[up_index_2], FOMC_words_2[down_index_2], FOMC_words_2[unchanged_index_2]
 
fig = plt.figure(figsize=(7,2.2)) # figsize=(9,3)
plt.title('Word Counts of 3 Categories (w/o the outlier)', fontsize = 15)
labels = 'Up(17.6%)','Down(13.5%)','Unchanged(68.9%)'

# vert=False; showmeans=True:
plt.boxplot( [box_1, box_2, box_3], labels = labels, widths= 0.7,
            vert= False, showmeans= True)
# plt.show()
# plt.axis([1500,5000, 0.5, 3.5])
# plt.xticks(range(0, 223, 25))  
# plt.yticks(list(range(0, 5001, 500)))  
plt.grid(color='silver', linestyle='-', linewidth= 1, b=None, which='major', axis='both')

plt.tight_layout()
fig.savefig('13_Boxplot_Words.png', dpi = 900, bbox_inches='tight')
plt.close()


## Chinese
import os
os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()

# [1] Boxplot_01_words
# datafile = u'D:\\pythondata\\learn\\matplotlib.xlsx'
# data = pd.read_excel(datafile)
box_1, box_2, box_3 = FOMC_words_2[up_index_2], FOMC_words_2[down_index_2], FOMC_words_2[unchanged_index_2]
 
fig = plt.figure(figsize=(7,2.2)) # figsize=(9,3)
plt.title('升息、降息、利率不變三類樣本的字詞數統計 (扣除了一個離群值)', fontsize = 15)
labels = '升息 (17.6%)','降息 (13.5%)','利率不變 (68.9%)'

# vert=False; showmeans=True:
plt.boxplot( [box_1, box_2, box_3], labels = labels, widths= 0.7,
            vert= False, showmeans= True)
# plt.show()
# plt.axis([1500,5000, 0.5, 3.5])
# plt.xticks(range(0, 223, 25))  
# plt.yticks(list(range(0, 5001, 500)))  
plt.grid(color='silver', linestyle='-', linewidth= 1, b=None, which='major', axis='both')

plt.tight_layout()
fig.savefig('13_Boxplot_Words_Chinese.png', dpi = 900, bbox_inches='tight')
plt.close()



#### 11-2 Vocabulary counts

box_1, box_2, box_3 = FOMC_vocabularies_2[up_index_2], FOMC_vocabularies_2[down_index_2], FOMC_vocabularies_2[unchanged_index_2]
 
 
fig = plt.figure(figsize=(7,1.7))
plt.title('Voacblary Counts of 3 Categories (w/o the outlier)', fontsize = 15)
labels = 'Up(17.6%)','Down(13.5%)','Unchanged(68.9%)'
 
# vert=False；showmeans=True：
plt.boxplot( [box_1, box_2, box_3], labels = labels, widths= 0.7,
            vert= False, showmeans= True)
# plt.axis([540,1000, 0.5, 3.5])

plt.grid(color='silver', linestyle='-', linewidth= 1, b=None, which='major', axis='both') # linewidth= 1.2

fig.savefig('14_Boxplot_voacblaries.png', dpi= 900, bbox_inches='tight')
plt.close()



### 11-3 STTR_100

box_1, box_2, box_3 = STTR_100[up_index_2], STTR_100[down_index_2], STTR_100[unchanged_index_2]
 
fig =  plt.figure(figsize=(7,1.8)) # figsize=(9,3)
plt.title('STTR of 3 Categories (sampling 100 words)(w/o the outlier)',fontsize = 15)
labels = 'Up(17.6%)','Down(13.5%)','Unchanged(68.9%)'
 
# vert=False；showmeans=True：
plt.boxplot( [box_1, box_2, box_3], labels = labels, widths= 0.7,
            vert= False, showmeans= True)
plt.grid(color='silver', linestyle='-', linewidth= 1, b=None, which='major', axis='both') # linewidth= 1.2
# plt.tight_layout()

# plt.show()
fig.savefig('15_Boxplot_STTR_100.png', dpi= 900, bbox_inches='tight')
plt.close()



#### 11-4 STTR_300

box_1, box_2, box_3 = STTR_300[up_index_2], STTR_300[down_index_2], STTR_300[unchanged_index_2]
 
fig =  plt.figure(figsize=(7,1.8)) # figsize=(9,3)
plt.title('STTR of 3 Categories (sampling 300 words)(w/o the outlier)',fontsize = 15)
labels = 'Up(17.6%)','Down(13.5%)','Unchanged(68.9%)'
 
# vert=False；showmeans=True：
plt.boxplot( [box_1, box_2, box_3], labels = labels, widths= 0.7,
            vert= False, showmeans= True)
plt.grid(color='silver', linestyle='-', linewidth= 1, b=None, which='major', axis='both') # linewidth= 1.2
# plt.tight_layout()

# plt.show()
fig.savefig('16_Boxplot_STTR_300.png', dpi= 900, bbox_inches='tight')
plt.close()



#### 11-5 STTR_500

box_1, box_2, box_3 = STTR_500[up_index_2], STTR_500[down_index_2], STTR_500[unchanged_index_2]
 
fig =  plt.figure(figsize=(7,1.8)) # figsize=(9,3)
plt.title('STTR of 3 Categories (sampling 500 words)(w/o the outlier)',fontsize = 15)
labels = 'Up(17.6%)','Down(13.5%)','Unchanged(68.9%)'
 
# vert=False；showmeans=True：
plt.boxplot( [box_1, box_2, box_3], labels = labels, widths= 0.7,
            vert= False, showmeans= True)
plt.grid(color='silver', linestyle='-', linewidth= 1, b=None, which='major', axis='both') # linewidth= 1.2
# plt.tight_layout()

# plt.show()
fig.savefig('17_Boxplot_STTR_500.png', dpi= 900, bbox_inches='tight')
plt.close()


## Chinese

box_1, box_2, box_3 = STTR_500[up_index_2], STTR_500[down_index_2], STTR_500[unchanged_index_2]
 
fig =  plt.figure(figsize=(7,1.8)) # figsize=(9,3)
plt.title('升息、降息、利率不變樣本的 STTR 箱型圖',fontsize = 15)
labels = '升息 (17.6%)','降息 (13.5%)','利率不變 (68.9%)'
 
# vert=False；showmeans=True：
plt.boxplot( [box_1, box_2, box_3], labels = labels, widths= 0.7,
            vert= False, showmeans= True)
plt.grid(color='silver', linestyle='-', linewidth= 1, b=None, which='major', axis='both') # linewidth= 1.2
# plt.tight_layout()

# plt.show()
fig.savefig('17_Boxplot_STTR_500_Chinese.png', dpi= 900, bbox_inches='tight')
plt.close()






#%% (12) Adding Fed Chairman Terms to EDA

os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.subplots(figsize=(18,10))


plt.subplot(2,1,1)

# plt.figure(figsize=(18,5))
plt.grid()

plt.axvspan(98, 107, color='thistle', alpha=0.5, lw=0)
plt.axvspan(179, 198, color='thistle', alpha=0.5, lw=0)
plt.axvspan(325, 334, color='thistle', alpha=0.5, lw=0)

mpl.rc('xtick', labelsize=13.5) 
mpl.rc('ytick', labelsize=15) 
# plt.xticks(range(0, 333, 29), fontsize = 13) 
plt.xticks(range(0, 333, 29), 
           labels = ['1993-01-01', '1995-06-01', '1997-11-01', '2000-04-01', '2002-09-01', '2005-02-01', 
                     '2007-07-01', '2009-12-01', '2012-05-01', '2014-10-01', '2017-03-01', '2019-08-01'],
           fontsize = 13,  rotation= 3.5)
plt.plot(real_rate.FEDFUNDS, c='blue')

plt.axis([-2, 336, -0.2, 7.0]) 
plt.ylabel('Percent', fontsize = 18)
plt.title('Effective Federal Funds Rate & Business cycle (1993/01/01 ~ 2020/10/01)', fontsize = 20)

# plt.savefig('18_Fed_Chairman_Rate.png', dpi= 1000, bbox_inches='tight')
# plt.close()



plt.subplot(2,1,2)

plt.grid()

plt.axvspan(157, 252, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(252, 300, color='lightgreen', alpha=0.5, lw=0)
plt.axvspan(300, 334, color='pink', alpha=0.5, lw=0)

plt.annotate('Alan Greenspan', xy=(86, 7.7),
             xytext= (86, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Ben Bernanke', xy=(208, 7.7),
             xytext= (208, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Janet Yellen', xy=(275, 7.7),
             xytext= (275, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Jerome', xy=(318, 7.7),
             xytext= (318, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Powell', xy=(318, 6.6),
             xytext= (318, 6.6),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

mpl.rc('xtick', labelsize=13.5) 
mpl.rc('ytick', labelsize=15) 
# plt.xticks(range(0, 333, 29), fontsize = 13) 
plt.xticks(range(0, 333, 29), 
           labels = ['1993-01-01', '1995-06-01', '1997-11-01', '2000-04-01', '2002-09-01', '2005-02-01', 
                     '2007-07-01', '2009-12-01', '2012-05-01', '2014-10-01', '2017-03-01', '2019-08-01'],
           fontsize = 13, rotation= 3.5)

plt.yticks(range(0, 8, 1), fontsize = 13) 
plt.plot(real_rate.FEDFUNDS, c='blue')

plt.axis([-2, 336, -0.2, 8.0]) 


plt.ylabel('Percent', fontsize = 18)
plt.title('Effective Federal Funds Rate & Fed Chairman Terms (1993/01/01 ~ 2020/10/01)', fontsize = 20)


plt.savefig('20_Fed_Chairman_Funds_Rate_1_Chinese.png', dpi= 1000, bbox_inches='tight')
plt.close()





# ==== Chinese version ====

os.chdir(r'D:\G03_1\FOMC\FOMC_05_output_figures')
os.getcwd()
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.subplots(figsize=(15,9))

plt.subplot(2,1,1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

# plt.figure(figsize=(18,5))
plt.grid()

plt.axvspan(98, 107, color='thistle', alpha=0.5, lw=0)
plt.axvspan(179, 198, color='thistle', alpha=0.5, lw=0)
plt.axvspan(325, 334, color='thistle', alpha=0.5, lw=0)

mpl.rc('xtick', labelsize=13.5) 
mpl.rc('ytick', labelsize=15) 
# plt.xticks(range(0, 333, 29), fontsize = 13) 
plt.xticks(range(0, 333, 29), 
           labels = ['1993-01-01', '1995-06-01', '1997-11-01', '2000-04-01', '2002-09-01', '2005-02-01', 
                     '2007-07-01', '2009-12-01', '2012-05-01', '2014-10-01', '2017-03-01', '2019-08-01'],
           fontsize = 13,  rotation= 3.5)
plt.plot(real_rate.FEDFUNDS, c='blue')

plt.axis([-2, 336, -0.2, 7.0]) 
plt.ylabel('百分比', fontsize = 18)
plt.title('聯邦基金利率變動與景氣循環 (1993/01/01 ~ 2020/10/01) (陰影區為衰退期)', fontsize = 20)

# plt.savefig('18_Fed_Chairman_Rate.png', dpi= 1000, bbox_inches='tight')
# plt.close()



plt.subplot(2,1,2)

plt.grid()

plt.axvspan(157, 252, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(252, 300, color='lightgreen', alpha=0.5, lw=0)
plt.axvspan(300, 334, color='pink', alpha=0.5, lw=0)

plt.annotate('Alan Greenspan', xy=(86, 7.7),
             xytext= (86, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Ben Bernanke', xy=(208, 7.7),
             xytext= (208, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Janet Yellen', xy=(275, 7.7),
             xytext= (275, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Jerome', xy=(318, 7.7),
             xytext= (318, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Powell', xy=(318, 6.6),
             xytext= (318, 6.6),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

mpl.rc('xtick', labelsize=13.5) 
mpl.rc('ytick', labelsize=15) 
# plt.xticks(range(0, 333, 29), fontsize = 13) 
plt.xticks(range(0, 333, 29), 
           labels = ['1993-01-01', '1995-06-01', '1997-11-01', '2000-04-01', '2002-09-01', '2005-02-01', 
                     '2007-07-01', '2009-12-01', '2012-05-01', '2014-10-01', '2017-03-01', '2019-08-01'],
           fontsize = 13, rotation= 3.5)

plt.yticks(range(0, 8, 1), fontsize = 13) 
plt.plot(real_rate.FEDFUNDS, c='blue')

plt.axis([-2, 336, -0.2, 8.0]) 


plt.ylabel('百分比', fontsize = 18)
plt.title('聯邦基金利率變動與聯準會主席任期 (1993/01/01 ~ 2020/10/01)', fontsize = 20)


plt.savefig('20_Fed_Chairman_Funds_Rate_1_Chinese.png', dpi= 1000, bbox_inches='tight')
plt.close()




#### 12-2 Fed Funds Rate & Business Cycle & Fed Chairman Terms & STTR_500

plt.subplots(figsize=(18,10))


plt.subplot(2,1,1)

# plt.figure(figsize=(18,5))
plt.grid()

plt.axvspan(157, 252, color='lightblue', alpha=0.5, lw=0)
plt.axvspan(252, 300, color='lightgreen', alpha=0.5, lw=0)
plt.axvspan(300, 334, color='pink', alpha=0.5, lw=0)

plt.annotate('Alan Greenspan', xy=(86, 7.7),
             xytext= (86, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Ben Bernanke', xy=(208, 7.7),
             xytext= (208, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Janet Yellen', xy=(275, 7.7),
             xytext= (275, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Jerome', xy=(318, 7.7),
             xytext= (318, 7.7),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

plt.annotate('Powell', xy=(318, 6.6),
             xytext= (318, 6.6),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 22)

mpl.rc('xtick', labelsize=13.5) 
mpl.rc('ytick', labelsize=15) 
# plt.xticks(range(0, 333, 29), fontsize = 13) 
plt.xticks(range(0, 333, 29), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 13)
plt.yticks(range(0, 8, 1), fontsize = 13) 
plt.plot(real_rate.FEDFUNDS, c='blue')

plt.axis([-2, 336, -0.2, 8.0]) 
plt.ylabel('Percent', fontsize = 18)
plt.title('Effective Federal Funds Rate & Fed Chairman Terms (1993/01/01 ~ 2020/10/01)', fontsize = 20)



plt.subplot(2,1,2)

import numpy as np

# plt.figure(figsize=(15, 5)) # Change figure size
plt.grid() # Simply add grid by default

STTR_500 = np.array(STTR_all[2])

# plt.plot(list(range(0,221)), STTR, 'bo')
plt.plot(up_index_3 + 1,  STTR_500[up_index_3], 'bo', label="Up")  # up_index 
plt.plot(down_index_3 + 1,  STTR_500[down_index_3], 'rs', label="Down")  # down_index 
plt.plot(unchanged_index_3 + 1,  STTR_500[unchanged_index_3], 'g*', label="Unchanged")  # unchanged_index 

plt.plot(list(range(1,222)), STTR_500, 'k', linewidth = 1.2) # label='Line'
# Draw a line. '--' is dashed line style. 'k' is black.

plt.ylabel('STTR', fontsize = 13)
plt.title('STTR (sampling 500 words) of FOMC Minutes  (1993/01/01 ~ 2020/10/01)', fontsize = 16)
plt.axis([-2, 223, 0.545, 0.635]) 
plt.xticks(range(1, 224, 20), 
           labels = ['1993/02','1995/08','1998/02','2000/08','2003/01','2005/08',
                     '2008/01','2010/08','2013/01','2015/07','2018/01','2020/07'],
           fontsize = 12.5)  
plt.yticks(np.arange(0.54, 0.64, 0.01), fontsize = 12.5)

# Shadow
plt.axvspan(65, 70, color='thistle', alpha=0.5, lw=0)
plt.axvspan(119, 131, color='thistle', alpha=0.5, lw=0)
plt.axvspan(216, 221, color='thistle', alpha=0.5, lw=0)


# Annotation 
crisis_data = [
    (65, '2001/03. Peak of Dot-Com Bubble.'),
    (119, '2007/12. Peak of Financial Crisis.'),
    (216, '2020/02. Peak of COVID-19.')]

x, label = crisis_data[0]
plt.annotate(label, xy=(x, 0.58),
             xytext= (x, 0.57),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1
    
x, label = crisis_data[1]
plt.annotate(label, xy=(x, 0.567),
             xytext= (x, 0.5557),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='center', verticalalignment= 'top', fontsize = 16) # rotation= 1

x, label = crisis_data[2]
plt.annotate(label, xy=(x, 0.56),
             xytext= (x, 0.55),
             arrowprops= dict(facecolor='black', headwidth= 4, width=2, headlength= 4),
             horizontalalignment='right', verticalalignment= 'top', fontsize = 16) # rotation= 1

# plt.tight_layout() # tight layout for plt.show()
plt.legend(fontsize = 12.5, loc = 'lower left') # Indicate the labed markers


plt.savefig('21_Fed_Chairman_Funds_Rate_STTR_500_02.png', dpi= 1000, bbox_inches='tight')
plt.close()


