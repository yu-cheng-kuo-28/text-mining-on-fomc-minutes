# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:48:18 2021

@author: Morton
"""

#%% (1) Set the environment
import os
os.chdir(r'D:\G03_1\FOMC\FOMC_03_all_text_data')
os.getcwd()


#%% (2) Input

# 1990
f_19900207 = open('FOMC_19900207.txt','r',encoding='utf8').read()
f_19900327 = open('FOMC_19900327.txt','r',encoding='utf8').read()
f_19900515 = open('FOMC_19900515.txt','r',encoding='utf8').read()
f_19900703 = open('FOMC_19900703.txt','r',encoding='utf8').read()
f_19900821 = open('FOMC_19900821.txt','r',encoding='utf8').read()
f_19901002 = open('FOMC_19901002.txt','r',encoding='utf8').read()
f_19901113 = open('FOMC_19901113.txt','r',encoding='utf8').read()
f_19901218 = open('FOMC_19901218.txt','r',encoding='utf8').read()


# 1991~1995
f_19910206 = open('FOMC_19910206.txt','r',encoding='utf8').read()
f_19910326 = open('FOMC_19910326.txt','r',encoding='utf8').read()
f_19910514 = open('FOMC_19910514.txt','r',encoding='utf8').read()
f_19910703 = open('FOMC_19910703.txt','r',encoding='utf8').read()
f_19910820 = open('FOMC_19910820.txt','r',encoding='utf8').read()
f_19911001 = open('FOMC_19911001.txt','r',encoding='utf8').read()
f_19911105 = open('FOMC_19911105.txt','r',encoding='utf8').read()
f_19911217 = open('FOMC_19911217.txt','r',encoding='utf8').read()

f_19920205 = open('FOMC_19920205.txt','r',encoding='utf8').read()
f_19920331 = open('FOMC_19920331.txt','r',encoding='utf8').read()
f_19920519 = open('FOMC_19920519.txt','r',encoding='utf8').read()
f_19920701 = open('FOMC_19920701.txt','r',encoding='utf8').read()
f_19920818 = open('FOMC_19920818.txt','r',encoding='utf8').read()
f_19921006 = open('FOMC_19921006.txt','r',encoding='utf8').read()
f_19921117 = open('FOMC_19921117.txt','r',encoding='utf8').read()
f_19921222 = open('FOMC_19921222.txt','r',encoding='utf8').read()

f_19930203 = open('FOMC_19930203.txt','r',encoding='utf8').read()
f_19930323 = open('FOMC_19930323.txt','r',encoding='utf8').read()
f_19930518 = open('FOMC_19930518.txt','r',encoding='utf8').read()
f_19930707 = open('FOMC_19930707.txt','r',encoding='utf8').read()
f_19930817 = open('FOMC_19930817.txt','r',encoding='utf8').read()
f_19930921 = open('FOMC_19930921.txt','r',encoding='utf8').read()
f_19931116 = open('FOMC_19931116.txt','r',encoding='utf8').read()
f_19931221 = open('FOMC_19931221.txt','r',encoding='utf8').read()

f_19940204 = open('FOMC_19940204.txt','r',encoding='utf8').read()
f_19940322 = open('FOMC_19940322.txt','r',encoding='utf8').read()
f_19940517 = open('FOMC_19940517.txt','r',encoding='utf8').read()
f_19940706 = open('FOMC_19940706.txt','r',encoding='utf8').read()
f_19940816 = open('FOMC_19940816.txt','r',encoding='utf8').read()
f_19940927 = open('FOMC_19940927.txt','r',encoding='utf8').read()
f_19941115 = open('FOMC_19941115.txt','r',encoding='utf8').read()
f_19941220 = open('FOMC_19941220.txt','r',encoding='utf8').read()

f_19950201 = open('FOMC_19950201.txt','r',encoding='utf8').read()
f_19950328 = open('FOMC_19950328.txt','r',encoding='utf8').read()
f_19950523 = open('FOMC_19950523.txt','r',encoding='utf8').read()
f_19950706 = open('FOMC_19950706.txt','r',encoding='utf8').read()
f_19950822 = open('FOMC_19950822.txt','r',encoding='utf8').read()
f_19950926 = open('FOMC_19950926.txt','r',encoding='utf8').read()
f_19951115 = open('FOMC_19951115.txt','r',encoding='utf8').read()
f_19951219 = open('FOMC_19951219.txt','r',encoding='utf8').read()


# 1996~2000
f_19960130 = open('FOMC_19960130.txt','r',encoding='utf8').read()
f_19960326 = open('FOMC_19960326.txt','r',encoding='utf8').read()
f_19960521 = open('FOMC_19960521.txt','r',encoding='utf8').read()
f_19960702 = open('FOMC_19960702.txt','r',encoding='utf8').read()
f_19960820 = open('FOMC_19960820.txt','r',encoding='utf8').read()
f_19960924 = open('FOMC_19960924.txt','r',encoding='utf8').read()
f_19961113 = open('FOMC_19961113.txt','r',encoding='utf8').read()
f_19961217 = open('FOMC_19961217.txt','r',encoding='utf8').read()

f_19970204 = open('FOMC_19970204.txt','r',encoding='utf8').read()
f_19970325 = open('FOMC_19970325.txt','r',encoding='utf8').read()
f_19970520 = open('FOMC_19970520.txt','r',encoding='utf8').read()
f_19970701 = open('FOMC_19970701.txt','r',encoding='utf8').read()
f_19970819 = open('FOMC_19970819.txt','r',encoding='utf8').read()
f_19970930 = open('FOMC_19970930.txt','r',encoding='utf8').read()
f_19971112 = open('FOMC_19971112.txt','r',encoding='utf8').read()
f_19971216 = open('FOMC_19971216.txt','r',encoding='utf8').read()

f_19980203 = open('FOMC_19980203.txt','r',encoding='utf8').read()
f_19980331 = open('FOMC_19980331.txt','r',encoding='utf8').read()
f_19980519 = open('FOMC_19980519.txt','r',encoding='utf8').read()
f_19980630 = open('FOMC_19980630.txt','r',encoding='utf8').read()
f_19980818 = open('FOMC_19980818.txt','r',encoding='utf8').read()
f_19980929 = open('FOMC_19980929.txt','r',encoding='utf8').read()
f_19981117 = open('FOMC_19981117.txt','r',encoding='utf8').read()
f_19981222 = open('FOMC_19981222.txt','r',encoding='utf8').read()

f_19990202 = open('FOMC_19990202.txt','r',encoding='utf8').read()
f_19990330 = open('FOMC_19990330.txt','r',encoding='utf8').read()
f_19990518 = open('FOMC_19990518.txt','r',encoding='utf8').read()
f_19990629 = open('FOMC_19990629.txt','r',encoding='utf8').read()
f_19990824 = open('FOMC_19990824.txt','r',encoding='utf8').read()
f_19991005 = open('FOMC_19991005.txt','r',encoding='utf8').read()
f_19991116 = open('FOMC_19991116.txt','r',encoding='utf8').read()
f_19991221 = open('FOMC_19991221.txt','r',encoding='utf8').read()

f_20000202 = open('FOMC_20000202.txt','r',encoding='utf8').read()
f_20000321 = open('FOMC_20000321.txt','r',encoding='utf8').read()
f_20000516 = open('FOMC_20000516.txt','r',encoding='utf8').read()
f_20000628 = open('FOMC_20000628.txt','r',encoding='utf8').read()
f_20000822 = open('FOMC_20000822.txt','r',encoding='utf8').read()
f_20001003 = open('FOMC_20001003.txt','r',encoding='utf8').read()
f_20001115 = open('FOMC_20001115.txt','r',encoding='utf8').read()
f_20001219 = open('FOMC_20001219.txt','r',encoding='utf8').read()


# 2001~2005
f_20010131 = open('FOMC_20010131.txt','r',encoding='utf8').read()
f_20010320 = open('FOMC_20010320.txt','r',encoding='utf8').read()
f_20010515 = open('FOMC_20010515.txt','r',encoding='utf8').read()
f_20010627 = open('FOMC_20010627.txt','r',encoding='utf8').read()
f_20010821 = open('FOMC_20010821.txt','r',encoding='utf8').read()
f_20011002 = open('FOMC_20011002.txt','r',encoding='utf8').read()
f_20011106 = open('FOMC_20011106.txt','r',encoding='utf8').read()
f_20011211 = open('FOMC_20011211.txt','r',encoding='utf8').read()

f_20020130 = open('FOMC_20020130.txt','r',encoding='utf8').read()
f_20020319 = open('FOMC_20020319.txt','r',encoding='utf8').read()
f_20020507 = open('FOMC_20020507.txt','r',encoding='utf8').read()
f_20020626 = open('FOMC_20020626.txt','r',encoding='utf8').read()
f_20020813 = open('FOMC_20020813.txt','r',encoding='utf8').read()
f_20020924 = open('FOMC_20020924.txt','r',encoding='utf8').read()
f_20021106 = open('FOMC_20021106.txt','r',encoding='utf8').read()
f_20021210 = open('FOMC_20021210.txt','r',encoding='utf8').read()

f_20030129 = open('FOMC_20030129.txt','r',encoding='utf8').read()
f_20030318 = open('FOMC_20030318.txt','r',encoding='utf8').read()
f_20030506 = open('FOMC_20030506.txt','r',encoding='utf8').read()
f_20030625 = open('FOMC_20030625.txt','r',encoding='utf8').read()
f_20030812 = open('FOMC_20030812.txt','r',encoding='utf8').read()
f_20030916 = open('FOMC_20030916.txt','r',encoding='utf8').read()
f_20031028 = open('FOMC_20031028.txt','r',encoding='utf8').read()
f_20031209 = open('FOMC_20031209.txt','r',encoding='utf8').read()

f_20040128 = open('FOMC_20040128.txt','r',encoding='utf8').read()
f_20040316 = open('FOMC_20040316.txt','r',encoding='utf8').read()
f_20040504 = open('FOMC_20040504.txt','r',encoding='utf8').read()
f_20040630 = open('FOMC_20040630.txt','r',encoding='utf8').read()
f_20040810 = open('FOMC_20040810.txt','r',encoding='utf8').read()
f_20040921 = open('FOMC_20040921.txt','r',encoding='utf8').read()
f_20041110 = open('FOMC_20041110.txt','r',encoding='utf8').read()
f_20041214 = open('FOMC_20041214.txt','r',encoding='utf8').read()

f_20050202 = open('FOMC_20050202.txt','r',encoding='utf8').read()
f_20050322 = open('FOMC_20050322.txt','r',encoding='utf8').read()
f_20050503 = open('FOMC_20050503.txt','r',encoding='utf8').read()
f_20050630 = open('FOMC_20050630.txt','r',encoding='utf8').read()
f_20050809 = open('FOMC_20050809.txt','r',encoding='utf8').read()
f_20050920 = open('FOMC_20050920.txt','r',encoding='utf8').read()
f_20051101 = open('FOMC_20051101.txt','r',encoding='utf8').read()
f_20051213 = open('FOMC_20051213.txt','r',encoding='utf8').read()


# 2006~2010
f_20060131 = open('FOMC_20060131.txt','r',encoding='utf8').read()
f_20060328 = open('FOMC_20060328.txt','r',encoding='utf8').read()
f_20060510 = open('FOMC_20060510.txt','r',encoding='utf8').read()
f_20060629 = open('FOMC_20060629.txt','r',encoding='utf8').read()
f_20060808 = open('FOMC_20060808.txt','r',encoding='utf8').read()
f_20060920 = open('FOMC_20060920.txt','r',encoding='utf8').read()
f_20061025 = open('FOMC_20061025.txt','r',encoding='utf8').read()
f_20061212 = open('FOMC_20061212.txt','r',encoding='utf8').read()

f_20070131 = open('FOMC_20070131.txt','r',encoding='utf8').read()
f_20070321 = open('FOMC_20070321.txt','r',encoding='utf8').read()
f_20070509 = open('FOMC_20070509.txt','r',encoding='utf8').read()
f_20070628 = open('FOMC_20070628.txt','r',encoding='utf8').read()
f_20070807 = open('FOMC_20070807.txt','r',encoding='utf8').read()
f_20070918 = open('FOMC_20070918.txt','r',encoding='utf8').read()
f_20071031 = open('FOMC_20071031.txt','r',encoding='utf8').read()
f_20071211 = open('FOMC_20071211.txt','r',encoding='utf8').read()

f_20080130 = open('FOMC_20080130.txt','r',encoding='utf8').read()
f_20080318 = open('FOMC_20080318.txt','r',encoding='utf8').read()
f_20080430 = open('FOMC_20080430.txt','r',encoding='utf8').read()
f_20080625 = open('FOMC_20080625.txt','r',encoding='utf8').read()
f_20080805 = open('FOMC_20080805.txt','r',encoding='utf8').read()
f_20080916 = open('FOMC_20080916.txt','r',encoding='utf8').read()
f_20081029 = open('FOMC_20081029.txt','r',encoding='utf8').read()
f_20081216 = open('FOMC_20081216.txt','r',encoding='utf8').read()

f_20090128 = open('FOMC_20090128.txt','r',encoding='utf8').read()
f_20090318 = open('FOMC_20090318.txt','r',encoding='utf8').read()
f_20090429 = open('FOMC_20090429.txt','r',encoding='utf8').read()
f_20090624 = open('FOMC_20090624.txt','r',encoding='utf8').read()
f_20090812 = open('FOMC_20090812.txt','r',encoding='utf8').read()
f_20090923 = open('FOMC_20090923.txt','r',encoding='utf8').read()
f_20091104 = open('FOMC_20091104.txt','r',encoding='utf8').read()
f_20091216 = open('FOMC_20091216.txt','r',encoding='utf8').read()

f_20100127 = open('FOMC_20100127.txt','r',encoding='utf8').read()
f_20100316 = open('FOMC_20100316.txt','r',encoding='utf8').read()
f_20100428 = open('FOMC_20100428.txt','r',encoding='utf8').read()
f_20100623 = open('FOMC_20100623.txt','r',encoding='utf8').read()
f_20100810 = open('FOMC_20100810.txt','r',encoding='utf8').read()
f_20100921 = open('FOMC_20100921.txt','r',encoding='utf8').read()
f_20101103 = open('FOMC_20101103.txt','r',encoding='utf8').read()
f_20101214 = open('FOMC_20101214.txt','r',encoding='utf8').read()


# 2011~2015
f_20110126 = open('FOMC_20110126.txt','r',encoding='utf8').read()
f_20110315 = open('FOMC_20110315.txt','r',encoding='utf8').read()
f_20110427 = open('FOMC_20110427.txt','r',encoding='utf8').read()
f_20110622 = open('FOMC_20110622.txt','r',encoding='utf8').read()
f_20110809 = open('FOMC_20110809.txt','r',encoding='utf8').read()
f_20110921 = open('FOMC_20110921.txt','r',encoding='utf8').read()
f_20111102 = open('FOMC_20111102.txt','r',encoding='utf8').read()
f_20111213 = open('FOMC_20111213.txt','r',encoding='utf8').read()

f_20120125 = open('FOMC_20120125.txt','r',encoding='utf8').read()
f_20120313 = open('FOMC_20120313.txt','r',encoding='utf8').read()
f_20120425 = open('FOMC_20120425.txt','r',encoding='utf8').read()
f_20120620 = open('FOMC_20120620.txt','r',encoding='utf8').read()
f_20120801 = open('FOMC_20120801.txt','r',encoding='utf8').read()
f_20120913 = open('FOMC_20120913.txt','r',encoding='utf8').read()
f_20121024 = open('FOMC_20121024.txt','r',encoding='utf8').read()
f_20121212 = open('FOMC_20121212.txt','r',encoding='utf8').read()

f_20130130 = open('FOMC_20130130.txt','r',encoding='utf8').read()
f_20130320 = open('FOMC_20130320.txt','r',encoding='utf8').read()
f_20130501 = open('FOMC_20130501.txt','r',encoding='utf8').read()
f_20130619 = open('FOMC_20130619.txt','r',encoding='utf8').read()
f_20130731 = open('FOMC_20130731.txt','r',encoding='utf8').read()
f_20130918 = open('FOMC_20130918.txt','r',encoding='utf8').read()
f_20131030 = open('FOMC_20131030.txt','r',encoding='utf8').read()
f_20131218 = open('FOMC_20131218.txt','r',encoding='utf8').read()

f_20140129 = open('FOMC_20140129.txt','r',encoding='utf8').read()
f_20140319 = open('FOMC_20140319.txt','r',encoding='utf8').read()
f_20140430 = open('FOMC_20140430.txt','r',encoding='utf8').read()
f_20140618 = open('FOMC_20140618.txt','r',encoding='utf8').read()
f_20140730 = open('FOMC_20140730.txt','r',encoding='utf8').read()
f_20140917 = open('FOMC_20140917.txt','r',encoding='utf8').read()
f_20141029 = open('FOMC_20141029.txt','r',encoding='utf8').read()
f_20141217 = open('FOMC_20141217.txt','r',encoding='utf8').read()

f_20150128 = open('FOMC_20150128.txt','r',encoding='utf8').read()
f_20150318 = open('FOMC_20150318.txt','r',encoding='utf8').read()
f_20150429 = open('FOMC_20150429.txt','r',encoding='utf8').read()
f_20150617 = open('FOMC_20150617.txt','r',encoding='utf8').read()
f_20150729 = open('FOMC_20150729.txt','r',encoding='utf8').read()
f_20150917 = open('FOMC_20150917.txt','r',encoding='utf8').read()
f_20151028 = open('FOMC_20151028.txt','r',encoding='utf8').read()
f_20151216 = open('FOMC_20151216.txt','r',encoding='utf8').read()


# 2016~2020
f_20160127 = open('FOMC_20160127.txt','r',encoding='utf8').read()
f_20160316 = open('FOMC_20160316.txt','r',encoding='utf8').read()
f_20160427 = open('FOMC_20160427.txt','r',encoding='utf8').read()
f_20160615 = open('FOMC_20160615.txt','r',encoding='utf8').read()
f_20160727 = open('FOMC_20160727.txt','r',encoding='utf8').read()
f_20160921 = open('FOMC_20160921.txt','r',encoding='utf8').read()
f_20161102 = open('FOMC_20161102.txt','r',encoding='utf8').read()
f_20161214 = open('FOMC_20161214.txt','r',encoding='utf8').read()

f_20170201 = open('FOMC_20170201.txt','r',encoding='utf8').read()
f_20170315 = open('FOMC_20170315.txt','r',encoding='utf8').read()
f_20170503 = open('FOMC_20170503.txt','r',encoding='utf8').read()
f_20170614 = open('FOMC_20170614.txt','r',encoding='utf8').read()
f_20170726 = open('FOMC_20170726.txt','r',encoding='utf8').read()
f_20170920 = open('FOMC_20170920.txt','r',encoding='utf8').read()
f_20171101 = open('FOMC_20171101.txt','r',encoding='utf8').read()
f_20171213 = open('FOMC_20171213.txt','r',encoding='utf8').read()

f_20180131 = open('FOMC_20180131.txt','r',encoding='utf8').read()
f_20180321 = open('FOMC_20180321.txt','r',encoding='utf8').read()
f_20180502 = open('FOMC_20180502.txt','r',encoding='utf8').read()
f_20180613 = open('FOMC_20180613.txt','r',encoding='utf8').read()
f_20180801 = open('FOMC_20180801.txt','r',encoding='utf8').read()
f_20180926 = open('FOMC_20180926.txt','r',encoding='utf8').read()
f_20181108 = open('FOMC_20181108.txt','r',encoding='utf8').read()
f_20181219 = open('FOMC_20181219.txt','r',encoding='utf8').read()

f_20190130 = open('FOMC_20190130.txt','r',encoding='utf8').read()
f_20190320 = open('FOMC_20190320.txt','r',encoding='utf8').read()
f_20190501 = open('FOMC_20190501.txt','r',encoding='utf8').read()
f_20190619 = open('FOMC_20190619.txt','r',encoding='utf8').read()
f_20190731 = open('FOMC_20190731.txt','r',encoding='utf8').read()
f_20190918 = open('FOMC_20190918.txt','r',encoding='utf8').read()
f_20191030 = open('FOMC_20191030.txt','r',encoding='utf8').read()
f_20191211 = open('FOMC_20191211.txt','r',encoding='utf8').read()

f_20200129 = open('FOMC_20200129.txt','r',encoding='utf8').read()
f_20200315 = open('FOMC_20200315.txt','r',encoding='utf8').read()
f_20200429 = open('FOMC_20200429.txt','r',encoding='utf8').read()
f_20200610 = open('FOMC_20200610.txt','r',encoding='utf8').read()
f_20200729 = open('FOMC_20200729.txt','r',encoding='utf8').read()
f_20200916 = open('FOMC_20200916.txt','r',encoding='utf8').read()


#%% (3) Collect all the text as a corpus 
CORPUS = [
f_19900207, f_19900327, f_19900515, f_19900703, f_19900821, f_19901002, f_19901113, f_19901218,
f_19910206, f_19910326, f_19910514, f_19910703, f_19910820, f_19911001, f_19911105, f_19911217,
f_19920205, f_19920331, f_19920519, f_19920701, f_19920818, f_19921006, f_19921117, f_19921222,
f_19930203, f_19930323, f_19930518, f_19930707, f_19930817, f_19930921, f_19931116, f_19931221,
f_19940204, f_19940322, f_19940517, f_19940706, f_19940816, f_19940927, f_19941115, f_19941220,
f_19950201, f_19950328, f_19950523, f_19950706, f_19950822, f_19950926, f_19951115, f_19951219,
f_19960130, f_19960326, f_19960521, f_19960702, f_19960820, f_19960924, f_19961113, f_19961217,
f_19970204, f_19970325, f_19970520, f_19970701, f_19970819, f_19970930, f_19971112, f_19971216,
f_19980203, f_19980331, f_19980519, f_19980630, f_19980818, f_19980929, f_19981117, f_19981222,
f_19990202, f_19990330, f_19990518, f_19990629, f_19990824, f_19991005, f_19991116, f_19991221,
f_20000202, f_20000321, f_20000516, f_20000628, f_20000822, f_20001003, f_20001115, f_20001219,
f_20010131, f_20010320, f_20010515, f_20010627, f_20010821, f_20011002, f_20011106, f_20011211,
f_20020130, f_20020319, f_20020507, f_20020626, f_20020813, f_20020924, f_20021106, f_20021210,
f_20030129, f_20030318, f_20030506, f_20030625, f_20030812, f_20030916, f_20031028, f_20031209,
f_20040128, f_20040316, f_20040504, f_20040630, f_20040810, f_20040921, f_20041110, f_20041214,
f_20050202, f_20050322, f_20050503, f_20050630, f_20050809, f_20050920, f_20051101, f_20051213,
f_20060131, f_20060328, f_20060510, f_20060629, f_20060808, f_20060920, f_20061025, f_20061212,
f_20070131, f_20070321, f_20070509, f_20070628, f_20070807, f_20070918, f_20071031, f_20071211,
f_20080130, f_20080318, f_20080430, f_20080625, f_20080805, f_20080916, f_20081029, f_20081216,
f_20090128, f_20090318, f_20090429, f_20090624, f_20090812, f_20090923, f_20091104, f_20091216,
f_20100127, f_20100316, f_20100428, f_20100623, f_20100810, f_20100921, f_20101103, f_20101214,
f_20110126, f_20110315, f_20110427, f_20110622, f_20110809, f_20110921, f_20111102, f_20111213,
f_20120125, f_20120313, f_20120425, f_20120620, f_20120801, f_20120913, f_20121024, f_20121212,
f_20130130, f_20130320, f_20130501, f_20130619, f_20130731, f_20130918, f_20131030, f_20131218,
f_20140129, f_20140319, f_20140430, f_20140618, f_20140730, f_20140917, f_20141029, f_20141217,
f_20150128, f_20150318, f_20150429, f_20150617, f_20150729, f_20150917, f_20151028, f_20151216,
f_20160127, f_20160316, f_20160427, f_20160615, f_20160727, f_20160921, f_20161102, f_20161214,
f_20170201, f_20170315, f_20170503, f_20170614, f_20170726, f_20170920, f_20171101, f_20171213,
f_20180131, f_20180321, f_20180502, f_20180613, f_20180801, f_20180926, f_20181108, f_20181219,
f_20190130, f_20190320, f_20190501, f_20190619, f_20190731, f_20190918, f_20191030, f_20191211,
f_20200129, f_20200315, f_20200429, f_20200610, f_20200729, f_20200916]



#%% (4) Pre-processing_02 ---- pre-processing function
""" 
1. Expanding Contractions -> 2. Lower Case Transformation -> 3. Removing Non-alphabet Characters ->
4. Tokenization -> 5. Removing stop words -> 6. Stemming (Porter Stemmer)
"""

'''
pip install contractions
'''

def prepocessing_stemming(text):
    
    # 1. Expanding Contractions
    import contractions
    contraction_text = contractions.fix(text)

        
    # 2. Lower Case Transformation
    lower_text = contraction_text.lower()


    # 3. Removing Non-alphabet Characters
    import re
    remove_text = re.sub("[^a-zA-Z]", " ", lower_text) 

    
    # 4. Tokenization
    from nltk.tokenize import word_tokenize
    word_tokenize_list = word_tokenize(remove_text)

    
    # 5. Removing Stopwords
    import nltk
    def remove_stopwords(tokens):
        stopword_list = nltk.corpus.stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return filtered_tokens

    word_stopwords = remove_stopwords([word for word in word_tokenize_list])

    # 6. Stemming (Porter Stemmer)
    from nltk.stem.porter import PorterStemmer
    stemmer_porter = PorterStemmer()
    stemmed_words = [stemmer_porter.stem(word) for word in word_stopwords]
    
    return stemmed_words



#%% (5) Pre-processing_03 ---- apply the pre-processing function

FOMC_prepro = list(range(0,222))

for i in range(0,222):
    FOMC_prepro[i] = prepocessing_stemming(CORPUS[i+24])



#%% (6) Save the files to pickle-type files and open it from the pickle-type files

os.chdir(r'D:\G03_1\FOMC\FOMC_04_pickle_data')
os.getcwd()


len(FOMC_prepro[221])

# Serialization: Note that all data structures of Python need to be open in binary form
import pickle

for i in range(0,222):
    name = str(i+1)+'.txt'
    with open(name, 'wb') as file:
        pickle.dump(FOMC_prepro[i], file, pickle.HIGHEST_PROTOCOL)


FOMC_pickle = list(range(0,222))

for i in range(0,222):
    name = str(i+1)+'.txt'
    with open(name, 'rb') as file:
        FOMC_pickle[i] = pickle.load(file)

len(FOMC_pickle[221])
print(FOMC_pickle[221][0:20])
print(FOMC_pickle[0][0:20])
