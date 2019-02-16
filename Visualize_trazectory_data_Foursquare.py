# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:51:27 2018

@author: Kel3vra
"""

import pandas as pd
import datetime as dt
import pyproj
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['font.size'] = 10
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1



#Convert txt to csv New York city
nycity = []
with open("dataset_tsmc2014/dataset_TSMC2014_NYC.txt", "r") as f:
    for i in f:
        nycity.append(i)
f.close()
with open("dataset_TSMC2014_NYC.csv", "w+") as csvfile:
    csvfile.write("userID\tvenueID\tvenueCategoryID\tveneuCategory\tlatitude\tlongitude\ttimezoneOffset\tutcTimestamp\n")
    csvfile.writelines(nycity)
csvfile.close()
#Convert txt to csv Tokyo city
tkcity = []
with open("dataset_tsmc2014/dataset_TSMC2014_TKY.txt", "r") as f:
    for i in f:
        tkcity.append(i)
f.close()
with open("dataset_TSMC2014_TKY.csv", "w+") as csvfile:
    csvfile.write("userID\tvenueID\tvenueCategoryID\tveneuCategory\tlatitude\tlongitude\ttimezoneOffset\tutcTimestamp\n")
    csvfile.writelines(tkcity)
csvfile.close()
#read the csv file of New York city and Tokyo city
nyc=pd.read_csv("dataset_TSMC2014_NYC.csv", sep='\t', encoding ='ISO-8859-1')
tkc=pd.read_csv("dataset_TSMC2014_TKY.csv", sep='\t', encoding ='ISO-8859-1') 

#The data converting timestamps by splitting them to date and time
timestamp = pd.to_datetime(nyc['utcTimestamp'],errors='coerce')
nyc['new_date'] = [d.date() for d in timestamp]
nyc['new_time'] = [d.time() for d in timestamp]
nyc['datetime'] = timestamp
#The data converting timestamps by splitting them to date and time
timestamp1 = pd.to_datetime(tkc['utcTimestamp'],errors='coerce')
tkc['new_date'] = [d.date() for d in timestamp1]
tkc['new_time'] = [d.time() for d in timestamp1]
tkc['datetime'] = timestamp1

#New York city visualize the data
nyc = nyc.assign(timeOfDay=pd.cut(
                        nyc.datetime.dt.hour,
                        [-1, 12, 15, 19, 24],
                        labels=['Morning','Lunch' , 'Afternoon', 'Evening']))

nyc_venues=nyc[['veneuCategory','timeOfDay']]
nyc_venues1 = pd.crosstab(nyc['veneuCategory'],nyc['timeOfDay'], margins=True)
sorted_df = pd.DataFrame(nyc_venues1.sort_values(by = ['All'], ascending = [False]))     
df2 = sorted_df.head(40)
df2 =df2.sort_values(by = ['All'], ascending = [True])
df2 = df2.drop(['All'], axis=1)
df2 = df2.drop(['All'], axis=0)
df2.plot.barh(stacked=True);
plt.savefig('NewYork.png')
plt.show()


#Tokyo city visualize the data
tkc =tkc.assign(
        timeOfDay=pd.cut(
                tkc.datetime.dt.hour,
                [-1, 12, 15, 19, 24],
                labels=['Morning','Lunch' , 'Afternoon', 'Evening']))

tkc_venues=tkc[['veneuCategory','timeOfDay']]
tkc_venues1 = pd.crosstab(tkc['veneuCategory'],tkc['timeOfDay'], margins=True)
sorted_df = pd.DataFrame(tkc_venues1.sort_values(by = ['All'], ascending = [False]))     
df3 = sorted_df.head(40)
df3 =df3.sort_values(by = ['All'], ascending = [True])
df3 = df3.drop(['All'], axis=1)
df3 = df3.drop(['All'], axis=0)
df3.plot.barh(stacked=True);
plt.savefig('Tokyo.png')
plt.show()


