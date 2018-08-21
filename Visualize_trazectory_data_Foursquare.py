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
import plots as pl



nycity = []
with open("dataset_tsmc2014/dataset_TSMC2014_NYC.txt", "r") as f:
    for i in f:
        nycity.append(i)
f.close()
with open("dataset_TSMC2014_NYC.csv", "w+") as csvfile:
    csvfile.write("userID\tvenueID\tvenueCategoryID\tveneuCategory\tlatitude\tlongitude\ttimezoneOffset\tutcTimestamp\n")
    csvfile.writelines(nycity)
csvfile.close()

tkcity = []
with open("dataset_tsmc2014/dataset_TSMC2014_TKY.txt", "r") as f:
    for i in f:
        tkcity.append(i)
f.close()
with open("dataset_TSMC2014_TKY.csv", "w+") as csvfile:
    csvfile.write("userID\tvenueID\tvenueCategoryID\tveneuCategory\tlatitude\tlongitude\ttimezoneOffset\tutcTimestamp\n")
    csvfile.writelines(tkcity)
csvfile.close()
nyc=pd.read_csv("dataset_TSMC2014_NYC.csv", sep='\t', encoding ='ISO-8859-1')
tkc=pd.read_csv("dataset_TSMC2014_TKY.csv", sep='\t', encoding ='ISO-8859-1') 

timestamp = pd.to_datetime(nyc['utcTimestamp'],errors='coerce')
nyc['new_date'] = [d.date() for d in timestamp]
nyc['new_time'] = [d.time() for d in timestamp]
nyc['datetime'] = timestamp

timestamp1 = pd.to_datetime(tkc['utcTimestamp'],errors='coerce')
tkc['new_date'] = [d.date() for d in timestamp1]
tkc['new_time'] = [d.time() for d in timestamp1]
tkc['datetime'] = timestamp1


nyc = nyc.assign(timeOfDay=pd.cut(
                        nyc.datetime.dt.hour,
                        [-1, 12, 15, 19, 24],
                        labels=['Morning','Lunch' , 'Afternoon', 'Evening']))

nyc_venues=nyc[['veneuCategory','timeOfDay']]
nyc_venues1 = pd.crosstab(nyc['veneuCategory'],nyc['timeOfDay'])
sorted_df = pd.DataFrame(nyc_venues1.sort_values(by = ['Morning','Lunch' , 'Afternoon', 'Evening'], ascending = [False,False,False,False]))     
df2 = sorted_df.head(40)
df2.sort_values(by = ['Morning','Lunch' , 'Afternoon', 'Evening'], ascending = [True,True,True,True]).plot.barh(stacked=True);












tkc.assign(
    timeOfDay=pd.cut(
        tkc.datetime.dt.hour,
        [-1, 12, 15, 19, 24],
        labels=['Morning','lunch' , 'Afternoon', 'Evening']))





