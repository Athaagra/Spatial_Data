#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 05:42:35 2022

@author: Optimus
"""

from pysal.model import spreg
from pysal.lib import weights
from pysal.explore import esda
import hvplot.pandas
from scipy import stats
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import osmnx as ox
import pysal as ps
from pyrosm import OSM, get_data
sns.set(style="whitegrid")

# Read listings
fp = "listings.csv.gz"
data = pd.read_csv(fp)
columnss=data.columns


# Read OSM data - get administrative boundaries

# define the place query
query = {'city': 'Austin'}

# get the boundaries of the place (add additional buffer around the query)
boundaries = ox.geocode_to_gdf(query, buffer_dist=5000)

# Let's check the boundaries on a map
boundaries.explore()


# Create a GeoDataFrame
data["geometry"] = gpd.points_from_xy(data["longitude"], data["latitude"])
data = gpd.GeoDataFrame(data, crs="epsg:4326")

# Filter geographically
data = gpd.sjoin(data, boundaries[["geometry"]])
data = data.reset_index(drop=True)

# Check the first rows
data.head()

testi=data
#data.hvplot(geo=True, tiles="OSM", alpha=0.5, width=600, height=600, hover_cols=["name"])
uniquezone=testi['ZONE'].unique()

t=[]
for i in range(len(testi['ZONE'])):
    if testi['ZONE'][i]==uniquezone[0]:
       t.append(1)
    if testi['ZONE'][i]==uniquezone[1]:
       t.append(2)
        #testi['ZONEOneHot'][i]=2
    if testi['ZONE'][i]==uniquezone[2]:
       t.append(3)
        #testi['ZONEOneHot'][i]=3
    if testi['ZONE'][i]==uniquezone[3]:
       t.append(4)
        #testi['ZONEOneHot'][i]=4
    if testi['ZONE'][i]==uniquezone[4]:
        t.append(5)
       #testi['ZONEOneHot'][i]=5
    if testi['ZONE'][i]==uniquezone[5]:
        t.append(6)
       #testi['ZONEOneHot'][i]=6
    if testi['ZONE'][i]==uniquezone[6]:
        t.append(7)
       #testi['ZONEOneHot'][i]=7
    if testi['ZONE'][i]==uniquezone[7]:
        t.append(8)
       #testi['ZONEOneHot'][i]=8
    if testi['ZONE'][i]==uniquezone[8]:
        t.append(9)
       #testi['ZONEOneHot'][i]=9
    if testi['ZONE'][i]==uniquezone[9]:
        t.append(10)
       #testi['ZONEOneHot'][i]=10
    if testi['ZONE'][i]==uniquezone[10]:
        t.append(11)
       #testi['ZONEOneHot'][i]=11
    if testi['ZONE'][i]==uniquezone[11]:
        t.append(12)
       #testi['ZONEOneHot'][i]=12

uniqueiucn=testi['IUCN'].unique()

q=[]
for i in range(len(testi['IUCN'])):
    if testi['IUCN'][i]==uniqueiucn[0]:
       q.append(1)
    if testi['IUCN'][i]==uniqueiucn[1]:
       q.append(2)
    if testi['IUCN'][i]==uniqueiucn[2]:
       q.append(3)
    if testi['IUCN'][i]==uniqueiucn[3]:
       q.append(4)

uniquestat=testi['STATUS'].unique()

v=[]
for i in range(len(testi['STATUS'])):
    if testi['STATUS'][i]==uniquestat[0]:
       v.append(1)
    if testi['STATUS'][i]==uniquestat[1]:
       v.append(2)
    if testi['STATUS'][i]==uniquestat[2]:
       v.append(3)
    if testi['STATUS'][i]==uniquestat[3]:
       v.append(4)
    if testi['STATUS'][i]==uniquestat[4]:
       v.append(5)
    if testi['STATUS'][i]==uniquestat[5]:
       v.append(6)
    if testi['STATUS'][i]==uniquestat[6]:
       v.append(7)
    if testi['STATUS'][i]==uniquestat[7]:
       v.append(8)
    if testi['STATUS'][i]==uniquestat[8]:
       v.append(9)
    if testi['STATUS'][i]==uniquestat[9]:
       v.append(10)
    if testi['STATUS'][i]==uniquestat[10]:
       v.append(11)
    if testi['STATUS'][i]==uniquestat[11]:
       v.append(12)
    if testi['STATUS'][i]==uniquestat[12]:
       v.append(13)
    if testi['STATUS'][i]==uniquestat[13]:
       v.append(14)
    if testi['STATUS'][i]==uniquestat[14]:
       v.append(15)
    if testi['STATUS'][i]==uniquestat[15]:
       v.append(16)
    if testi['STATUS'][i]==uniquestat[16]:
       v.append(17)
    #if testi['STATUS'][i]==uniquestat[17]:
    #   q.append(18)

testi['ZONEHot']=t
testi['IUCNHot']=q
testi['STATUSHot']=v
explanatory_vars = ['ZONEHot', 'IUCNHot','STATUSHot']

def has_pool(a):
    if 'Pool' in a:
        return 1
    else:
        return 0
    
data['pool'] = data['amenities'].apply(has_pool)

data["price"].head()

# Remove dollar sign and the thousand separator (comma, e.g. 1000,000.00) and convert to float
#yxs = data.loc[:,explanatory_vars + ['pool', 'price']].dropna()

# Remove dollar sign and the thousand separator (comma, e.g. 1000,000.00) and convert to float
#data["price"] = data["price"].str.replace("$", '', regex=True).str.replace(",", "").astype(float)
#data["log_price"] = np.log(data["price"] + 0.000001)

all_model_attributes = ['Area_km2'] + explanatory_vars
has_nans = False
for attr in all_model_attributes:
    if testi[attr].hasnans:
        has_nans = True
print("Has missing values:", has_nans)


data = testi.dropna(subset=all_model_attributes).copy()

w = weights.KNN.from_dataframe(data, k=8)
w.transform = 'R'


m1 = spreg.OLS(data[['Area_km2']].values, data[explanatory_vars].values, 
                  name_y = 'Area_km2', name_x = explanatory_vars)

print(m1.summary)

uniquenet=testi['Network'].unique()

net=[]
for i in range(len(testi['Network'])):
    if testi['Network'][i]==uniquenet[0]:
       net.append(1)
    if testi['Network'][i]==uniquenet[1]:
       net.append(2)
    if testi['Network'][i]==uniquenet[2]:
       net.append(3)
    if testi['Network'][i]==uniquenet[3]:
       net.append(4)
    if testi['Network'][i]==uniquenet[4]:
       net.append(5)
    if testi['Network'][i]==uniquenet[5]:
       net.append(6)
    if testi['Network'][i]==uniquenet[6]:
       net.append(7)
   

testi['NetworkHot']=net
# Create weigts
w_pool = weights.KNN.from_dataframe(data, k=8)
# Assign spatial lag based on the pool values
lagged = data.assign(w_pool=weights.spatial_lag.lag_spatial(w_pool, testi['NetworkHot'].values))
lagged.head()
# =============================================================================
# #Spatially lagged exogenous regressors
# =============================================================================
# Add pool to the explanatory variables

extended_vars = explanatory_vars + ['NetworkHot']
m2 = spreg.OLS(lagged[['Area_km2']].values, lagged[extended_vars].values, 
               name_y = 'Area_km2', name_x = extended_vars)
print(m2.summary)
# =============================================================================
# Spatially lagged endogenous regressors 
# =============================================================================
#variables = explanatory_vars + ["pool"]
#m3 = spreg.GM_Lag(data[['log_price']].values, data[variables].values, 
#                  w=w,
#                  name_y = 'ln(price)', name_x = variables)
#print(m3.summary)
# =============================================================================
# Prediction performance of spatial models
# =============================================================================
from sklearn.metrics import mean_squared_error as mse
mses = pd.Series({'OLS': mse(data["Area_km2"], m1.predy.flatten()), \
                  'OLS+W': mse(data["Area_km2"], m2.predy.flatten())#, \
#                  'Lag': mse(data["log_price"], m3.predy_e)
                    })
mses.sort_values()
