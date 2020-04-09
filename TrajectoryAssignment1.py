from typing import List, Any, Union
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import skmob
from datetime import timezone
import time
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot
from bokeh.io import output_notebook, output_file
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models.tools import HoverTool
from datetime import datetime, timedelta
from bokeh.tile_providers import CARTODBPOSITRON, STAMEN_TERRAIN
from bokeh.themes import built_in_themes
from bokeh.io import curdoc
from sympy import fft

time.time()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gpxpy

GPSdata = []
files = []
for filename in os.walk('020/Trajectory/'):
    #iles.append(filename)
    for file in range(len(filename[2])):
        print(file)
        with open('020/Trajectory/'+str(filename[2][file]), 'r') as f2:
            data = pd.read_csv(f2)
            print(data)
            #data = f2.read()
            GPSdata.append(data)
frame = pd.concat(GPSdata, axis=0, ignore_index=True)
frame = frame.dropna()

def totimestamp(dt, epoch=datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 24 * 60) / 10) * 365
# create a TrajDataFrame from a list
homeCord = [['Home', 52.163465, 4.461951, '2008-10-23 14:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 14:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 14:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 15:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 15:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 15:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 16:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 16:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 16:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 17:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 17:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 17:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 18:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 18:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 18:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 19:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 19:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 19:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 20:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 20:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 20:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 21:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 21:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 21:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 22:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 22:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 22:53:11'],
['Home', 52.163465, 4.461951, '2008-10-23 23:13:11'],
['Home', 52.163465, 4.461951, '2008-10-23 23:43:11'],
['Home', 52.163465, 4.461951, '2008-10-23 23:53:11'],
['Home', 52.163465, 4.461951, '2008-11-23 00:13:11'],
['Home', 52.163465, 4.461951, '2008-11-23 00:43:11'],
['Home', 52.163465, 4.461951, '2008-11-23 00:53:11'],
['Home', 52.163465, 4.461951, '2008-11-23 01:13:11'],
['Home', 52.163465, 4.461951, '2008-11-23 01:43:11'],
['Home', 52.163465, 4.461951, '2008-11-23 01:53:11'],
['Home', 52.163465, 4.461951, '2008-11-23 02:13:11'],
['Home', 52.163465, 4.461951, '2008-11-23 02:43:11'],
['Home', 52.163465, 4.461951, '2008-11-23 02:53:11'],
['Home', 52.163465, 4.461951, '2008-11-23 03:13:11'],
['Home', 52.163465, 4.461951, '2008-11-23 03:43:11'],
['Home', 52.163465, 4.461951, '2008-11-23 03:53:11'],
['Home', 52.163465, 4.461951, '2008-11-23 04:13:11'],
['Home', 52.163465, 4.461951, '2008-11-23 04:43:11'],
['Home', 52.163465, 4.461951, '2008-11-23 04:53:11'],
['Home', 52.163465, 4.461951, '2008-11-23 05:13:11'],
['Home', 52.163465, 4.461951, '2008-11-23 05:43:11'],
['Home', 52.163465, 4.461951, '2008-11-23 05:53:11']]

supermarketCord = [['supermarket', 52.1597, 4.4971, '2008-10-23 14:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 14:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 14:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 15:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 15:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 15:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 16:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 16:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 16:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 17:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 17:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 17:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 18:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 18:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 18:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 19:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 19:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 19:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 20:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 20:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 20:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 21:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 21:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 21:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 22:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 22:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 22:59:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 23:23:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 23:50:11'],
['supermarket', 52.1597, 4.4971, '2008-10-23 23:59:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 00:23:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 00:50:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 00:59:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 01:23:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 01:50:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 01:59:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 02:23:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 02:50:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 02:59:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 03:23:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 03:50:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 03:59:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 04:23:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 04:50:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 04:59:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 05:23:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 05:50:11'],
['supermarket', 52.1597, 4.4971, '2008-11-23 05:59:11']]

snelliusCord= [['snellius', 52.169709, 4.457111, '2008-10-23 14:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 14:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 14:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 15:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 15:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 15:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 16:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 16:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 16:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 17:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 17:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 17:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 18:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 18:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 18:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 19:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 19:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 19:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 20:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 20:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 20:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 21:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 21:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 21:50:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 22:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 22:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 22:55:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 23:05:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 23:30:11'],
['snellius', 52.169709, 4.457111, '2008-10-23 23:55:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 00:05:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 00:30:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 00:55:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 01:05:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 01:30:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 01:55:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 02:05:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 02:30:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 02:55:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 03:05:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 03:30:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 03:55:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 04:05:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 04:30:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 04:55:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 05:05:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 05:30:11'],
['snellius', 52.169709, 4.457111, '2008-11-23 05:55:11']]

homecord= pd.DataFrame.from_records(homeCord)
supermarketcord = pd.DataFrame.from_records(supermarketCord)
snelliuscord = pd.DataFrame.from_records(snelliusCord)

def simulate(homeCord, supermarketCord,snelliusCord):
    data_list = [homeCord, supermarketCord, snelliusCord]
    data = np.concatenate(data_list)
    simulatedData = skmob.TrajDataFrame(data, latitude=1, longitude=2, datetime=3)
    # print a portion of the TrajDataFrame
    simulatedData.sort_values(by=['datetime'], inplace=True, ascending=False)
    print(simulatedData.head())
    timestamp = []
    for i in simulatedData['datetime']:
        timestamp.append(totimestamp(i))
    df = pd.DataFrame(timestamp)
    simulatedData['datetime1']=df

    return simulatedData

# Create Geo Map Plot
def plotMap():
    # Show the plot embedded i
    output_notebook()

    lat = frame['Latitude'].values.tolist()
    lon = frame['Longtitude'].values.tolist()

    lst_lat = []
    lst_lon = []
    i = 0

    # Convert lat and long values inot merc_projection
    for i in range(len(lon)):
        r_major = 6378137.000
        x = r_major * math.radians(lon[i])
        scale = x/lon[i]
        y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 +
            lat[i] * (math.pi/180.0)/2.0)) * scale
        lst_lon.append(x)
        lst_lat.append(y)
        i += 1

    frame['coords_x'] = lst_lat
    frame['coords_y'] =lst_lon

    lats = frame['coords_x'].tolist()
    longs = frame['coords_y'].tolist()
    mags = frame['Field3'].tolist()

    # Create datasource
    cds = ColumnDataSource(data=dict(
        lat=lats,
        lon=longs,
        mag=mags
    ))
    # Tooltip
    TOOLTIPS = [
        ("Field3", "@mag")
    ]

    # Create figure
    geoplot = figure(title= 'Map',
               plot_width=1000,
               plot_height=500,
               x_range=(-2000000, 6000000),
               y_range=(-1000000, 7000000),
               tooltips=TOOLTIPS)

    geoplot.circle(x='lon', y='lat', fill_color='#009999', fill_alpha=1, source=cds, legend_label='Field')
    geoplot.add_tile(CARTODBPOSITRON)

    #Style the mat plot
    geoplot.title.align = 'center'
    geoplot.title.text_font_size = '20pt'
    geoplot.title.text_font = 'serif'

    #Legend
    geoplot.legend.location = 'bottom_right'
    geoplot.legend.background_fill_color='black'
    geoplot.legend.background_fill_alpha= 1
    geoplot.legend.click_policy = 'hide'
    geoplot.legend.label_text_color = 'white'
    geoplot.xaxis.visible=False
    geoplot.yaxis.visible=False
    geoplot.axis.axis_label=None
    geoplot.axis.visible=False
    geoplot.grid.grid_line_color=None
    output_file("foo.html")
    show(geoplot)
    return geoplot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

newframe=frame.iloc[:200]

BBox = ((newframe.Longtitude.min(),newframe.Longtitude.max(),newframe.Latitude.min(),newframe.Latitude.max()))

print('BBox: {}'.format(BBox))

newwc = newframe['Latitude'].value_counts()
newwc = newwc.iloc[:25]
counts = newframe.apply(pd.value_counts)
ax = newwc.plot.bar(x='Latitude', y='val',fontsize=6, rot=60)
print(newwc)
neec = newframe.groupby(["Longtitude", "Latitude"]).size()
print(neec)
ruh_m = plt.imread('map.png')

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(newframe.Longtitude, newframe.Latitude, zorder=1, alpha= 0.2, c='b', s=10)
ax.set_title('Plotting Spatial Data on Riyadh Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')



def normalize(a):
    amin, amax = min(a), max(a)
    for i,val in enumerate(a):
        a[i] = 2*(val-amin) / (amax-amin) -1
    return a

def plotSeries(dataSeries):
    df = pd.DataFrame(dataSeries['datetime'])
    df['latitude'] = dataSeries['lat']
    df.to_csv(index=False)
    #print(df)
    df.to_csv (r'timeseries.csv', index = False, header=True)
    df0 = pd.read_csv('timeseries.csv',index_col=0)
    df0.plot(rot=60)
    pyplot.show()
    return df0

import numpy
import matplotlib.pyplot as plt

#def autocorr1(data):
#autocorr=[]

datas=simulate(homecord,supermarketcord,snelliuscord)
dataSer=plotSeries(datas)
print(dataSer)
autocorr=[]
c=0
for i in range(len(dataSer['latitude'])):
    xt = dataSer['latitude'][i]
    xmean=np.mean(dataSer['latitude'])
    if i==143:
        xtk = dataSer['latitude'][i]
    else:
        xtk = dataSer['latitude'][i+1]
    rkuo = (xt - xmean)
    rkt =  (xtk - xmean)
    rku = rkuo *rkt
    rkl = np.square(xt - xmean)
    rk = rku/rkl
    autocorr.append(rk)
    #autocorr_coef.append([i,rk])
def autocorr_coef_normalize(ACF):
    autocorr_coef = []
    acr = normalize(ACF)
    for i in range(len(ACF)):
        autocorr_coef.append([i,ACF[i]])
    return autocorr_coef

def autocorr_coef_period(normautocor):
    acr_period=[]
    for v in range(len(normautocor)):
        print(normautocor[v][1])
        if normautocor[v][1] < 0 :
            x=24
        else:
            x=168
        acr_period.append([v,x])
    return acr_period




def autocorrelation(ACF,ACFN,PACF):
    meanautocorr = np.mean(ACF)
    stdautocorr = np.std(ACF)
    print('Mean Autocorrelation: {}'.format(meanautocorr))
    print('Standard Deviation: {}'.format(stdautocorr))
    dataframeACFC = pd.DataFrame(data=ACFN, columns=['A','B']);
    dataframePeriod = pd.DataFrame(data=PACF , columns=['A','B']);
    # Draw a scatter plot ACFC
    dataframeACFC.plot.scatter(x='A', y='B', title='Autocorrelation coefficients');
    plt.show(block=True);
    # Draw a scatter plot Period
    dataframePeriod.plot.scatter(x='A', y='B', title='Periodicity 24 and 168');
    plt.show(block=True);




#periodogram function
def periodogram(data_autocorr):
    from scipy import signal
    import matplotlib.pyplot as plt
    f , Pxx_spec = signal.periodogram(data_autocorr, len(autocorr)*2, 'flattop', scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('frequency')
    plt.ylabel('Linear spectrum')
    plt.show()


def addNoiseOne(data, rate): # rate has a value in [0,1] and is used as parameter to define the level of noise added
  import numpy as np
  data = np.array(data)
  noise = np.random.normal(0, rate, data.shape)
  noisydata = data + noise
  #noisydata=autocorr_coef_normalize(noisydata)
  return noisydata

def addNoiseTwo(data, rate):
    import numpy as np
    data = np.array(data)
    noise = np.random.normal(rate, 1, data.shape)
    noisydata = data + noise
    #noisydata = autocorr_coef_normalize(noisydata)
    return noisydata


#compare performance
normautocorr=autocorr_coef_normalize(autocorr)
autocorperiod=autocorr_coef_period(normautocorr)
autocorrelation(autocorr,normautocorr,autocorperiod)
periodogram(autocorr)
noisydata1 = addNoiseOne(autocorr, 0.5)
normautocorr1=autocorr_coef_normalize(noisydata1)
autocorperiod1=autocorr_coef_period(normautocorr1)
autocorrelation(noisydata1,normautocorr1,autocorperiod1)
periodogram(noisydata1)
noisydata2 = addNoiseTwo(autocorr, 0.5)
normautocorr2=autocorr_coef_normalize(noisydata2)
autocorperiod2=autocorr_coef_period(normautocorr2)
autocorrelation(noisydata2,normautocorr2,autocorperiod2)
periodogram(noisydata2)
plotMap()
#PeriodogramPeridicity(noisydata2)
#GPS autocorrelation etc.
normautocorre=autocorr_coef_normalize(newframe['Latitude'])
autocorperiodo=autocorr_coef_period(normautocorr)
autocorrelation(newframe['Latitude'],normautocorre,autocorperiodo)
periodogram(newframe['Latitude'])
