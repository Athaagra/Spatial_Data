# Load the Pandas libraries with alias 'pd'
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from sklearn.cluster import KMeans
from bokeh.tile_providers import CARTODBPOSITRON
# Read data from file 'filename.csv'
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, LogColorMapper
from bokeh.models import MultiPolygons, Plot
import geopandas as gpd
from bokeh.models import ColumnDataSource, Range1d
#import pysal as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from bokeh.palettes import RdYlBu11 as palette
from bokeh.models import LogColorMapper
import libpysal
import esda
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import numpy as np
def getPointCoords(row, geom, coord_type):
    """Calculates coordinates ('x' or 'y') of a Point geometry"""
    if coord_type == 'x':
        return row[geom].x
    elif coord_type == 'y':
        return row[geom].y
def getLineCoords(row, geom, coord_type):
    """Returns a list of coordinates ('x' or 'y') of a LineString geometry"""
    if coord_type == 'x':
        return list( row[geom].coords.xy[0])
    elif coord_type == 'y':
        return list( row[geom].coords.xy[1])
def getPolyCoords(row, geom, coord_type):
    """Returns the coordinates ('x' or 'y') of edges of a Polygon exterior"""
    # Parse the exterior of the coordinate
    exterior = row[geom].exterior
    if coord_type == 'x':
        # Get the x coordinates of the exterior
        return list( exterior.coords.xy[0])
    elif coord_type == 'y':
        # Get the y coordinates of the exterior
        return list( exterior.coords.xy[1])

A_pi = r"data/southAfrica/healthsites.shp"
A_gi = r"South_Africa_Polygon.shp"
# Calculate x coordinates of the line
A_p = gpd.read_file(A_pi)
#A_r = gpd.read_file(A_ro)
A_g = gpd.read_file(A_gi)
print(A_g)
A_p['x'] = A_p.apply(getPointCoords, geom='geometry', coord_type='x', axis=1)
# Calculate x,y coordinates of the line
A_p['y'] = A_p.apply(getPointCoords, geom='geometry', coord_type='y', axis=1)
# Calculate x,y coordinates of the polygon
A_g['x'] = A_g.apply(getPolyCoords, geom='geometry', coord_type='x', axis=1)
# Calculate x,y coordinates of the line
A_g['y'] = A_g.apply(getPolyCoords, geom='geometry', coord_type='y', axis=1)
p_df = A_p.drop('geometry', axis=1).copy()
g_df = A_g.drop('geometry', axis=1).copy()
print(g_df)
#g_df = A_g.drop('geometry', axis=1).copy()
#gsource = ColumnDataSource(g_df)

datapoints = p_df['healthcare']
d_n=[]
colors=[]
for i in range(len(datapoints)):
    if datapoints[i]=='hospital':
        value=1
        color='red'
    elif datapoints[i]=='clinic':
        value=2
        color='blue'
    elif datapoints[i]=='dentist':
        value=3
        color='orange'
    else:
        value=0
        color='yellow'
    d_n.append(value)
    colors.append(color)

p_df['CL']= pd.DataFrame(data = colors , columns=['CL'])
psource = ColumnDataSource(p_df)
gsource = ColumnDataSource(g_df)
from bokeh.palettes import RdYlBu11 as palette
from bokeh.models import LogColorMapper
# Create the color mapper
color_mapper = LogColorMapper(palette=palette)
# Initialize our figure
p = figure(title="South-Africa Network")

# Plot grid
# Add schools on top (as yellow points)
p.circle('x', 'y', size=3, source=psource,
         color='CL')
p.patches('x', 'y', source=gsource,
         fill_color={'field': 'pop', 'transform': color_mapper},
         fill_alpha=1.0, line_color="black", line_width=0.05)
# let's also add the hover over info tool
tooltip = HoverTool()
tooltip.tooltips = [('Healthcare Type', '@healthcare'),
#                    ('Type of road', '@TYYP'),
                    ('Fcode', '@f_code')]
p.add_tools(tooltip)

# Save the figure
outfp = r"data\roads_pop_kmad_map.html"
output_file(outfp)
show(p)
print(d_n)
fmatrix=[]
for i in range(len(p_df['x'])):
    #print(p_df['x'][i])
    #print(int(str(p_df['x'][i])[:2]))
    for d in range(len(p_df['x'])):
        #print(p_df['x'][d])
        init = int(str(p_df['x'][i])[:2])
        nvar = int(str(p_df['x'][d])[:2])
        if init==nvar:
            neigh = p_df['x'][d]
        else:
            neigh=0
        fmatrix.append([i,p_df['x'][i],neigh])
print(len(fmatrix))
print(fmatrix)
matrix = np.array(fmatrix)
Dataset=[]
for i in range(len(p_df['x'])):
    result = np.where(matrix == i)
    neighboors = matrix[result[0]]
    nei=[]
    for q in range(len(neighboors[:,2])):
        if  neighboors[:,2][q] > 0:
            nei.append(q)
    nn = np.array(nei)
    Stand = len(nn)
    Standarization=[]
    for b in nn:
        resultie = 1/Stand * neighboors[:,2][b]
        Standarization.append(resultie)
    total = sum(Standarization)
    Dataset.append([i,p_df['x'][i],total,d_n[i]])

print(Dataset)
dfD = pd.DataFrame(Dataset)
dfD = dfD.sort_values(dfD.columns[3] , ascending=False)
dfDtr =dfD[:125]
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 20]
x=dfDtr[dfDtr.columns[1]]
print(x)
y=dfDtr[dfDtr.columns[2]]
print(y)
print(len(x))
print(len(y))
df = pd.DataFrame({'Spatial_Lag':y,
                   'log':x})
ax = df.plot.bar(x='log', y='Spatial_Lag',fontsize=8, rot=90)
dfDte =dfD[125:]
from sklearn import svm
Xtr =np.array(dfDtr[dfDtr.columns[1]]).reshape(125,1)
Xtr1 =np.array(dfDtr[dfDtr.columns[2]]).reshape(125,1)
Xtrain = np.hstack((Xtr,Xtr1))
ytr = np.array(dfDtr[dfDtr.columns[3]]).reshape(125,1)
Xte = np.array(dfDte[dfDte.columns[1]]).reshape(1226,1)
Xte1 = np.array(dfDte[dfDte.columns[2]]).reshape(1226,1)
#yte = np.array(dfDte[dfDte.columns[3]]).reshape(1226,1)
Xtest = np.hstack((Xte,Xte1))
clf = svm.SVC()
clf.fit(Xtrain, ytr)
yte=clf.predict(Xtest)
yte = np.array(yte).reshape(1226,1)
Train =np.hstack((Xtrain,ytr))
Test = np.hstack((Xtest,yte))
Datasetl = np.vstack((Train,Test))
xplt = list(p_df['x'])
data = list(dfD[dfD.columns[2]])
#
#lista = m_df['DIST_KM'][:1936]
print(len(xplt))
xplti = xplt[:1296]
w = libpysal.weights.lat2W(36,36)
w.transform = "R"
y1 = libpysal.weights.lag_spatial(w,xplti)
spatial_auto = esda.Moran(y1, w)
print(spatial_auto.I)
import seaborn as sns
# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot values
sns.regplot(x=xplti, y=y1, data=xplti)
# Display
plt.show()
colorsl=[]
for i in range(len(Datasetl)):
    if Datasetl[:,2][i]==1:
        color='red'
    elif Datasetl[:,2][i]==2:
        color='blue'
    elif Datasetl[:,2][i]== 3:
        color='orange'
    else:
        value=0
        color='yellow'
    colorsl.append(color)

yplt = list(p_df['y'])
yplt=np.array(yplt).reshape(len(yplt),1)
colorsl=np.array(colorsl).reshape(len(colorsl),1)
ycol = np.hstack((yplt,colorsl))
Datasetl=np.hstack((Datasetl,ycol))
df_l= pd.DataFrame(Datasetl, columns=['x','spatial_lag','type','y','color'])
sortx = np.array(p_df['x']).reshape(len(colorsl),1)
print(len(sortx))
colorll=[]
for i in xplt:
    print(i)
    colora=colorsl[np.where(sortx == i)]
    print(colora)
    colorll.append(str(colora[0]))
p_df['CL']= pd.DataFrame(data = colorll, columns=['CL'])
psourcel = ColumnDataSource(p_df)
from bokeh.palettes import RdYlBu11 as palette
from bokeh.models import LogColorMapper
# Create the color mapper
color_mapper = LogColorMapper(palette=palette)
# Initialize our figure
p = figure(title="South-Africa Network")
# Add schools on top (as yellow points)
p.circle('x', 'y', size=3, source=psourcel,
         color='CL')
p.patches('x', 'y', source=gsource,
         fill_color={'field': 'pop', 'transform': color_mapper},
         fill_alpha=1.0, line_color="black", line_width=0.05)
# let's also add the hover over info tool
tooltip = HoverTool()
tooltip.tooltips = [('Healthcare Type', '@healthcare'),
#                    ('Type of road', '@TYYP'),
                    ('Fcode', '@f_code')]
p.add_tools(tooltip)

# Save the figure
outfp = r"data\roads_pop_kmad_asap.html"
output_file(outfp)
show(p)























