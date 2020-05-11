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
import csv

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

def getLineCoords(row, geom, coord_type):
    """Returns a list of coordinates ('x' or 'y') of a LineString geometry"""
    if coord_type == 'x':
        return list( row[geom].coords.xy[0])
    elif coord_type == 'y':
        return list( row[geom].coords.xy[1])

#polygon= []

#with open('polygonSA - Copy.csv') as csvDataFile:
#    csvReader = csv.reader(csvDataFile)
#    print(csvReader)
#    for row in csvReader:
#        polygon.append(row)

polygon = pd.read_csv('polygonSA - Copy.csv')
def extract(coo, polygon):
    fstrow = polygon[0:27510]
    ndrow = polygon[27510:55083]
    thrdrow = polygon[55083:60446]
    fothrow = polygon[60446:87947]
    fifrow = polygon[87947:107764]
    sixthrow = polygon[107764:125210]
    svrow = polygon[125210:146276]
    eigrow = polygon[146276:173824]
    ninthrow = polygon[173824:201374]
    z=fstrow[coo].to_numpy()
    o=ndrow[coo].to_numpy()
    t=thrdrow[coo].to_numpy()
    th=fothrow[coo].to_numpy()
    fo=fifrow[coo].to_numpy()
    fi=sixthrow[coo].to_numpy()
    si=svrow[coo].to_numpy()
    se=eigrow[coo].to_numpy()
    ei=ninthrow[coo].to_numpy()
    z = z[0:5363]
    o = o[0:5363]
    t = t[0:5363]
    th = th[0:5363]
    fo = fo[0:5363]
    fi = fi[0:5363]
    si = si[0:5363]
    se = se[0:5363]
    ei = ei[0:5363]
    z.shape=(1,5363)
    o.shape=(1,5363)
    t.shape=(1,5363)
    th.shape=(1,5363)
    fo.shape=(1,5363)
    fi.shape=(1,5363)
    si.shape=(1,5363)
    se.shape=(1,5363)
    ei.shape=(1,5363)
    matrix = np.vstack((z[0], o[0], t[0], th[0], fo[0], fi[0], si[0], se[0], ei[0]))
    return matrix
#File path
buildSouthAfr= r"data/southAfricaTwo/builtupa_zaf.shp"
bSAfr = gpd.read_file(buildSouthAfr)
print(bSAfr)
#matrix_x = extract('x', polygon)
#matrix_y = extract('y', polygon)
#print(bSAfr['geometry'].iloc[1])
#bSAfr['x'] = matrix_x.tolist()
#bSAfr['y'] = matrix_y.tolist()
#polygons = len(bSAfr['geometry'])
bSAfr['x'] = bSAfr.apply(getLineCoords, geom='geometry', coord_type='x', axis=1)
# Calculate x,y coordinates of the line
bSAfr['y'] = bSAfr.apply(getLineCoords, geom='geometry', coord_type='y', axis=1)

m_df = bSAfr.drop('geometry', axis=1).copy()
psource = ColumnDataSource(m_df)

from bokeh.palettes import RdYlBu11 as palette
from bokeh.models import LogColorMapper
# Create the color mapper
color_mapper = LogColorMapper(palette=palette)
# Initialize our figure
p = figure(title="South-Africa Network")

# Plot grid
# Add schools on top (as yellow points)
p.multi_line('x', 'y', source=psource, color='red', line_width=1)
# let's also add the hover over info tool
tooltip = HoverTool()
p.add_tools(tooltip)

# Save the figure
outfp = r"data\roads_pop_kmad_aap.html"
output_file(outfp)
show(p)

#for i in range(0,polygons):
#    Row=bSAfr['geometry'].iloc[9]
#    print(Row)
#    row = Row.wkt
#    rows=len(row)/96.363
#    print(len(row))
#    print(int(rows))
#polygons = len(bSAfr['geometry'])+1
#for i in range(1, 2): #polygons):
#    one_point=bSAfr['geometry'].iloc[i]
#    print(one_point)
#    b=10
#    c=28
#    print(i)
#    for g in range(10, 27520):
#        xd = one_point.wkt[b:b + point]
#        b= b+point+21
#        yd = one_point.wkt[c:c + point]
#        c= c+point+21
#        print(g)
#        print(xd)
#        print(yd)
#     #   x.append(float(xd))
     #   y.append(float(yd))
    #y = one_point.wkt[point+1:]
#print(bSAfr)



#grid_Afr = r"data/caf_admbnda_adm1_200k_sigcaf_reach_itos_v2.shx"
#grid_Afr = gpd.read_file(grid_Afr)
#print(grid_Afr['geometry']['exterior'])
#exterior = grid_Afr['geometry'].exterior
#print(exterior)
#print(grid_Afr['boundary'])
#print(exterior.coords.xy[0])
#un =grid_Afr.unary_union
#print(un.exterior.coords.xy[0])



