import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import scipy.spatial
from scipy.spatial import Delaunay
import networkx as nx
import shapely
import shapely.geometry as geometry
import matplotlib
from PIL import Image
from mapping_area import combine_points
import pickle
from descartes import PolygonPatch
from shapely.geometry import shape
from shapely.ops import transform
import math
from shapely.ops import cascaded_union, polygonize
from PIL import ImageDraw

def alpha_shapes(points, alpha):
    plt.close('all')
    #http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
    if len(points) < 4:
        polygon = geometry.Polygon(lst).convex_hull
        return polygon

    def add_edge(edges, edge_points, coords, i, j ):
        if (i, j) in edges or (j, i) in edges:
            #point exits
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array(points)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    concave_hull, edge_points = cascaded_union(triangles), edge_points
    return concave_hull



def calculate_metrics(lst, polygon, image):

    map_unit = 0.19166667 #meters/pixel
    yard_conv = 1.09361 # yards/meter
    acre_conv = 4840 # yards^2/acre

    p_area = shape(polygon).area # pixels^2
    print(p_area)
    acre = (((np.sqrt(p_area) * map_unit) * yard_conv)**2)/acre_conv
    print(acre)
    p_perimeter = shape(polygon).length
    perimeter = p_perimeter * map_unit


    p = 'area_labeled.png'
    plt.close('all')
    im = Image.open(image)
    im.load()

    fig, ax = plt.subplots()
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    # ax.imshow(im, interpolation='nearest')
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)

    for c in lst:
        x, y = c
        point = plt.Circle((x,y), 8, color='lime', linewidth=2, fill=False)
        ax.add_patch(point)

    ax.add_patch(patch)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    # plt.savefig('../images/jafay_poly.png')




if __name__ == '__main__':
    area_dict = pickle.load(open('../data/corr_coor_dict.pkl', 'rb'))
    coordinate_lst = combine_points(area_dict)

    concave_hull = alpha_shapes(coordinate_lst,alpha=.0049)#87
    calculate_metrics(coordinate_lst, concave_hull, '../images/jafay_area.png')
