from sklearn.metrics.pairwise import pairwise_distances_argmin
import shapely.geometry as geometry
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import pylab as pl
import numpy as np
import pickle
import os

def combine_points(dic):
    coordinate_lst = []

    for k, v in dic.items():
        for i in v:
            coordinate_lst.append((i[0], i[1]))
    return coordinate_lst

def plot_points(lst, image, outfile_path):
    p = 'area_labeled.png'
    plt.close('all')
    x = [tup[0] for tup in lst]
    y = [tup[1] for tup in lst]

    im = Image.open(image)
    im.load()

    color = 'lime'
    title = 'Labeled Mounds'

    order = zip(lst, color, title)
    f, ax = plt.subplots()

    # ax.set_title(title)
    ax.imshow(im, interpolation='nearest')
    for c in lst:
        x, y = c
        point = plt.Circle((x,y), 8, color=color, linewidth=2, fill=False)
        ax.add_patch(point)

    ax.set_axis_off()

    plt.tight_layout()
    # plt.show()

    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)
        plt.savefig(outfile_path+'/'+p)
    else:
        plt.savefig(outfile_path+'/'+p)


def cluster_mounds(lst, image):
    kmeans = KMeans(n_clusters=2, random_state=6,).fit(lst)

    k_means_cluster_centers = np.sort(kmeans.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(lst, k_means_cluster_centers)

    p = 'area_labeled.png'
    plt.close('all')
    im = Image.open(image)
    im.load()

    colors = ['#4EACC5', '#FF9C34']
    title = 'K-Means Clustering'
    n_clusters = 2


    order = zip(lst, colors)

    f, ax = plt.subplots()
    print(lst[:10])
    for k, col in zip(range(n_clusters), colors):
        my_members = np.where(k_means_labels == k)
        cluster_center = k_means_cluster_centers[k]
        for i in list(my_members[0]):

            ax.plot(lst[i][0], lst[i][1], 'w',
                    markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)

    ax.set_title(title)
    ax.imshow(im, interpolation='nearest')

    ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    # for c in lst:
    #     x, y = c
    #     point = plt.Circle((x,y), 8, color=color, linewidth=2, fill=False)
    #     ax.add_patch(point)

if __name__ == '__main__':
    area_dict = pickle.load(open('../data/corr_coor_dict.pkl', 'rb'))
    coordinate_lst = combine_points(area_dict)
    plot_points(coordinate_lst, '../images/jafay_area.png', '../images/')

    # cluster_mounds(coordinate_lst, '../images/jafay_area.png')
