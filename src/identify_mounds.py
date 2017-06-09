#water shed
#blob detection skimage
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh, canny
from skimage.color import rgb2gray
from skimage import exposure
import matplotlib.pyplot as plt
from collections import Counter
import operator
from PIL import Image
import numpy as np
import os

def remove_errors(frequency, blobs):
    '''
    Removes points if they occur in a vertical row and above a set frequency
    input: An int frequency value and blob array containing the coordinates and sigma value
    output: The corrected array removing repetitious values
    '''
    #remove multiple duplicates
    unique, counts = np.unique(blobs[:, 1], return_counts=True)
    cnts = dict(zip(unique, counts))
    max_value = max(cnts.items(), key=operator.itemgetter(1))

    if max_value[1] > frequency:
        blobs_ = blobs[blobs[:,1] != max_value[0]]
        return blobs_
    else:
        blobs_ = blobs
        return blobs_


def label_mounds(filepath, outfile_path):
    '''
    input: The filepath where the images live and outfile_path where label images are sent
    output: New labeled images
    '''

    plt.close('all')
    images = os.listdir(filepath)[:-1]
    print(images)
    # images = ['0-3.png']
    blobcnt_dict = {}

    for p in images[:10]:
        im = Image.open(filepath + '/' + p)
        im.load()
        im_data = np.asarray(im)
        grey_im = rgb2gray(im_data)

        #adaptive equalization
        img_adapteq = exposure.equalize_adapthist(grey_im, clip_limit=0.008)


        blobs_log = blob_log(img_adapteq, min_sigma=2.7, max_sigma=12, threshold=.18, num_sigma = 10, overlap=.2)

        blobs_log = remove_errors(8, blobs_log)

        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
        # blobs_log.

        blob_lst = blobs_log
        color = 'lime'
        title = 'Laplacian of Gaussian Blobbing'

        order = zip(blob_lst, color, title)
        f, ax = plt.subplots()

        ax.set_title(title)
        ax.imshow(img_adapteq, interpolation='nearest')
        for b in blob_lst:
            y, x, r = b
            c = plt.Circle((x,y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()

        plt.tight_layout()

        if not os.path.exists(outfile_path):
            os.makedirs(outfile_path)
            plt.savefig(outfile_path+'/'+p)
        else:
            plt.savefig(outfile_path+'/'+p)

        blobcnt_dict[p] = len(blob_lst)


    return blobcnt_dict






if __name__ == '__main__':
    blob_cnts = label_mounds('../../Capstone_images/NN_ready_images', '../../Capstone_images/labeled_nn_images')
