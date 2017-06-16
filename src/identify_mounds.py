#water shed
#blob detection skimage
from train_classifier import crop_mound
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh, canny
from skimage.color import rgb2gray
from skimage import exposure
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
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


def label_blobs(filepath, outfile_path=None):
    '''
    input: The filepath where the images live and outfile_path where label images are sent if you need the images saved.
    output: New labeled images
    '''

    plt.close('all')
    images = os.listdir(filepath)
    # images = ['0-3.png']
    blobInfo_dict = defaultdict(dict)

    for p in images:
        im = Image.open(filepath + '/' + p)
        im.load()
        im_data = np.asarray(im)
        grey_im = rgb2gray(im_data)
        gamma_corrected = exposure.adjust_gamma(grey_im, 5)
        p2, p98 = np.percentile(gamma_corrected, (10, 98))
        img_rescale = exposure.rescale_intensity(gamma_corrected, in_range=(p2, p98))


        #Laplacian of Gaussian (LoG) method
        blobs_log = blob_log(img_rescale, min_sigma=2, max_sigma=10, threshold=.25, overlap=.1)


        # blobs_log = remove_errors(8, blobs_log)

        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        blob_lst = blobs_log

        value = blobInfo_dict.get(p, None)
        if value is None:
            blobInfo_dict[p]['coordinates'] = blob_lst

        # color = 'lime'
        # title = 'Laplacian of Gaussian Blobbing'
        #
        # order = zip(blob_lst, color, title)
        # f, ax = plt.subplots()
        #
        # ax.set_title(title)
        # ax.imshow(im, interpolation='nearest')
        # for b in blob_lst:
        #     y, x, r = b
        #     c = plt.Circle((x,y), r, color=color, linewidth=2, fill=False)
        #     ax.add_patch(c)
        # ax.set_axis_off()
        #
        # plt.tight_layout()
        #
        # if not os.path.exists(outfile_path):
        #     os.makedirs(outfile_path)
        #     plt.savefig(outfile_path+'/'+p)
        # else:
        #     plt.savefig(outfile_path+'/'+p)




    return blobInfo_dict






if __name__ == '__main__':
    blobInfo_dict = label_mounds('../../Capstone_images/NN_ready_images', '../../Capstone_images/labeled_nn_images')
