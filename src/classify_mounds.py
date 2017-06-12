from identify_mounds import *
from PIL import Image
import numpy as np
import os
import subprocess
from collections import defaultdict
import pickle


def labeler(dic):
    '''
    A Helper funtion to label the mound images( 1: mound, 0: not_mound) as they appear and save them to a dictionary.
    input: a list of images
    output list of list containting labels for each mound
    '''
    image_lst = dic.keys()

    for image in image_lst:
        full_display = Image.open('../../Capstone_images/NN_ready_images/'+image)
        full_display.show()
        dic[image]['label'] = []
        im_lst = dic[image]['img_lst']

        for i,im in enumerate(im_lst):
            im = Image.open(im)
            im.show()
            y = dic[image]['coordinates'][i][0]
            x = dic[image]['coordinates'][i][1]
            print("We are looking at image {}, at coordinate {}, {}".format(image, x, y))
            while True:
                label_input = int(input('Is this a prairie dog mound?'))
                print(label_input)
                if label_input == 1 or label_input == 0:
                    print("You entered a proper value!")
                    dic[image]['label'].append(label_input)
                    break
                elif label_input != 1 or label_input != 0:
                    print("This value has to be entered as a 1 or a 0.")
            im.close()

        full_display.close()
    pickle.dump(dic, open('../data/label_data.pkl', 'wb'))


def crop_mound(image_lst, dic, filepath):
    '''
    This function will take in an image, iterable of blob coordinates and crop a 16x16 pixel image around the coordinates.
    input: image name, iterable of pixel coordinates and directory filpath
    output: A 16x16 image of the mound
    '''

    for image in image_lst:
        im = Image.open(filepath + '/' + image).copy()
        outfile_path = '../../Capstone_images/mound_imgs/'
        coordinates = dic[image]['coordinates']
        dic[image]['img_lst'] = []

        for i, c in enumerate(coordinates):
            y = c[0]
            x = c[1]
            ux, uy, lx, ly = x - 16/2, y - 16/2, x + 16/2, y + 16/2
            bounds = [ux, uy, lx, ly]
            for v in bounds:
                if v < 226:
                    v = v
                else:
                    v = 266

            c = im.crop((ux, uy, lx, ly))

            if not os.path.exists(outfile_path):
                os.makedirs(outfile_path)
                c.save(outfile_path + '{}-{}'.format(x,y) + '-' + image)
                dic[image]['img_lst'].append(outfile_path + '{}-{}'.format(x,y) + '-' + image)
            else:
                c.save(outfile_path + '{}-{}'.format(x,y) + '-' + image)
                dic[image]['img_lst'].append(outfile_path + '{}-{}'.format(x,y) + '-' + image)

        #this will be where your model is sent the images to classify and will return the new list of blob coordinates.

        return dic



def nomound_mound(images):
    #once you have this trained you can then just pass in the images and identify each one.
    '''
    Classifies the input 15X15 image as a prairie dog mount or not
    input: prairie dog mound images
    output: returns the coordinates of not prairie dog mound
    '''
    pass




if __name__ == '__main__':
    blobInfo_dict = label_mounds('../../Capstone_images/NN_ready_images', '../../Capstone_images/labeled_nn_images')

    img_lst = blobInfo_dict.keys()
    blobInfo_dict = crop_mound(img_lst, blobInfo_dict, '../../Capstone_images/NN_ready_images')

    blobInfo_dict = labeler(blobInfo_dict)
