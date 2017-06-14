from identify_mounds import *
from PIL import Image
import numpy as np
np.random.seed(6)
import os
import subprocess
from collections import defaultdict
import pickle
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


def labeler(dic):
    '''
    A Helper funtion to label the mound images( 1: mound, 0: not_mound) as they appear and save them to a dictionary.
    input: dic
    output: Pickeled dictionary containing labels for each mound
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
                    print('You have {} left to label for this image.'.format(len(im_lst)-(i+1)))
                    dic[image]['label'].append(label_input)
                    print(dic[image]['label'])
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



def nomound_mound(X, y):
    #once you have this trained you can then just pass in the images and identify each one.
    '''
    Classifies the input 15X15 image as a prairie dog mount or not
    input: prairie dog mound images
    output: returns the coordinates of not prairie dog mound
    '''
    pass


def train_nomound_mound(dic):
    X = []
    y = []
    X_ = []
    #mounds = 165, not_mounds = 614
    for im in dic:
        X.extend(dic[im]['img_lst'])
        y.extend(dic[im]['label'])

    for im in X:
        arr = imread(im)
        X_.append(arr)

    X_ = (np.array(X_).reshape(779, 4, 16, 16)/255).astype('float32')
    y = np.array(y).astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X_, y, stratify = y)

    model = Sequential()

    batch_size = 128
    nb_epoch = 12

    nb_filters = 32 #neurons
    kernel_size = (3, 3)
    input_shape = (4, 16, 16)
    pool_size = (2, 2)

    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), input_shape=input_shape, kernel_initializer='normal'))
    model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.50))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])


    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])










if __name__ == '__main__':
    # blobInfo_dict = label_mounds('../../Capstone_images/NN_ready_images', '../../Capstone_images/labeled_nn_images')
    #
    # img_lst = blobInfo_dict.keys()
    # blobInfo_dict = crop_mound(img_lst, blobInfo_dict, '../../Capstone_images/NN_ready_images')

    blobInfo_dict = pickle.load(open('../data/label_data.pkl', 'rb'))
    # train_nomound_mound(blobInfo_dict)
