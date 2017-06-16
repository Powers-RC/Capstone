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
from sklearn.metrics import confusion_matrix



def labeler(dic):
    '''
    A Helper funtion to label the mound images( 1: mound, 0: not_mound) as they appear and save them to a dictionary.
    input: dic
    output: Pickeled dictionary containing labels for each mound
    '''
    # image_lst =

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
                    pickle.dump(dic, open('../data/label_data2.pkl', 'wb'))
                    break
                elif label_input != 1 or label_input != 0:
                    print("This value has to be entered as a 1 or a 0.")
            im.close()

        full_display.close()
    pickle.dump(dic, open('../data/label_data.pkl', 'wb'))


def crop_mound(dic, filepath, outfile_path):
    '''
    This function will take in an iterable containing blob coordinates and crop a 16x16 pixel image around the coordinates.
    input: dictionary of pixel coordinates and directory filpath to where those images live.
    output: A 16x16 image of the mound
    '''
    image_lst = os.listdir(filepath)

    for image in image_lst:
        im = Image.open(filepath + '/' + image).copy()
        outfile_path = outfile_path
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
                folder = outfile_path
                c.save(outfile_path + '{}-{}'.format(x,y) + '-' + image)
                dic[image]['img_lst'].append(outfile_path + '{}-{}'.format(x,y) + '-' + image)

    filename = '../data/dict_data.pkl'
    if not os.path.isfile(filename):
        pickle.dump(dic, open(filename, 'wb'))
    else:
        pickle.dump(dic, open(filename, 'wb'))

        #this will be where your model is sent the images to classify and will return the new list of blob coordinates.


def train_nomound_mound(dic):
    X = []
    y = []
    X_ = []
    for im in dic:
        X.extend(dic[im]['img_lst'])
        y.extend(dic[im]['label'])
    for im in X:
        arr = imread(im)
        X_.append(arr)
        len_X = len(X_)

    X_ = (np.array(X_).reshape(len_X, 4, 16, 16)/255).astype('float32')
    y = np.array(y).astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X_, y, stratify = y)

    model = Sequential()

    batch_size = 20
    nb_epoch = 20

    nb_filters = 300 #neurons
    kernel_size = (3, 3)
    input_shape = (4, 16, 16)


    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), input_shape=input_shape, kernel_initializer='TruncatedNormal'))
    model.add(Activation('relu'))


    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])


    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))

    filename = '../data/mound_classifier.HDF5'
    model.save(filename)

    # score = model.evaluate(X_test, y_test, verbose=0)

    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    #
    # y_predict = model.predict_classes(X_test, batch_size=batch_size)
    #
    # conf_mat = confusion_matrix(y_test, y_predict)
    # print(conf_mat)
    #
    # precision = conf_mat[1,1]/ (conf_mat[1,1] +conf_mat[0,1])
    # print('Precision, PPV: {}'.format(precision))
    #
    # recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0 ])
    # print('Recall, TPR: {}'.format(recall))
    #
    # #weighted average of percision and recall
    # f1_score = 2 * (precision * recall)/(precision + recall)
    # print('The F1_score for this model is {}'.format(f1_score))
    #
    # print('The number of mounds in test set is: {}'.format(sum(y_test)))





if __name__ == '__main__':
    # blobLabel_dict = label_mounds('../../Capstone_images/NN_ready_images', '../../Capstone_images/labeled_nn_images')
    #
    # img_lst = blobInfo_dict.keys()
    # blobInfo_dict = crop_mound(img_lst, blobInfo_dict, '../../Capstone_images/NN_ready_images')

    #adding more training images

    # blobInfo_dict = pickle.load(open('../data/label_data.pkl', 'rb'))
    # train_nomound_mound(blobInfo_dict)
    # blobTrain_dict = {}
    # blobTrain_dict.update(blobInfo_dict)
    # blobTrain_dict.update(blobLabel_dict)
    #
    # img_lst = ['10-1.png', '10-10.png', '10-11.png', '10-12.png', '10-13.png', '10-14.png', '10-2.png', '10-3.png', '10-4.png', '10-5.png']
    #
    # blobTraindict = crop_mound(img_lst, blobTrain_dict, '../../Capstone_images/NN_ready_images')
    # labeler(blobTraindict, img_lst)

    blobTraindict = pickle.load(open('../data/label_data2.pkl', 'rb'))
    train_nomound_mound(blobTraindict)
