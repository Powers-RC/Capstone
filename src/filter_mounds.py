from identify_mounds import label_blobs
from train_classifier import labeler, crop_mound, train_nomound_mound
from keras.models import load_model
from scipy.misc import imread
import numpy as np
import pickle
import itertools
import copy
import pdb

def predict_labels(dic, model):
    ''' Predicts if the images passed into the model are a prairie dog mound or unknown object.
    input: A dictionary contatining the filepath to the images you are trying to label
    output: creates a pickle object contatining the dictionary with an added key for there labels and the values for each image
    '''

    for im in dic:
        X = []
        X_ = []
        X.extend(dic[im]['img_lst'])
        dic[im]['labels'] = []
        dic[im]['coordinates'] = dic[im]['coordinates'].tolist()
        for mi in X:
            arr = imread(mi)
            X_.append(arr)

        len_X = len(X_)
        X_ = (np.array(X_).reshape(len_X, 4, 16, 16)/255).astype('float32')
        y_predict = model.predict_classes(X_)
        y_predict = y_predict.tolist()
        dic[im]['labels'].append(list(itertools.chain.from_iterable(y_predict)))

    pickle.dump(dic, open('../data/labeled_dict.pkl', 'wb'))

def remove_blobObj(dic):
    ''' Removes any blob that was labeled as an unknown object
    input: The dictionary produced by predict_labels function
    output: A dictionary contatining only prairie dog mounds
    '''
    
    d = copy.deepcopy(dic)
    new_dic ={}
    for im in d:
        new_dic[im] = []
        for i, v in enumerate(d[im]['labels'][0]):
            if v == 1:
                new_dic[im].append(d[im]['coordinates'][i])


    pickle.dump(new_dic, open('../data/mound_dict.pkl', 'wb'))




if __name__ == '__main__':
    # coordinateBlob_dict = label_blobs('../../Capstone_images/NN_ready_images')
    #
    # crop_mound(coordinateBlob_dict, '../../Capstone_images/NN_ready_images','../../Capstone_images/mound_imgs/')

    # dict_data = pickle.load(open('../data/dict_data.pkl', 'rb'))
    # model =  load_model('../data/mound_classifier.HDF5')
    # #
    # y = predict_labels(dict_data, model)

    # labeled_dict = pickle.load(open('../data/labeled_dict.pkl', 'rb'))
    # #
    # d = remove_blobObj(labeled_dict)
