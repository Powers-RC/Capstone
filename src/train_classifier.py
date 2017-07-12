from identify_mounds import *
from PIL import Image
import numpy as np
import os
import subprocess
from collections import defaultdict
import pickle
from scipy.misc import imread
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics


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


def train_nomound_mound(X, y, filters, batch_sizes, epochs):
    # X_train, X_test, y_train, y_test,
    model = Sequential()


    kernel_size = (3, 3)
    input_shape = (4, 16, 16)
    pool_size =(2, 2)
    nb_filters = filters

    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), input_shape=input_shape, kernel_initializer='TruncatedNormal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer='TruncatedNormal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss = 'binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])

    model.fit(X, y, batch_size=batch_sizes, epochs=epochs, verbose=0)


    filename = '../data/mound_classifier.HDF5'
    model.save(filename)


    # y_predict = model.predict(X_test, batch_size=batch_sizes)
    # score = model.evaluate(X_test, y_test, verbose=1)
    #
    # return y_predict, y_test


    # precision_for_filters = []
    # recall_for_filters = []
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    #
    # y_predict = model.predict_classes(X_test, batch_size=batch_sizes)
    #
    # conf_mat = confusion_matrix(y_test, y_predict)
    # print(conf_mat)
    #
    # precision = conf_mat[1,1]/ (conf_mat[1,1] +conf_mat[0,1])
    # precision_for_filters.append(precision)
    # print('Precision, PPV: {}'.format(precision))
    #
    # recall = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0 ])
    # recall_for_filters.append(recall)
    # print('Recall, TPR: {}'.format(recall))
    #
    # #weighted average of percision and recall
    # f1_score = 2 * (precision * recall)/(precision + recall)
    # print('The F1_score for this model is {}'.format(f1_score))
    #
    # print('The number of mounds in test set is: {}'.format(sum(y_test)))
    # return  precision




    # plt.plot(num_filters, precision_for_filters, 'r-')
    # plt.plot(num_filters, recall_for_filters, 'b-')
    # plt.xlabel("Number of Filters")
    # plt.ylabel('Decimal Percent')
    # plt.title('How Filter Numbers Effect Precision & Recall')
    # plt.savefig('../images/PR_filters.png')

def run_gridsearch(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.3, random_state=6)

    classifier = KerasClassifier(train_nomound_mound)
    metrics = ['precision']
    for score in metrics:
        print('Tuning hyperparameters for {}'.format(score))


        batch_sizes = [1, 3, 5, 10, 20, 40, 50, 75, 100]
        epochs = [10, 25, 50, 75, 100, 150, 200, 300 ]
        filters = [3, 5, 10, 15, 20, 25, 40]

        param_grid = dict(filters=filters, batch_sizes=batch_sizes, epochs=epochs)
        validator = GridSearchCV(classifier, param_grid=param_grid, n_jobs= 1, scoring=score)
        print(validator)
        grid_results = validator.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print(grid_results.best_params_)
        print()

        means = grid_results.cv_results_['mean_test_score']
        stds = grid_results.cv_results_['std_test_score']
        params = grid_results.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, grid_results.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def plot_roc(y_predict, y_test):

    plt.close('all')
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('../images/roc_curve.png')





if __name__ == '__main__':
    seed = 6
    np.random.seed(seed)
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

    dic = pickle.load(open('../data/label_data2.pkl', 'rb'))
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
    # X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.1)




    # n_splits = 5
    # skf = StratifiedKFold(n_splits=n_splits, random_state=6)
    # sum_prec = 0
    # for train_index, test_index in skf.split(X_train, y_train):
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     print("Testing KFold")
    #     X_tr, X_val = X_train[train_index], X_train[test_index]
    #     y_tr, y_val = y_train[train_index], y_train[test_index]
    #     sum_prec += train_nomound_mound(X_tr, X_val, y_tr, y_val, filters = 5, batch_sizes = 10, epochs = 75)
    #
    # print(sum_prec/n_splits)

    # run_gridsearch(X_, y)
    #best overall 20 filters, 5 batch, 100 epochs: avg_prec = .625
    #best precision 5 filters 10 batch, 150 epochs: avg_prec = .681
    #more parameters best precision 5 filters 10 batch, 75 epochs: avg_prec =.719

    # y_predict, y_test = train_nomound_mound(X_train, X_test, y_train, y_test, filters = 5, batch_sizes = 10, epochs = 75)
    #
    # plot_roc(y_predict, y_test)

    train_nomound_mound(X_, y, filters = 5, batch_sizes = 10, epochs = 75)
