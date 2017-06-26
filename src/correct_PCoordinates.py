import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import os


def correct_coordinates(dic):
    '''
    Given an input of coordinates from the small images scale up coordinates to screenshot coordinates
    input: A dictionary contatining the photos each with their associated coordinates
    output:Returns a dictionary with the reduced number of photos and corrected coordinates
    '''
    new_dic = {}

    for k, v in dic.items():
        new_dic[k] = [[i[0], i[1]] for i in v]

    big_first = 0
    photo_num = 0
    sub_num = 0
    h_shift = 0
    v_shift = 0

    alt_coordinates = defaultdict(list)
    while photo_num < 82:
        big_image = '{}.png'.format(big_first)
        if sub_num < 14:
            im = '{}-{}.png'.format(photo_num, sub_num)
            if h_shift < 4:
                results = new_dic.get(im)
                if results:
                    for v in results:
                        v[0] = v[0] + (226 * h_shift)
                        v[1] = v[1] + (226 * v_shift)
                        alt_coordinates[big_image].append([v[0], v[1]])
                sub_num += 1
                h_shift += 1
            elif h_shift == 4:
                results = new_dic.get(im)
                if results:
                    for v in results:
                        v[0] = v[0] + (226 * h_shift)
                        v[1] = v[1] + (226 * v_shift)
                        alt_coordinates[big_image].append([v[0], v[1]])
                sub_num += 1
                h_shift = 0
                v_shift += 1
        elif sub_num == 14:
            im = '{}-{}.png'.format(photo_num, sub_num)
            results = new_dic.get(im)
            if results:
                for v in results:
                    v[0] = v[0] + (226 * h_shift)
                    v[1] = v[1] + (226 * v_shift)
                    alt_coordinates[big_image].append([v[0], v[1]])
            big_first += 1
            photo_num += 1
            sub_num = 0
            h_shift = 0
            v_shift = 0

    final_dict = second_corrected_coordinates(alt_coordinates)

    pickle.dump(final_dict, open('../data/corr_coor_dict.pkl', 'wb'))

def second_corrected_coordinates(dic):
    ''' Similar function as above but applies the coordinate transformations to the entire area of interest.
    input: Dictionary from first corrected coordinates
    Output: A dictionary containing praririe dog coorinates for the whole area
    '''
    final_coordinates = defaultdict(list)

    photo_num = 0
    h_shift = 0
    v_shift = 0

    while photo_num < 82:
        im = '{}.png'.format(photo_num)
        if h_shift < 8:
            results = dic.get(im)
            if results:
                for v in results:
                    v[0] = v[0] + (1130 * h_shift)
                    v[1] = v[1] + (679 * v_shift)
                    final_coordinates[im].append([v[0], v[1]])
            h_shift += 1
            photo_num += 1
        elif h_shift == 8:
            results = dic.get(im)
            if results:
                for v in results:
                    v[0] = v[0] + (1130 * h_shift)
                    v[1] = v[1] + (679 * v_shift)
                    final_coordinates[im].append([v[0], v[1]])
            h_shift = 0
            v_shift += 1
            photo_num += 1

    return final_coordinates





if __name__ == '__main__':
    mound_dict = pickle.load(open('../data/mound_dict.pkl', 'rb'))
    correct_coordinates(mound_dict)
