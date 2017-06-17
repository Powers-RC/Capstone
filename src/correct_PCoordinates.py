import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import os


def correct_coordinates(dic):
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
