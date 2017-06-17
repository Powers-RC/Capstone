import shapely.geometry as geometry
import pylab as pl
import pickle

def combine_points(dic):
    coordinate_lst = []

    for k, v in dic.items():
        for i in v:
            coordinate_lst.append((i[0], i[1]))
    return coordinate_lst

def plot_points(lst):
    x = [tup[0] for tup in lst]
    y = [tup[1] for tup in lst]

    pl.figure(figsize=(9040, 5432))
    _= pl.plot(x,y,'o',color='#f16824')
    pl.show()

if __name__ == '__main__':
    area_dict = pickle.load(open('../data/corr_coor_dict.pkl', 'rb'))
    test = combine_points(area_dict)
    plot_points(test)
