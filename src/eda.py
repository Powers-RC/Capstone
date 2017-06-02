import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from pandas.tools.plotting import scatter_matrix
from scipy.stats.kde import gaussian_kde
from kmz_extraction import PlacemarkHandler, make_table
from zipfile import ZipFile
import xml.sax, xml.sax.handler
import csv


def plot_eda_graphs(df):
    columns = df.columns
    fig = plt.figure()

    plot_num = 1
    for col in columns:
        print(col)
        xvals = df[col].unique
        ax = plt.subplot(3,5,plot_num)
        ax.plot(xvalsdf[col])
        ax.set_title(col)

        plot_num += 1
    plt.tight_layout()

def make_scatter_matrix(df):
    df = df[['PD_ID','Year', 'Name', 'Acres', 'Hectares',
       'Plague', 'Manager', 'Perimeter', 'Activity', 'YearAcquir',
       'sdeWildlif', 'Shape_area', 'Shape_len']]
    scatter = pd.scatter_matrix(df)
    plt.savefig('/Users/One-Life.Grind./Galvanize/Capstone/images/num_df_scatter.png')

def plot_hist_kde(df):
    df = df[['Year', 'Name']]
    X = np.sort(df.Year.unique()).astype(int, copy=False)
    df = df.groupby('Year').count()
    y = df.Name

    X_vals = np.linspace(0, max(y))
    kde_pdf = gaussian_kde(y)
    y2 = kde_pdf(X_vals)


    fig = plt.figure()
    minorLocator = MultipleLocator(5)
    ax = fig.add_subplot(111)
    ax.plot(X_vals, y2, color='r', linestyle='-')
    ax.hist(y, bins=30, ec='b', normed=True)
    ax.set_title('Colony Size Frequency')
    ax.xaxis.set_minor_locator(minorLocator)
    ax.set_xlabel('Number of Colonies')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../images/colony_hist.png')
    return y


def plot_year_area_count(df):
    df = df[['Year', 'Name']]
    X = np.sort(df.Year.unique()).astype(int, copy=False)
    df = df.groupby('Year').count()
    y = df.Name

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X,y)
    ax.set_title('Colony Numbers Per Year')
    ax.xaxis.set_major_locator(ticker.MultipleLocator())
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Colonies')
    plt.xticks(rotation=45)
    plt.savefig('../images/colony_growth.png')
    return df




if __name__ == '__main__':
    plt.close('all')
    df = pd.read_csv('../data/OSMPPrairieDogColonies.csv')
    num_df = df[['PD_ID', 'Year', 'Acres', 'Hectares', 'Perimeter', 'YearAcquir','sdeWildlif', 'Shape_area', 'Shape_len']]

    # make_scatter_matrix(num_df)
    x = plot_year_area_count(df)
    x2 = plot_hist_kde(df)

    filename = '../data/OSMPPrairieDogColonies.kmz'

    kmz = ZipFile(filename, 'r')
    kml = kmz.open('doc.kml', 'r')

    parser = xml.sax.make_parser()
    handler = PlacemarkHandler()
    parser.setContentHandler(handler)
    parser.parse(kml)
    kmz.close

    mapping = handler.mapping
    areas, coordinates = make_table(mapping)
