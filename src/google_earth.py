from kmz_extraction import PlacemarkHandler, extract_pdid, join_id_coordinates, extract_coordinates
from zipfile import ZipFile
import xml.sax, xml.sax.handler
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import subprocess
import re
import time
import csv
import pdb
import math

def create_folders(coordinates):
    for s in coordinates:
        s = s.split(",")
        name = s[0]
        filepath = "../images/" + name
        if not os.path.exists(filepath):
            os.makedirs(filepath)

def get_screen_res():
    '''
    Gets the screen resolution of a mac
    output: returns the pixel width and height of your monitor
    '''

    results = str(subprocess.Popen(['system_profiler SPDisplaysDataType'],stdout=subprocess.PIPE, shell=True).communicate()[0])
    res = re.search('Resolution: \d* x \d*', results).group(0).split(' ')
    width, height = res[1], res[3]
    return width, height

def take_area_photos(coordinates):
    '''
    Takes a screen shot using GE from some starting quardinates, shifting the size of the image each iteration.
    input: an iterable of image coodinate locations
    output: a png image file
    '''
    width, height = get_screen_res()
    get_screen_res()
    driver = webdriver.Chrome()
    driver.set_window_size(width, height)
    driver.maximize_window()
    driver.switch_to_window(driver.window_handles[0])


    for i, c in enumerate(coordinates):
        filepath = '/Users/One-Life.Grind./Galvanize/Capstone/images/Jafay/' + str(i) + '.png'
        lat, lon = c.split(',')
        url = 'http://earth.google.com/web/@{},{},1615.83371101a,196.41606213d,35y,0h,0t,0r'.format(lat, lon)
        driver.get(url)
        time.sleep(30)
        driver.save_screenshot(filepath)

    driver.close()


# https://earth.google.com/web/@40.07879835,-105.16907049,1615.82020731a,195.38944324d,35y,0h,0t,0r
#
# https://earth.google.com/web/@40.0788,-105.169072,1615.83371101a,196.41606213d,35y,0h,0t,0r
def parse_area(max_lat, min_lat, max_lon, min_lon):
    '''
    Given area coordinates parses area based on screenshot image size
    input: string max and min coordinates of latitude and longitude
    output: lst of coordinates for each are the screenshot images
    '''
    #https://gis.stackexchange.com/questions/228489/how-to-convert-image-pixel-to-latitude-and-longitude

    #https://gis.stackexchange.com/questions/48949/epsg-3857-or-4326-for-googlemaps-openstreetmap-and-leaflet

    #https://spie.org/membership/spie-professional-magazine/spie-professional-archives-and-special-content/2016_october_archive/optics-of-google-earth

    #lhttps://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters


    # max_lat, min_lat, max_lon, min_lon = float(max_lat), float(min_lat), float(max_lon), float(min_lon)
    coordinates = ['{},{}'.format(max_lat, max_lon)]
    new_lat, new_lon = max_lat, max_lon
    map_unit = .19166667 #scale: 23m/i , PPI: 120p/i
    h_offset = 720 * map_unit #corrected val = 1130
    v_offset = 237 * map_unit #corrected val = 679
    R = 6378137

    while new_lat >= min_lat:
        true_max_lon = max_lon
        if new_lon <= min_lon:
            d_lon = h_offset/(R * math.cos(new_lat * math.pi/180))
            new_lon = new_lon +  (d_lon * (180/math.pi))
            coordinates.append('{},{}'.format(new_lat, new_lon))
        else:
            d_lat = v_offset/R
            new_lat = new_lat - (d_lat * 180/math.pi)
            new_lon = true_max_lon
            coordinates.append('{},{}'.format(new_lat, new_lon))


    return coordinates



def find_boundries(area, coordinates):
    '''
    Find the min and max boundries from field coordinates.
    input: coorinates and an area name(str)
    output: returns the high and low lat lons for that area
    '''

    d = {area: i.split() for i in coordinates if i.split(',')[0] == 'Jafay'}
    s = ''.join(d[area]).split(',')[1:]
    lat_lst = []
    lon_lst = []

    for c in s:
        lat, lon = c.split(';')
        lat_lst.append(lat)
        lon_lst.append(lon)

    max_lat = max(lat_lst)
    min_lat = min(lat_lst)
    max_lon =  max(lon_lst)
    min_lon = min(lon_lst)
    return max_lat, min_lat, max_lon, min_lon




if __name__ == '__main__':


    # max_lat, min_lat, max_lon, min_lon = find_boundries('Jafay', coordinates)
    max_lat, max_lon, min_lat, min_lon = 40.078800, -105.169072, 40.072700, -105.150064
    c = parse_area(max_lat, min_lat, max_lon, min_lon)


    take_area_photos(c)
