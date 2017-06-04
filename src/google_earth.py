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
    delay = 60

    for i, c in enumerate(coordinates):
        # a = c.split(',')
        # print(a)
        # filepath = '/Users/One-Life.Grind./Galvanize/Capstone/images/Canino_Lanocito_Richardson/' + str(i) + '.png'
        # lat, lon = c.split(';')[1:]
        url = "https://earth.google.com/web/@{},{},1573.2221546a,277.89739987d,35y,0.69169508h,0.45101597t,-0r".format(40.0851667, -105.1784944)
        driver.get(url)
        time.sleep(30)
        driver.save_screenshot(filepath)
        break

def parse_area(max_lat, min_lat, max_lon, min_lon):
    '''
    Given area coordinates parses area based on screenshot image size
    input: string max and min coordinates of latitude and longitude
    output: lst of coordinates for each are the screenshot images
    '''
    #https://gis.stackexchange.com/questions/228489/how-to-convert-image-pixel-to-latitude-and-longitude

    #https://gis.stackexchange.com/questions/48949/epsg-3857-or-4326-for-googlemaps-openstreetmap-and-leaflet




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
    filename = '../data/OSMPPrairieDogColonies.kmz'

    kmz = ZipFile(filename, 'r')
    kml = kmz.open('doc.kml', 'r')

    parser = xml.sax.make_parser()
    handler = PlacemarkHandler()
    parser.setContentHandler(handler)
    parser.parse(kml)
    kmz.close

    mapping = handler.mapping
    coordinates = extract_coordinates(mapping)
    pd_id = extract_pdid(mapping)

    table = join_id_coordinates(pd_id, coordinates)

    max_lat, min_lat, max_lon, min_lon = find_boundries('Jafay', coordinates)


    take_area_photos(coordinates)
