from kmz_extraction import PlacemarkHandler, extract_pdid, join_id_coordinates, extract_coordinates
from zipfile import ZipFile
import xml.sax, xml.sax.handler
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from PIL import Image
import time
import csv
import os
import pdb

def create_folders(coordinates):
    for s in coordinates:
        s = s.split(",")
        name = s[0]
        filepath = "../images/" + name
        if not os.path.exists(filepath):
            os.makedirs(filepath)

def take_area_photos(coordinates):
    driver = webdriver.Chrome()
    driver.set_window_size(1440, 900)
    driver.maximize_window()
    driver.switch_to_window(driver.window_handles[0])
    delay = 60

    # area_lst = [area.split(',')for area in coordinates]
    #
    # for area in area_lst:
    #     if area[0] == area_name:
    #         coordinates = area[1:]
    for i, c in enumerate(coordinates):
        filepath = '/Users/One-Life.Grind./Galvanize/Capstone/images/Canino_Lanocito_Richardson/' + str(i) + '.png'
        lat, lon = c.split(';')
        url = "https://earth.google.com/web/@{},{},1573.2221546a,277.89739987d,35y,0.69169508h,0.45101597t,-0r".format(40.0851667, -105.1784944)
        driver.get(url)
        # print(WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, 'icon'))))
        # driver.implicitly_wait(60)
        time.sleep(30)
        driver.save_screenshot(filepath)
        break

def parse_area(max_lat, min_lat, max_lon, min_lon):
    '''
    Given area coordinates parses area based on screenshot image size
    input: string max and min coordinates of latitude and longitude
    output: lst of coordinates for each are the screenshot images
    '''
    pass


def crop_image(filepath, outfile_path):
    '''
    This function is designed to crop screenshot images to produce the raw image.

    input: Takes in the filepath to a file containg all the images
    output: Saves cropped images to the outfile path argument
    '''

    image_files = os.listdir(filepath)
    for p in image_files:
        im = Image.open(filepath+'/'+p)
        c_im = im.crop((55, 0, 1131, 675))
        c_im.save(outfile_path+'/'+p)

def make_small_image():
    pass 




def find_boundries(area, coordinates):
    '''
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

    #croping images in Jafay directory
    x = crop_image('../images/Jafay', '../images/cropped_Jafay')


    # take_area_photos(coordinates, "Brewbaker")
