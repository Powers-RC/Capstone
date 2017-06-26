from zipfile import ZipFile
import xml.sax, xml.sax.handler
import csv
import re
import pdb
'''
Coverts kmz to kml file

'''

class PlacemarkHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.inName = False #handle XML parser events
        self.inPlacemark = False
        self.mapping = {}
        self.buffer = ""
        self.name_tag = ""

    def startElement(self, name, attributes):
        if name == "Placemark": #on start Placemark tag
            self.inPlacemark = True
            self.buffer = ""
        if self.inPlacemark:
            if name == "name": # on start title tag
                self.inName = True #save name text to follow
    def characters(self, data):
        if self.inPlacemark: # on text within tag
            self.buffer += data #save text if in title

    def endElement(self, name):
        self.buffer = self.buffer.strip('\n\t')

        if name == "Placemark":
            self.inPlacemark = False
            self.name_tag = "" #clear current name

        elif name == "name" and self.inPlacemark:
            self.inName = False #on end title tag
            self.name_tag = self.buffer.strip()
            self.mapping[self.name_tag] = {}

        elif self.inPlacemark:
            if name in self.mapping[self.name_tag]:
                self.mapping[self.name_tag][name] += self.buffer
            else:
                self.mapping[self.name_tag][name] = self.buffer
        self.buffer = ""

def extract_coordinates(mapping):
    '''
    Extract the coordinates from the kml file
    input: A file contatining the geospatical data for each area
    output: csv format with with the areas and their associated corrdinates
    '''

    output = []
    csv_format = []
    area_names_lst = []

    #extracts the coordinates for the colony
    for d in mapping:
        area_names_lst.append(d)
        area = d
        separator = ','
        c = mapping[area]["coordinates"]
        c = c.replace(" ", "")
        s = c.split(',')
        coordinates = s
        coordinates = [s.strip("0") for s in coordinates]
        lst_evens = range(0,len(coordinates)-1,2)
        area = area.replace(",", "_")
        comb_coor = [area]

        for i in lst_evens:
            col = coordinates[i+1] + ";" + coordinates[i]
            comb_coor.append(col)
        csv_format.append(','.join(comb_coor))

    return csv_format



def extract_pdid(mapping):
    '''
    Extracted the prairie dog id from data source
    input:file containing prairie dog colony area id's
    output: a list of the prairie dog colony id's
    '''

    pd_id_lst = []
    for d in mapping:
        dis = mapping[d]["description"]
        dis = dis.replace("\n", "")
        try:
            pd_id = re.search("PD_ID</td><td>\d+</td>", dis)
            pd_id = pd_id.group(0)
            pd_id = re.sub('[^0-9]', '', pd_id)
            pd_id_lst.append(pd_id)
        except:
            pd_id = ""
            return pd_id

    return pd_id_lst

def join_id_coordinates(id_lst, c_lst):
    '''
    Joined the prairie dog id's and their mapped coordinates
    input: list of id's and coordinates
    out: list of colony id and coordinates
    '''

    table = []
    for i in range(len(id_lst)):
        row = id_lst[i] + ',' + c_lst[i]
        table.append(row)

    return table




def write_csv(csv_string):
    '''
    Creates csv file with the id's and mapped corrdinates
    input: the list of colony id's and their mapped coordinates
    output: csv with data
    '''
    
    with open('location_data.csv', "w") as f:
        # f.write("area_name, coordinates, \n".format())
        for s in csv_string:
            split = s.split(',')
            for s in split:
                f.write(str(s))
                f.write(',')
            f.write('\n')




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
    c = write_csv(table)


    #reference for class above http://programmingadvent.blogspot.com/2013/06/kmzkml-file-parsing-with-python.html
