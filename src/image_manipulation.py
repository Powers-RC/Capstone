from PIL import Image
import numpy as np
import os


def crop_image(filepath, outfile_path):
    '''
    This function is designed to crop screenshot images to produce the raw image.

    input: Takes in the filepath to a file containg all the images
    output: Saves cropped images to the outfile path argument
    '''

    image_files = os.listdir(filepath)
    for p in image_files:
        im = Image.open(filepath+'/'+p)
        print(im.info)
        c_im = im.crop((1440 - 600, 711-600, 1440, 711))
        c_im.save(outfile_path+'/'+p)


def make_small_image():
    pass


if __name__ == '__main__':
    x = crop_image('../images/Jafay', '../images/cropped_Jafay')
