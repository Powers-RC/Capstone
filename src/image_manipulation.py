from PIL import Image
from PIL import ImageFile
import PIL.ImageOps
from sklearn.model_selection import train_test_split
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
        c_im = im.crop((56, 0, 1186, 679))
        c_im.save(outfile_path+'/'+p)




def make_small_image(filepath, outfile_path):
    '''
    Takes in crop ready images and produces multiple smaller images used in training.

    input: Filepath to large images, outputfile for cropped images
    output: File of images & list of image names
    '''

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cropped_imgs = os.listdir(filepath)
    images = []
    for p in cropped_imgs:
        im = Image.open(filepath+'/'+p)
        x, y = im.size
        start_hb, start_vb, end_hb, end_vb = 0, 0, 226, 226

        c = 0
        while end_vb <= y:
            new_p = p.split('.')
            new_p = new_p[0] + '-' + '{}'.format(c) + '.' + new_p[1]

            if end_hb < x:
                c_im = im.crop((start_hb, start_vb, end_hb, end_vb))
                c_im.save(outfile_path+'/'+new_p)
                images.append(new_p)
                start_hb, end_hb= start_hb + 226, end_hb + 226
                c += 1

            elif end_hb == x:
                c += 1
                c_im = im.crop((start_hb, start_vb, end_hb, end_vb))
                c_im.save(outfile_path+'/'+new_p)
                images.append(new_p)
                start_hb, start_vb, end_hb, end_vb = 0, start_vb + 226, 226, end_vb + 226
        c = 0

    return images

def produce_images(images, outfile_path):
    '''
    Takes in an iterable of imgaes and rotates or mirrors the images to produce additional trainable images. Only for training images!

    Input: Iterable of image file names and filepath to add them to
    Output: Additonal images to train models on
    '''
    X_train, X_test = train_test_split(images, test_size=.20, random_state=6)

    X_train, validation = train_test_split(X_train, test_size = .20, random_state=6)
    X_train_ = X_train

    for p in X_train:
        new_p = p.split('.')
        im = Image.open(outfile_path + '/'+p)
        m = PIL.ImageOps.mirror(im)
        new_m = new_p[0] + '-' + 'm' + '.' + new_p[1]
        m.save(outfile_path+'/'+new_m)
        m1 = m.rotate(90)
        new_m1 = new_p[0] + '-' + 'm1' + '.' + new_p[1]
        m1.save(outfile_path+'/'+new_m1)
        m2 = m.rotate(180)
        new_m2 = new_p[0] + '-' + 'm2' + '.' + new_p[1]
        m2.save(outfile_path+'/'+new_m2)
        m3 = m.rotate(270)
        new_m3 = new_p[0] + '-' + 'm3' + '.' + new_p[1]
        m3.save(outfile_path+'/'+new_m3)

        r1 = im.rotate(90)
        new_r1 = new_p[0] + '-' + 'r1' + '.' + new_p[1]
        r1.save(outfile_path+'/'+new_r1)
        r2 = im.rotate(180)
        new_r2 = new_p[0] + '-' + 'r2' + '.' + new_p[1]
        r2.save(outfile_path+'/'+new_r2)
        r3 = im.rotate(270)
        new_r3 = new_p[0] + '-' + 'r3' + '.' + new_p[1]
        r3.save(outfile_path+'/'+new_r3)

        X_train_.extend([m, m1, m2, m3, r1, r2, r3])

    return X_train_, X_test, validation





if __name__ == '__main__':
    # x = crop_image('../images/Jafay', '../images/cropped_Jafay')
    images = \
    make_small_image('../images/cropped_Jafay', '../images/NN_ready_images')

    X_train, X_test, validation = produce_images(images, '../images/NN_ready_images')
