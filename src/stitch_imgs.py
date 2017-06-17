from PIL import Image
import pickle
import numpy as np
import os


def stitch_resize():
    new_image = Image.new('RGBA', (9040, 5432))

    imgs = np.array(os.listdir('../../Capstone_images/cropped_Jafay'))
    idx = np.argsort([int(im.split('.')[0]) for im in imgs])
    sorted_imgs = [str(imgs[i]) for i in idx]
    print(sorted_imgs)


    x = 0
    y = 0
    im_count = 0
    while im_count < 82:
        for im in sorted_imgs:
            print('../../Capstone_images/cropped_Jafay/' + im)
            im = Image.open('../../Capstone_images/cropped_Jafay/' + im)
            crop = im.crop((0,0,1130, 494))
            img = crop.resize((1130, 679), Image.ANTIALIAS)
            if x <= 7910:
                new_image.paste(img, (x, y))
                x += 1130
                im_count += 1
                print('Top if statement')
            elif x == 9040:
                new_image.paste(img, (x,y))
                x = 0
                y += 679
                im_count += 1
                print('Bottom if statement')


    new_image.save('../images/jafay_area.png')
#had to crop becuase shifting was off, need to readjust.
if __name__ == '__main__':
    pickle.load(open('../data/corr_coor_dict.pkl', 'rb'))
    stitch_resize()
