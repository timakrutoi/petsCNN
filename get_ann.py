import glob
import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def save2f(fname, l, path='/home/tima/Desktop/oxfordPets/'):
    labels = {
        'Maine_Coon': '0',
        'leonberger': '1',
        'pug': '2',
        'Bombay': '3',
        'beagle': '4',
        'keeshond': '5',
        'havanese': '6',
        'newfoundland': '7',
        'scottish_terrier': '8',
        'Abyssinian': '9',
        'american_bulldog': '10',
        'Siamese': '11',
        'saint_bernard': '12',
        'german_shorthaired': '13',
        'shiba_inu': '14',
        'samoyed': '15',
        'Sphynx': '16',
        'staffordshire_bull_terrier': '17',
        'chihuahua': '18',
        'great_pyrenees': '19',
        'Bengal': '20',
        'Russian_Blue': '21',
        'basset_hound': '22',
        'english_setter': '23',
        'Persian': '24',
        'american_pit_bull_terrier': '25',
        'yorkshire_terrier': '26',
        'japanese_chin': '27',
        'Birman': '28',
        'Egyptian_Mau': '29',
        'British_Shorthair': '30',
        'boxer': '31',
        'wheaten_terrier': '32',
        'pomeranian': '33',
        'Ragdoll': '34',
        'english_cocker_spaniel': '35',
        'miniature_pinscher': '36'
    }

    li = []

    with open(path+fname, 'w') as file:
        for i in l:
            if not i.endswith('jpg'):
                print(i)
                continue
            try:
                img = Image.open(i)  # open the image file
                img.verify()  # verify that it is, in fact an image
                im = plt.imread(i)
                if im.shape[-1] > 3:
                    print(f'imposter {im.shape} {i}')
                    continue
            except (IOError, SyntaxError) as e:
                print('Bad file:', i)  # print out the names of corrupt files
                continue

            t = i[len('/home/tima/Desktop/oxfordPets/images/'):]
            lab_tmp = t[:-4].split(sep='_')
            lab = ''
            # f = plt.imread(i)
            if len(lab_tmp) > 2:
                lab += lab_tmp[0]
                for j in range(1, len(lab_tmp) - 1):
                    lab += '_' + lab_tmp[j]
            else:
                lab = lab_tmp[0]
            file.write(i[len('/home/tima/Desktop/oxfordPets/images/'):] + ', ' + labels[lab] + '\n')
            # print(i[len('/home/tima/Desktop/oxfordPets/images/'):] + ', ' + lab)

    for i, j in enumerate(li):
        print(f'\'{j}\': \'{i}\',')


if __name__ == '__main__':
    l = glob.glob(glob.escape('/home/tima/Desktop/oxfordPets/images/') + '*')

    random.shuffle(l)

    g = int(0.7 * len(l))

    train = l[:g]
    test = l[g:]

    # save2f('test.csv', l)
    save2f('annotations_train.csv', train)
    save2f('annotations_test.csv', test)

    print(len(train))
    print(len(test))
    print(len(l))
