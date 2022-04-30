import argparse

from petsCNN import petsCNN
import torch
from torchvision.io import read_image
from torchvision import transforms
import numpy as np


def predict(pic, model):
    image = transforms.Resize((300, 300))(read_image(pic))

    image = image.reshape((1, 3, 300, 300))

    model.eval()
    y = model(image.float())

    return y


if __name__ == '__main__':

    check_name = 'checkpoints/t5/model_e0_acc0.879'
    pic_name = '/home/tima/Desktop/oxfordPets/images/Maine_Coon_1.jpg'

    labels = {
        '0': 'Maine_Coon',
        '1': 'leonberger',
        '2': 'pug',
        '3': 'Bombay',
        '4': 'beagle',
        '5': 'keeshond',
        '6': 'havanese',
        '7': 'newfoundland',
        '8': 'scottish_terrier',
        '9': 'Abyssinian',
        '10': 'american_bulldog',
        '11': 'Siamese',
        '12': 'saint_bernard',
        '13': 'german_shorthaired',
        '14': 'shiba_inu',
        '15': 'samoyed',
        '16': 'Sphynx',
        '17': 'staffordshire_bull_terrier',
        '18': 'chihuahua',
        '19': 'great_pyrenees',
        '20': 'Bengal',
        '21': 'Russian_Blue',
        '22': 'basset_hound',
        '23': 'english_setter',
        '24': 'Persian',
        '25': 'american_pit_bull_terrier',
        '26': 'yorkshire_terrier',
        '27': 'japanese_chin',
        '28': 'Birman',
        '29': 'Egyptian_Mau',
        '30': 'British_Shorthair',
        '31': 'boxer',
        '32': 'wheaten_terrier',
        '33': 'pomeranian',
        '34': 'Ragdoll',
        '35': 'english_cocker_spaniel',
        '36': 'miniature_pinscher'
    }

    model = petsCNN()
    checkpoint = torch.load(check_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    res = predict(pic_name, model).detach().numpy()
    # print(res)
    l = np.argsort(res)
    l = np.flip(l)

    print('petsCNN thinks its {} with {} confidence'.format(labels[f'{l[0, 0]}'], res[0, l[0, 0]]))
    print('other likely predictions: ')
    for i in range(1, 5):
        print('\t({}) its {} with {} confidence'.format(i, labels[f'{l[0, i]}'], res[0, l[0, i]]))
