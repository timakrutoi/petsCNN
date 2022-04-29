import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# for dataset class
import torchvision.models
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader

# for checkpoints
import time

# for whatever
import pandas as pd
import os
from tqdm import tqdm
import numpy as np


class petsCNN(nn.Module):
    def __init__(self):
        super(petsCNN, self).__init__()

        # 250 240 120 110 55
        # 150 140 70 60 30

        # 150 146 73 70 35
        self.mod = torchvision.models.resnet34()

        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(1000),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.25),
        #     nn.Linear(1000, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.25),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 37)
        # )

        self.lc = nn.Linear(1000, 37)

    def forward(self, x):

        x = self.mod(x)
        # x = self.classifier(x)
        x = self.lc(x)

        return x


class OxfordPets(Dataset):
    def __init__(self, annotations_file, img_dir, batch_size=100, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # p = transforms.Compose([transforms.Resize((150, 150))])

        image = transforms.Resize((150, 150))(read_image(img_path))

        # image = image.reshape((1, 3, 150, 150))
        label = self.img_labels.iloc[idx, 1]
        # print(image.shape, label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float(), label


def get_dataLoader(data_path):

    train_data = OxfordPets(
        annotations_file=data_path + 'annotations_train.csv',
        img_dir=data_path + 'images'
    )

    test_data = OxfordPets(
        annotations_file=data_path + 'annotations_test.csv',
        img_dir=data_path + 'images'
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
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

    data_path = '/home/tima/Desktop/oxfordPets/'
    total_epoch = 20
    start_epoch = 10
    make_checkpoints = True
    checkpoints_path = 'checkpoints/'

    lr = 0.01  # learning rate
    momentum = 0.9

    model = petsCNN()
    checkpoint = torch.load('checkpoints/model_t2_e9_acc0.210')
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(petsCNN.parameters(model), lr, momentum)

    train_loader, test_loader = get_dataLoader(data_path)

    print(f'==>>> total training batch number: {len(train_loader)}')
    print(f'==>>> total testing  batch number: {len(test_loader)}')

    for epoch in range(start_epoch, total_epoch):
        train_iterator = tqdm(train_loader, ncols=100, desc='Epoch: {}, training'.format(epoch))

        for batch_idx, (x, target) in enumerate(train_iterator):
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, target)  # torch.from_numpy(np.array(target)).long())

            loss.backward()
            optimizer.step()

        train_iterator.close()

        # ==================================================================
        # Testing
        total_cnt = 0
        correct_cnt = 0
        test_loss = 0
        batch_idx = 0
        acc = 0
        test_iterator = tqdm(test_loader, ncols=128, desc='Epoch: {}, testing '.format(epoch))

        for batch_idx, (x, target) in enumerate(test_iterator):  # reading test data
            y = model(x)
            loss = criterion(y, target)

            test_loss += loss.item()
            _, predict = y.max(1)
            total_cnt += target.size(0)
            correct_cnt += predict.eq(target).sum().item()
            acc = (correct_cnt * 1.) / total_cnt
            test_iterator.set_postfix(str='acc: {:.3f}, loss: {:.3f}'.format(acc, test_loss))
            test_iterator.update()

        test_iterator.close()

        if make_checkpoints:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, checkpoints_path + 'model_t2_e{}_acc{:.3f}'.format(epoch, acc))

