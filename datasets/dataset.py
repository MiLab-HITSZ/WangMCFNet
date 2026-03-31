
from datasets.augmentations import augmentations
from albumentations import CenterCrop,Compose,Resize,RandomCrop,Normalize
from albumentations.pytorch import ToTensorV2
import albumentations as A
import random
from datasets.data8000 import *
from torch.utils.data import Dataset

 
import os
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms 
class DeepfakeDataset(Dataset):

    def __init__(self,phase='train',datalabel='ff-5', resize=(384,384),imgs_per_video=30,min_imgs=30,normalize=dict(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),frame_interval=10,max_imgs=300,augment='augment0'):
        assert phase in ['train', 'val', 'test']
        self.datalabel=datalabel
        self.phase = phase
        self.imgs_per_video=imgs_per_video
        self.frame_interval=frame_interval
        self.num_classes = 2
        self.epoch=0
        self.max_imgs=max_imgs
        if min_imgs:
            self.min_imgs=min_imgs
        else:
            self.min_imgs=max_imgs*0.3
        self.dataset=[]
        self.aug=augmentations[augment]
        resize_=(int(resize[0]/0.8),int(resize[1]/0.8))
        self.resize=resize
        self.trans = Compose([Resize(*resize_),CenterCrop(*resize),#ToTensorV2(normalize=normalize)])
                              Normalize(mean=normalize['mean'], std=normalize['std']),ToTensorV2()  # 注意这里使用 ToTensorV2()
        ])
        if type(datalabel)!=str:
            self.dataset=datalabel
            return

        elif 'ff-all' in self.datalabel:
            for i in ['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']:
                self.dataset+=FF_dataset(i,self.datalabel.split('-')[2],phase)
        elif 'celebv2' in self.datalabel:
            self.dataset = celeb_dataset(phase)
        elif 'dfdc' in self.datalabel:
            self.dataset = dfdc_dataset(phase)
        elif 'dfw' in self.datalabel:
            self.dataset=dfw_dataset(phase)
        elif 'celebv1' in self.datalabel:
            json_path = '/mnt/data/wxp/cropped_celebv1/celebv1_cropped.json'
            cropped_dir = '/mnt/data/wxp/cropped_celebv1'
            self.dataset = celebv1_dataset(json_path )
    def next_epoch(self):
        self.epoch+=1

    def make_balance(data):
        tr = list(filter(lambda x: x[1] == 0, data))
        tf = list(filter(lambda x: x[1] == 1, data))
        if len(tr) > len(tf):
            tr, tf = tf, tr
        rate = len(tf) // len(tr)
        res = len(tf) - rate * len(tr)
        tr = tr * rate + random.sample(tr, res)
        return tr + tf
    def __getitem__(self, item):
        if 'ff' in self.datalabel:

            vid=self.dataset[item//self.imgs_per_video]
            vd=sorted(os.listdir(vid[0]))
            if len(vd)<self.min_imgs:
                return self.__getitem__((item+self.imgs_per_video)%(self.__len__()))
            ind=(item%self.imgs_per_video*self.frame_interval+self.epoch)%min(len(vd),self.max_imgs)
            ind=vd[ind]
            image =cv2.imread(os.path.join(vid[0],ind))# 130 130 3
            image_path = os.path.join(vid[0], ind)
            if os.path.isdir(image_path):
                print(f'跳过{image_path}')
                return self.__getitem__((item+self.imgs_per_video+1)%(self.__len__()))

            if not image_path.lower().endswith('.png'):
                item += 1  # Move to next file
                print(f'跳过{image_path}')
                return self.__getitem__((item+self.imgs_per_video+1)%(self.__len__()))
            if image is None:
                print(f'跳过{image_path}')
                print("cv2.imread 返回 None，图像未读入")
                return self.__getitem__((item + self.imgs_per_video + 1) % (self.__len__()))
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image from {image_path}. Skipping...")
                return self.__getitem__((item + self.imgs_per_video + 1) % (self.__len__()))

            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.phase == 'train':
                image = self.aug(image=image)['image']
            image=self.trans(image=image)['image']

            return image, vid[1], image_path
        else:
            image_path, label = self.dataset[item]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.trans(image=image)['image']
            return image, label, image_path
    def __len__(self):
        if 'ff' in self.datalabel:
            return len(self.dataset)*self.imgs_per_video#3240*3
        else:
            return len(self.dataset)