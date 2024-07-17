import os
import pickle
import string
import zipfile

import PIL
import numpy as np
import random
import torch
from torchvision import transforms
import pickle

from torch.utils.data import Dataset

from . import task

def load_pickle(file_path):
    f = open(file_path, 'rb')
    file_pickle = pickle.load(f)
    f.close()
    return file_pickle

def list_reader_all(file_list):

    img_list = []
    with open(file_list, 'rb') as file:
        for line in file.readlines():
            img_path = line
            img_list.append(img_path)
    return img_list

class CelebaDataset(Dataset):
    """Defines the base dataset class.

    This class supports loading data from a full-of-image folder, a lmdb
    database, or an image list. Images will be pre-processed based on the given
    `transform` function before fed into the data loader.

    NOTE: The loaded data will be returned as a directory, where there must be
    a key `image`.
    """

    def __init__(self,
                 root_dir,
                 pickle_path,
                 occ_dir,
                 occ_list,
                 resolution,
                 lenth=-1,
                 picklimit=True,
                 ratio = 2,
                 **_unused_kwargs):

        self.root_dir = root_dir
        self.resolution = resolution
        self.image_list_path = pickle_path
        self.occ_dir = occ_dir
        self.occ_list = task.occlist_reader(occ_list)
        self.ratio = ratio
        self.image_paths = []
        if picklimit:
            self.face_list = load_pickle(pickle_path)
            self.face_list = self.removeemptypath(root_dir, self.face_list)
        else:
            self.face_list = [
                os.path.join(path, filename)
                for path, dirs, files in os.walk(root_dir)
                for filename in files
                if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
            ]
        if lenth>0:
            self.face_list = self.face_list[0:lenth]
        self.length = len(self.face_list)

        self.transform = transforms.Compose(
            [transforms.Resize(self.resolution),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
             ]
        )

    def __len__(self):
        return self.length
    
    def removeemptypath(self,root,pathlist):
        newlist = []
        for path in pathlist:
            fullpath = os.path.join(root,path)
            if os.path.exists(fullpath):
                newlist.append(fullpath)
        return newlist
        

    def __getitem__(self, idx):
        l = self.length
        s_idx = idx % l
        if idx % self.ratio==0:
            f_idx = s_idx
        else:
            f_idx = random.randrange(l)

        gt_img = task.PIL_Reader(self.face_list[f_idx])
        id_img = task.PIL_Reader(self.face_list[s_idx])

        if idx % 2 == 0:
            gt_img, occimg,mask = task.generate_iregular_occ_mask(gt_img, self.occ_list, self.occ_dir,self.transform, [0,1, 2])
        else:
            occ_name = self.occ_list[np.random.randint(0, len(self.occ_list))]
            occPath = os.path.join(self.occ_dir, occ_name)
            gt_img, img_occ, mask = task.occluded_image_v2(gt_img, occPath, self.resolution)
            gt_img = self.transform(gt_img)
            occimg = self.transform(img_occ)
            mask = np.expand_dims(mask,axis=0)
            mask = torch.from_numpy(mask).float()
        
        id_img = self.transform(id_img)

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)
        return gt_img,occimg*(1-mask),id_img,mask,same
