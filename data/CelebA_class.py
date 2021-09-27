import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from torchvision import transforms
from PIL import Image
import cv2

class FaceDataSet(Dataset):
    def __init__(self, dataset_path, batch_size, img_size=224):
        """
        Args:
            dataset_path (str): dir path od dataset, should conclude imgs/ and latents/ subdirs.
            batch (int): batch size of data.
            
        """
        super(FaceDataSet, self).__init__()
        pic_dir = os.path.join(dataset_path, 'imgs')
        latent_dir = os.path.join(dataset_path, 'latents')

        tmp_list = os.listdir(pic_dir)
        self.pic_list = []
        self.latent_list = []
        for i in range(len(tmp_list)):
            ext = os.path.splitext(tmp_list[i])[-1]
            if ext in ['.jpg', '.png']:
                self.pic_list.append(pic_dir + '/' + tmp_list[i])
                self.latent_list.append(latent_dir + '/' + tmp_list[i][:-3] + 'npy')

        self.people_num = len(self.pic_list)

        self.type = 0
        self.bs = batch_size
        self.img_size = img_size
        self.count = 0

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        """
        Args:
            index (int): dataset index
        Returns:
            img_id (tensor): image of source person, HxWxC
            img_att (tensor): image of target person, HxWxC
            latent_id (tensor): latent of source person, HxWxC
            latent_att (tensor): latent of target person, HxWxC
            data_type (int): 0 for same person and none zero for different person
        """
        p1 = random.randint(0, self.people_num - 1)
        p2 = p1

        if self.type == 0:
            # load pictures from the same folder
            pass
        else:
            # load pictures from different folders
            while p2 == p1:
                p2 = random.randint(0, self.people_num - 1)

        pic_id_dir = self.pic_list[p1]
        pic_att_dir = self.pic_list[p2]
        latent_id_dir = self.latent_list[p1]
        latent_att_dir = self.latent_list[p2]

        img_id = Image.open(pic_id_dir).convert('RGB')
        img_id = img_id.resize((self.img_size, self.img_size))
        img_id = self.transformer(img_id)
        latent_id = np.load(latent_id_dir)[0] # 512
        latent_id = latent_id / np.linalg.norm(latent_id)
        latent_id = torch.from_numpy(latent_id)

        img_att = Image.open(pic_att_dir).convert('RGB')
        img_att = img_att.resize((self.img_size, self.img_size))
        img_att = self.transformer(img_att)
        latent_att = np.load(latent_att_dir)[0]
        latent_att = latent_att / np.linalg.norm(latent_att)
        latent_att = torch.from_numpy(latent_att)
        
        self.count += 1
        data_type = self.type
        if self.count == self.bs:
            #self.type = 1 - type
            self.type = (self.type + 1) % 5  # better
            self.count = 0
        
        return img_id, img_att, latent_id, latent_att, data_type
        
    def __len__(self):
        return len(self.pic_list)

    def name(self):
        return 'FaceDataSet'
