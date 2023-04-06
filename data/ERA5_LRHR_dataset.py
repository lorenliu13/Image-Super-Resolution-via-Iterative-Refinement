from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import cv2
import numpy as np


class ERA5_LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        """
        A custom dataset for LR/HR image pairs from ERA5 dataset
        dataroot: directory of lr, hr, sr images
        datatype: data type, 'img' or 'lmdb'
        l_resolution: lower resolution
        r_resolution: higher resolution
        split: phrase, train or val
        data_len: data length
        need_LR: whether need low resolution images
        """
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution)) # get sr image path
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution)) # get hr image path
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution)) # get lr image path
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        """
        A magic method to define the behavior of the "len" function
        when call len function on an object of this class
        Python will automatically call the "__len__" method here.
        """
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else: # if it is 'img' object
            img_HR = cv2.imread(self.hr_path[index], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE) # load the img
            img_HR = (img_HR / 65535).astype(np.float32) # restore it to 0 and 1
            img_SR = cv2.imread(self.sr_path[index], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            img_SR = (img_SR / 65535).astype(np.float32) # restore it to 0 and 1

            if self.need_LR:
                img_LR = cv2.imread(self.lr_path[index], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                img_LR = (img_LR / 65535).astype(np.float32) # restore it to 0 and 1

            # img_HR = Image.open(self.hr_path[index]).convert("RGB") # open the image and convert to RGB
            # img_SR = Image.open(self.sr_path[index]).convert("RGB") # open the image and convert to RGB
            # if self.need_LR:
            #     img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR: # if need_LR is true
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1)) # call a transform_augment function
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else: # if need_LR is false
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
