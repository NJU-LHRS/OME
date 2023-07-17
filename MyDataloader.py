import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path

'''
class ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img
'''

class ClsDataset(Dataset):
    def __init__(self, data_dir, label_npy, transform=None):
        self.data_dir = data_dir
        self.label_npy = label_npy
        self.transform = transform

        self.labels_dict = np.load(self.label_npy,allow_pickle=True).item()
        self.images_name = [i for i in self.labels_dict]
        self.images_dir = [os.path.join(self.data_dir,image_name+'.png') for image_name in self.labels_dict]
        self.labels = list(self.labels_dict.values())

    def __len__(self):
        return len(self.labels_dict)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        label = torch.tensor(self.labels[idx],dtype=float).view(-1)
        img = PIL.Image.open(self.images_dir[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return name, img, label

class BinaryClsDataset(ClsDataset):
    def __init__(self, data_dir, label_npy, classid, transform=None):
        super().__init__(data_dir, label_npy, transform)
        self.classid = classid

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)
        label = label[[self.classid]]
        return name, img, label

class ClsDatasetMSF(ClsDataset):

    def __init__(self, data_dir, label_npy, scales, inter_transform=None, unit=1):
        super().__init__(data_dir, label_npy, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

# class ClsDatasetMSFwithSegmask(ClsDatasetMSF):
#
#     def __init__(self, data_dir, label_npy, mask_dir, scales, inter_transform=None, unit=1):
#         super().__init__(data_dir, label_npy, scales, inter_transform, unit)
#         self.mask_dir = mask_dir
#
#     def __getitem__(self, idx):
#         name, msf_img_list, label = super().__getitem__(idx)
#
#         mask_path = os.path.join(self.mask_dir, name+'.png')
#         mask = PIL.Image.open(mask_path)
#         mask = torch.from_numpy(np.asarray(mask, dtype=float))
#
#         return name, msf_img_list, label, mask

class BinaryClsDatasetMSF(ClsDatasetMSF):
    def __init__(self, data_dir, label_npy, scales, classid, inter_transform=None, unit=1):
        super().__init__(data_dir, label_npy, scales, inter_transform, unit)
        self.classid = classid

    def __getitem__(self, idx):
        name, msf_img_list, label = super().__getitem__(idx)
        label = label[[self.classid]]
        return name, msf_img_list, label