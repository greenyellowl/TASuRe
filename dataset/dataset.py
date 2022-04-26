import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import lmdb
import six
import sys
from PIL import Image
import string
import bisect
import warnings
import torch
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import albumentations.augmentations.geometric
from torchvision import transforms
import albumentations.pytorch
import matplotlib.pyplot as plt


class MyDataset():
    def __init__(self, cfg, isTextZoom, data_dir, data_annotations_file="", isEval = False):
        self.datasets_folder_path = cfg.datasets_folder_path
        self.data_dir = data_dir
        self.data_annotations_file = data_annotations_file
        self.cfg = cfg
        self.isTextZoom = isTextZoom
        self.isEval = isEval

    def load_dataset(self):

        transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=8, p=1, border_mode=cv2.BORDER_CONSTANT),
        A.InvertImg(p=0.5),
        A.ChannelShuffle(p=0.5),
        ])
        transform_hr = A.Compose([
        A.augmentations.geometric.resize.Resize(self.cfg.height*self.cfg.scale_factor,self.cfg.width*self.cfg.scale_factor),
        A.augmentations.transforms.Normalize(self.cfg.norm_mean, self.cfg.norm_std),
        A.pytorch.transforms.ToTensorV2(),
        ])
        if not self.isEval:
            transform_lr = A.Compose([
            A.augmentations.geometric.resize.Resize(self.cfg.height,self.cfg.width),
            A.ImageCompression(p=0.5, quality_lower=50, quality_upper=100),
            A.Blur(p=0.5),
            A.augmentations.transforms.Normalize(self.cfg.norm_mean, self.cfg.norm_std),
            A.pytorch.transforms.ToTensorV2(),
            ])
            transforms = {'transform_init':transform, 'transform_hr':transform_hr, 'transform_lr':transform_lr}
        else:
            transform_lr = A.Compose([
            A.augmentations.geometric.resize.Resize(self.cfg.height,self.cfg.width),
            A.augmentations.transforms.Normalize(self.cfg.norm_mean, self.cfg.norm_std),
            A.pytorch.transforms.ToTensorV2(),
            ])
            transforms = {'transform_init':None, 'transform_hr':transform_hr, 'transform_lr':transform_lr}


        # Загрузка датасета
        if self.isTextZoom:
            imgs_dir_path = self.datasets_folder_path + self.data_dir
            train_dataset = lmdbDataset(root=imgs_dir_path, transforms = transforms, max_len=self.cfg.max_len)
        else:
            annotations_file_path = self.datasets_folder_path + self.data_annotations_file
            imgs_dir_path = self.datasets_folder_path + self.data_dir
            # train_dataset = ICDARImageDataset(annotations_file_path, imgs_dir_path)
            train_dataset = ICDARImageDataset(annotations_file_path, imgs_dir_path, transforms)
        return train_dataset


class ICDARImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=', ', header=None, engine='python', na_filter=False)
        self.img_dir = img_dir
        self.transforms = transforms
        self.target_transform = target_transform
        self.dataset_name = os.path.basename(os.path.dirname(img_dir))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, ImageReadMode.RGB).numpy()
        # image = read_image(img_path, ImageReadMode.RGB)
        # image = read_image(img_path)
        # image = Image.open(img_path)
        # print(image.shape)
        # print(image.shape)
        # print(image)
        if self.transforms:
            image = np.moveaxis(image, 0, -1)
        label = str(self.img_labels.iloc[idx, 1])
        # label = label
        # plt.imshow(image)
        # plt.show()
        if self.transforms:
            image_tr = self.transforms['transform_init'](image=image)['image'] if self.transforms['transform_init'] else image
            image_hr = self.transforms['transform_hr'](image=image_tr)
            image_lr = self.transforms['transform_lr'](image=image_tr)
        if self.target_transform:
            label = self.target_transform(label)
        # un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # img_un = un(image_hr['image'])
        # plt.imshow(torch.moveaxis(img_un,0,2))
        # plt.show()
        return image_hr['image'], image_lr['image'], label, self.dataset_name


class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='lower', max_len=32, test=False, val=False, transforms=None):
        super(lmdbDataset, self).__init__()

        self.root = root
        self.max_len = max_len
        self.voc_type = voc_type
        self.test = test
        self.val = val
        self.transforms = transforms
        self.dataset_name = os.path.basename(os.path.dirname(os.path.dirname(root)))+'_'+os.path.basename(root)

        self.open_lmdb()
        self.env.close()
        del self.env

        self.initialize = False

    def open_lmdb(self):
        self.env = lmdb.open(
            self.root,
            max_readers=1,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (self.root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        self.initialize = True


    def __len__(self):
        if self.test:
            return self.nSamples * 0.7
        elif self.val:
            return self.nSamples * 0.3
        else:
            return self.nSamples

    def __getitem__(self, index):
        if not self.initialize:
            self.open_lmdb()

        assert index <= len(self), 'index range error'
        if self.val:
            index = index + self.nSamples * 0.7 + 1
        else:
            index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        if len(word)>32:
            return self[index + 1]

        try:
            img = self.buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = self.buf2PIL(txn, b'image-%09d' % index, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]

        # label_str = self.str_filt(word, self.voc_type)
        # img.show()
        if self.transforms:
            img_np = np.array(img)
        # img_np_ma = np.moveaxis(img_np, -1, 0)
        if self.transforms:
            image_tr = self.transforms['transform_init'](image=img_np)['image'] if self.transforms['transform_init'] else img_np
            image_hr = self.transforms['transform_hr'](image=image_tr)['image']
            image_lr = self.transforms['transform_lr'](image=image_tr)['image']

        # un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # img_un = un(image_hr['image'])
        # plt.imshow(torch.moveaxis(img_un,0,2))
        # plt.show()

        return image_hr, image_lr, word, self.dataset_name

    @staticmethod
    def str_filt(str_, voc_type):
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        if voc_type == 'lower':
            str_ = str_.lower()
        for char in str_:
            if char not in alpha_dict[voc_type]:
                str_ = str_.replace(char, '')
        return str_

    @staticmethod
    def buf2PIL(txn, key, type='RGB'):
        imgbuf = txn.get(key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# class resizeNormalize(object):
#     def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
#         self.size = size
#         self.interpolation = interpolation
#         self.toTensor = transforms.ToTensor()
#         self.mask = mask
#
#     def __call__(self, img):
#         print(type(img))
#         img = img.resize(self.size, self.interpolation)
#         img_tensor = self.toTensor(img)
#         if self.mask:
#             mask = img.convert('L')
#             thres = np.array(mask).mean()
#             mask = mask.point(lambda x: 0 if x > thres else 255)
#             mask = self.toTensor(mask)
#             img_tensor = torch.cat((img_tensor, mask), 0)
#         return img_tensor


# class alignCollate_syn(object):
#     def __init__(self, imgH=32, imgW=64, down_sample_scale=2, keep_ratio=False, min_ratio=1, mask=False):
#         self.imgH = imgH
#         self.imgW = imgW
#         self.keep_ratio = keep_ratio
#         self.min_ratio = min_ratio
#         self.down_sample_scale = down_sample_scale
#         self.mask = mask
#
#     def __call__(self, batch):
#         images, label_strs = zip(*batch)
#         imgH = self.imgH
#         imgW = self.imgW
#         transform = resizeNormalize((imgW, imgH), self.mask)
#         transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
#
#         images_hr = [transform(image) for image in images]
#         images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)
#
#         images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images]
#         images_lr = [transform2(image) for image in images_lr]
#         images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)
#
#         return images_hr, images_lr, label_strs
#
#
# class alignCollate_real(alignCollate_syn):
#     def __call__(self, batch):
#         images_HR, images_lr, label_strs = zip(*batch)
#         imgH = self.imgH
#         imgW = self.imgW
#         transform = resizeNormalize((imgW, imgH), self.mask)
#         transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
#
#         images_HR = [transform(image) for image in images_HR]
#         images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)
#
#         images_lr = [transform2(image) for image in images_lr]
#         images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)
#
#         return images_HR, images_lr, label_strs