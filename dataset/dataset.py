import os
import random
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
from torchvision.transforms import ToTensor
import albumentations.pytorch
import matplotlib.pyplot as plt
import re


class MyDataset():
    def __init__(self, cfg, isTextZoom, data_dir, data_annotations_file="", isEval = False,
    mask = False):
        self.datasets_folder_path = cfg.datasets_folder_path
        self.data_dir = data_dir
        self.data_annotations_file = data_annotations_file
        self.cfg = cfg
        self.isTextZoom = isTextZoom
        self.isEval = isEval
        self.mask = mask
        self.regex = re.compile('[^a-zA-Z0-9]')

    def load_dataset(self):

        if self.cfg.enable_first_albumentations:
            transform = A.ReplayCompose([
                # A.RandomBrightnessContrast(p=0.5),
                # A.Rotate(limit=8, p=1, border_mode=cv2.BORDER_CONSTANT),
                A.InvertImg(p=0.5),
                A.ChannelShuffle(p=0.5),
            ])
        else:
            transform = None

        transform_hr = A.Compose([
        A.augmentations.geometric.resize.Resize(self.cfg.height, self.cfg.width, interpolation=cv2.INTER_CUBIC),
        ])

        height_lr = int(self.cfg.height / self.cfg.scale_factor)
        width_lr = int(self.cfg.width / self.cfg.scale_factor)
        if self.cfg.enable_lr_albumentations:
            transform_lr = A.Compose([
                # A.ImageCompression(p=1, quality_lower=50, quality_upper=100),
                A.Blur(p=0.75, blur_limit=[3,7]),
                A.augmentations.geometric.resize.Resize(height_lr, width_lr, interpolation=cv2.INTER_CUBIC),
            ])
        else:
            transform_lr = A.Compose([
                A.augmentations.geometric.resize.Resize(height_lr, width_lr, interpolation=cv2.INTER_CUBIC),
            ])

        transforms = {'transform_init':transform, 'transform_hr':transform_hr, 'transform_lr':transform_lr}

        # Загрузка датасета
        if self.isTextZoom:
            if not self.cfg.enable_first_albumentations_TextZoom:
                transform = None
            
            if not self.cfg.enable_lr_albumentations_TextZoom and self.cfg.TextZoom == 'real':
                transform_lr = A.Compose([
                    A.augmentations.geometric.resize.Resize(height_lr, width_lr, interpolation=cv2.INTER_CUBIC),
                ])

            # if self.cfg.real_TextZoom:
            #     transform_lr = None
            
            # transform_hr = None
            # transform_lr = None
            transforms = {'transform_init':transform, 'transform_hr':transform_hr, 'transform_lr':transform_lr}
            
            imgs_dir_path = self.datasets_folder_path + self.data_dir
            train_dataset = lmdbDataset(root=imgs_dir_path, transforms = transforms,
                                        max_len=self.cfg.max_len, regex=self.regex,
                                        cfg=self.cfg, val=self.isEval, mask=self.mask)
        else:
            annotations_file_path = self.datasets_folder_path + self.data_annotations_file
            imgs_dir_path = self.datasets_folder_path + self.data_dir
            # train_dataset = ICDARImageDataset(annotations_file_path, imgs_dir_path)
            train_dataset = ICDARImageDataset(annotations_file_path, imgs_dir_path, regex=self.regex, transforms=transforms, cfg=self.cfg)
        return train_dataset


class ICDARImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, regex, cfg, transforms=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=', ', header=None, engine='python', na_filter=False)
        self.img_dir = img_dir
        self.transforms = transforms
        self.target_transform = target_transform
        self.dataset_name = os.path.basename(os.path.dirname(img_dir))
        self.to_tensor = ToTensor()
        self.regex = regex
        self.cfg = cfg

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = read_image(img_path, ImageReadMode.RGB).numpy()

        if self.transforms:
            image = np.moveaxis(image, 0, -1)

        label = str(self.img_labels.iloc[idx, 1])
        if not self.cfg.allow_symbols:
            label = self.regex.sub('', label)
        if self.cfg.letters == 'lower':
            label = label.lower()
        elif self.cfg.letters == 'upper':
            label = label.upper()

        if self.transforms:
            image_tr = self.transforms['transform_init'](image=image)['image'] if self.transforms['transform_init'] else image
            image_hr = self.transforms['transform_hr'](image=image_tr)['image'] if self.transforms['transform_hr'] else image_tr
            image_lr = self.transforms['transform_lr'](image=image_tr)['image'] if self.transforms['transform_lr'] else image_tr

            image_hr = self.to_tensor(image_hr)
            image_lr = self.to_tensor(image_lr)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image_hr, image_lr, len(label), label, self.dataset_name


class lmdbDataset(Dataset):
    def __init__(self, regex, cfg, root=None, voc_type='lower', max_len=32,
                 test=False, val=False, transforms=None, mask=False):
        super(lmdbDataset, self).__init__()

        self.root = root
        self.max_len = max_len
        self.voc_type = voc_type
        self.test = test
        self.val = val
        self.transforms = transforms
        self.dataset_name = os.path.basename(os.path.dirname(os.path.dirname(root)))+'_'+os.path.basename(root)
        self.regex = regex
        self.cfg = cfg
        self.mask = mask

        self.open_lmdb()
        self.env.close()
        del self.env

        self.initialize = False

        self.to_tensor = ToTensor()

    def open_lmdb(self):
        self.env = lmdb.open(
            os.path.expanduser(self.root),
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
        # if self.test:
        #     quit(-1)
        #     return self.nSamples * 0.7
        # elif self.val:
        #     quit(-1)
        #     return self.nSamples * 0.3
        # else:
        return self.nSamples

    def __getitem__(self, index):
        if not self.initialize:
            self.open_lmdb()

        assert index <= len(self), 'index range error'
        # if self.val:
        #     index = index + self.nSamples * 0.7 + 1
        # else:
        #     index += 1
        index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        if not self.cfg.allow_symbols:
            word = self.regex.sub('', word)
        if self.cfg.letters == 'lower':
            word = word.lower()
        elif self.cfg.letters == 'upper':
            word = word.upper()
        if len(word)>32:
            return self[index + 1]

        if self.cfg.TextZoom == 'real':
            try:
                img_hr = self.buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                img_lr = self.buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            except TypeError:
                img_hr, img_lr = "", ""
                # img_hr = self.buf2PIL(txn, b'image-%09d' % index, 'RGB')
                # img_lr = self.buf2PIL(txn, b'image-%09d' % index, 'RGB')
            except IOError or len(word) > self.max_len:
                return self[index + 1]
        elif self.cfg.TextZoom == 'syn':
            try:
                img_hr = self.buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                img_lr = img_hr
            except TypeError:
                img_hr, img_lr = "", ""
                # img_hr = self.buf2PIL(txn, b'image-%09d' % index, 'RGB')
                # img_lr = self.buf2PIL(txn, b'image-%09d' % index, 'RGB')
            except IOError or len(word) > self.max_len:
                return self[index + 1]
        elif self.cfg.TextZoom == 'mix':
            try:
                img_hr = self.buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                if self.val or random.uniform(0, 1) < 0.5:
                    img_lr = self.buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
                else:
                    img_lr = img_hr

            except TypeError:
                img_hr, img_lr = "", ""
                # img_hr = self.buf2PIL(txn, b'image-%09d' % index, 'RGB')
                # img_lr = self.buf2PIL(txn, b'image-%09d' % index, 'RGB')
            except IOError or len(word) > self.max_len:
                return self[index + 1]

        # label_str = self.str_filt(word, self.voc_type)
        # img.show()
        # img_np_ma = np.moveaxis(img_np, -1, 0)im = Image.fromarray(A)
        if self.transforms:
            img_np_hr = np.array(img_hr)
            img_np_lr = np.array(img_lr)

            # Image.fromarray(img_np_hr).save(r"/home/helen/TextZoom/img_np_"+str(index)+"_hr.jpg")
            # Image.fromarray(img_np_lr).save(r"/home/helen/TextZoom/img_np_"+str(index)+"_lr.jpg")
            
            if self.transforms['transform_init']:
                image_tr_hr_a = self.transforms['transform_init'](image=img_np_hr)
                image_tr_hr = image_tr_hr_a['image']

                replay = image_tr_hr_a['replay']
                assert replay is not None

                image_tr_lr_a = A.ReplayCompose.replay(replay, image=img_np_lr)
                image_tr_lr = image_tr_lr_a['image']
            else:
                image_tr_hr = img_np_hr
                image_tr_lr = img_np_lr
                
            image_hr_np = self.transforms['transform_hr'](image=image_tr_hr)['image'] if self.transforms['transform_hr'] else image_tr_hr
            image_lr_np = self.transforms['transform_lr'](image=image_tr_lr)['image'] if self.transforms['transform_lr'] else image_tr_lr

            image_hr = self.to_tensor(image_hr_np)
            image_lr = self.to_tensor(image_lr_np)


            if self.mask:
                mask_lr = Image.fromarray(image_lr_np)
                mask_lr = mask_lr.convert('L')
                thres = np.array(mask_lr).mean()
                mask_lr = mask_lr.point(lambda x: 0 if x > thres else 255)
                mask_lr = self.to_tensor(mask_lr)
                image_lr = torch.cat((image_lr, mask_lr), 0)

        # un = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # img_un = un(image_hr['image'])
        # plt.imshow(torch.moveaxis(img_un,0,2))
        # plt.show()



        return image_hr, image_lr, len(word), word, self.dataset_name

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