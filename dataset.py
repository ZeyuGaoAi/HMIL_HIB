import os
import cv2
import glob
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import random

from utils import remap_label
from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration
from histomicstk.preprocessing.color_normalization import reinhard

from imgaug import augmenters as iaa
from augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

def padding(x):
    if x.shape[1]<128:
        left = int((128-x.shape[1])/2)
        right = int(128-x.shape[1]-left)
        x = np.pad(x,((0,0),(left,right),(0,0)),'constant',constant_values = (0,0))
    if x.shape[0]<128:
        up = int((128-x.shape[0])/2)
        down = int(128-x.shape[0]-up)
        x = np.pad(x,((up,down),(0,0),(0,0)),'constant',constant_values = (0,0))
    return x

def get_augmentation(rng):
    shape_augs = [
        # * order = ``0`` -> ``cv2.INTER_NEAREST``
        # * order = ``1`` -> ``cv2.INTER_LINEAR``
        # * order = ``2`` -> ``cv2.INTER_CUBIC``
        # * order = ``3`` -> ``cv2.INTER_CUBIC``
        # * order = ``4`` -> ``cv2.INTER_CUBIC``
        # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
        iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate by -A to +A percent (per axis)
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
            shear=(-5, 5),  # shear by -5 to +5 degrees
            rotate=(-179, 179),  # rotate by -179 to +179 degrees
            order=0,  # use nearest neighbour
            backend="cv2",  # opencv for fast processing\
            seed=rng,
        ),
        # set position to 'center' for center crop
        # else 'uniform' for random crop
        iaa.Fliplr(0.5, seed=rng),
        iaa.Flipud(0.5, seed=rng),
    ]
    
    trans_augs = [iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
#                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -A to +A percent (per axis)
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                order=0,  # use nearest neighbour
                backend="cv2",  # opencv for fast processing\
                seed=rng,),
              ]

    input_augs = [
        iaa.OneOf(
            [
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: median_blur(*args, max_ksize=3),
                ),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                ),
            ]
        ),
        iaa.Sequential(
            [
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_saturation(
                        *args, range=(-0.2, 0.2)
                    ),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_brightness(
                        *args, range=(-26, 26)
                    ),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_contrast(
                        *args, range=(0.75, 1.25)
                    ),
                ),
            ],
            random_order=True,
        ),
    ]

    return shape_augs, input_augs, trans_augs

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dataset):
        # parse options
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list
        self.list_sample = self.parse_input_list(root_dataset)

        # mean and std
        self.normalize = transforms.Normalize(
            mean = [0.829, 0.739, 0.829],
            std = [0.099, 0.235, 0.155]
        )

    def parse_input_list(self, root_dataset):
        list_sample = glob.glob('%s/*%s' % (root_dataset, '.npy'))

        self.num_sample = len(list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
        return list_sample

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(np.array(img))
#         img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long()
        return segm
    
    def cropping_center(self, x, crop_shape, batch=False):
        """Crop an input image at the centre.
        Args:
            x: input array
            crop_shape: dimensions of cropped array

        Returns:
            x: cropped array
        """
        orig_shape = x.shape
        if not batch:
            h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
            x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        else:
            h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
            x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        return x

    
class TrainDataset_hierarchy(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(TrainDataset_hierarchy, self).__init__(root_dataset)
        self.root_dataset = root_dataset

        # classify images into two classes: 1. h > w and 2. h <= w
        self.max_length = cfg.max_length
        self.augmentor = get_augmentation(0)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])

    def __getitem__(self, index):
        
#         shape_augs = self.shape_augs.to_deterministic()
#         input_augs = self.input_augs.to_deterministic()

        # load image and label
        record_path = self.list_sample[index]

        record = np.load(record_path, allow_pickle=True)

        patient_label = int(record[()]['p_label'])

        if patient_label == 0:
            patient_label = [0,0,0]
        elif patient_label == 1:
            patient_label = [0,0,1]
        elif patient_label == 2:
            patient_label = [0,1,2]
        elif patient_label == 3:
            patient_label = [0,1,3]
        elif patient_label == 4:
            patient_label = 4
        elif patient_label == 5:
            patient_label = [0,2,4]
        elif patient_label == 6:
            patient_label = [0,2,5]
        elif patient_label == 7:
            patient_label = [0,3,6]
        elif patient_label == 8:
            patient_label = [1,4,7]
        elif patient_label == 9:
            patient_label = [1,4,8]
        elif patient_label == 10:
            patient_label = [1,5,9]
        elif patient_label == 11:
            patient_label = [2,6,10]
        else:
            print('error patient label!')
            
        cell_imgs = record[()]['c_images']
        cell_rates = record[()]['c_rate']/100
        
        index_max = len(cell_imgs) - 1
        while len(cell_imgs)<self.max_length:
            index = random.randint(0, index_max)
            cell_imgs.append(cell_imgs[index])
        
        cell_imgs = cell_imgs[:self.max_length]
        
        cell_imgs_augmented = []
        
        # augmentations
        for img in cell_imgs:
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = reinhard(img, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])
            img = padding(img)
            img = self.shape_augs.augment_image(img)
            img = self.img_transform(img)
            cell_imgs_augmented.append(img)

        return [cell_imgs_augmented, patient_label, cell_rates]

    def __len__(self):
        return self.num_sample


class ValDataset_hierarchy(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(ValDataset_hierarchy, self).__init__(root_dataset)
        self.root_dataset = root_dataset
        self.max_length = cfg.max_length

    def __getitem__(self, index):

        # load image and label
        record_path = self.list_sample[index]

        record = np.load(record_path, allow_pickle=True)

        patient_label = record[()]['p_label']
        if patient_label == 0:
            patient_label = [0,0,0]
        elif patient_label == 1:
            patient_label = [0,0,1]
        elif patient_label == 2:
            patient_label = [0,1,2]
        elif patient_label == 3:
            patient_label = [0,1,3]
        elif patient_label == 4:
            patient_label = 4
        elif patient_label == 5:
            patient_label = [0,2,4]
        elif patient_label == 6:
            patient_label = [0,2,5]
        elif patient_label == 7:
            patient_label = [0,3,6]
        elif patient_label == 8:
            patient_label = [1,4,7]
        elif patient_label == 9:
            patient_label = [1,4,8]
        elif patient_label == 10:
            patient_label = [1,5,9]
        elif patient_label == 11:
            patient_label = [2,6,10]
        else:
            print('error patient label!')
        cell_imgs = record[()]['c_images']
        cell_rates = record[()]['c_rate']/100
        
        index_max = len(cell_imgs) - 1
        while len(cell_imgs)<self.max_length:
            index = random.randint(0, index_max)
            cell_imgs.append(cell_imgs[index])
        
        cell_imgs = cell_imgs[:self.max_length]
        
        cell_imgs_augmented = []
        
        # augmentations
        for img in cell_imgs:
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = reinhard(img, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])
            img = padding(img)
            img = self.img_transform(img)
            cell_imgs_augmented.append(img)
        
        return [cell_imgs_augmented, patient_label, cell_rates]

    def __len__(self):
        return self.num_sample
    