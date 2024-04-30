from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
import json
from LibMTL.utils import get_root_dir


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        return img_, label_, depth_ / sc, normal_


class NYUv2(Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """
    def __init__(self, root, mode='train', augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.pseudo_labels = {}
        self._pseudo_labels_updated = False

        with open(os.path.join(get_root_dir(), 'examples/nyu', '75labeled_25unlabeled_data.json'), 'r') as f:
            data_split = json.load(f)
        labeled_index, unlabeled_index = data_split['labeled'], data_split['unlabeled']
        
        if self.mode == 'labeled':
            self.index_list = labeled_index
            self.data_path = self.root + '/train'
        elif self.mode == 'unlabeled':
            self.index_list = unlabeled_index
            self.data_path = self.root + '/train'
        elif self.mode == 'test':
            data_len = len(fnmatch.filter(os.listdir(self.root + '/val/image'), '*.npy'))
            self.index_list = list(range(data_len))
            self.data_path = self.root + '/val'

    def update_pseudo_labels(self, pseudo_labels, unlabeled_images=None):
        # Update pseudo labels and images for unlabeled data
        self.pseudo_labels = pseudo_labels
        # self.unlabeled_images = unlabeled_images
        self._pseudo_labels_updated = True
    def __getitem__(self, i):
        index = self.index_list[i]
        if self.mode == 'unlabeled' and self._pseudo_labels_updated:
            
            pseudo_label = {key: value[i].float() for key, value in self.pseudo_labels.items()}
            image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(i)), -1, 0))
            image = image.to("cpu")
            return image.float(), {'normal': pseudo_label['normal'].float().to("cpu")}
        else:
            image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
            semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
            depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
            normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

            if self.augmentation:
                image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)
                if torch.rand(1) < 0.5:
                    image = torch.flip(image, dims=[2])
                    # semantic = torch.flip(semantic, dims=[1])
                    # depth = torch.flip(depth, dims=[2])
                    normal = torch.flip(normal, dims=[2])
                    normal[0, :, :] = - normal[0, :, :]
            # semantic+=1
            # semantic = F.one_hot(semantic.long(), num_classes=14).permute(2, 0, 1)
            return image.float(), {'normal': normal.float()}

    def __len__(self):
        if self.mode == 'unlabeled' and self._pseudo_labels_updated:
            with open(os.path.join(get_root_dir(), 'examples/nyu', '25labeled_75unlabeled_data.json'), 'r') as f:
                data_split = json.load(f)
            labeled_index, unlabeled_index = data_split['labeled'], data_split['unlabeled']
            # return len(labeled_index) + len(unlabeled_index)
            return len(self.index_list) 
        else:
            return len(self.index_list)

from torch.utils.data.dataset import ConcatDataset

class CombinedNYUv2(Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __getitem__(self, i):
        # For the combined dataset, return the item from the appropriate dataset
        if i < len(self.labeled_dataset):
            # print(f"{i=}")
            return self.labeled_dataset[i]
        else:
            # Adjust index for unlabeled dataset
            adjusted_index = i - len(self.labeled_dataset)
            # print(f"{adjusted_index=}")
            return self.unlabeled_dataset[adjusted_index]

    def __len__(self):
        # Combined length of both datasets
        return len(self.labeled_dataset) + len(self.unlabeled_dataset)
