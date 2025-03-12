import os
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import random
import cv2
from torchvision.transforms import functional as F


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        label_file_path = 'voc12/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]

class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, gen_attn=False, strong_augment=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn
        self.strong_augment = strong_augment


    def _blur(self, img, p=0.5):
        if random.random() < p:
            sigma = np.random.uniform(0.1, 2.0)
            img = img.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train or self.gen_attn :
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        if self.strong_augment:
            #image_aug = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = img.transpose(1,2,0) #chw->hwc
            #image_aug = np.array(image_aug, np.uint8)
            image_aug = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            image_aug = PIL.Image.fromarray(image_aug)  # array to image
            
            if random.random() < 0.8:
                image_aug = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_aug)
            image_aug = transforms.RandomGrayscale(p=0.2)(image_aug)
            image_aug = self._blur(image_aug)
            image_aug = np.array(image_aug)
            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
            #image_aug -=self.mean_bgr
            #image_aug = image_aug.transpose(2, 0, 1)  #hwc ->chw
            #image_aug = image_aug.astype(np.float32)
            image_aug = F.to_tensor(image_aug)
            #img = img.transpose(1,2,0)  #hwc ->chw
            #img= img.astype(np.float32)
            img = F.to_tensor(img)
            #img = PIL.Image.fromarray(np.array(img.numpy(), np.uint8))  # array to image
            #img = F.to_tensor(img)
            image_aug = F.normalize(image_aug, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            img = F.normalize(img, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            return img, image_aug, label

        return img, label

    def __len__(self):
        return len(self.img_name_list)

class COCOClsDatasetMS(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, scales, train=True, transform=None, gen_attn=False, unit=1):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.unit = unit
        self.scales = scales
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train or self.gen_attn:
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])

            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12Dataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, train=True, transform=None, gen_attn=False, strong_augment=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.strong_augment = strong_augment

    def _blur(self, img, p=0.5):
        if random.random() < p:
            sigma = np.random.uniform(0.1, 2.0)
            img = img.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)
        
        if self.strong_augment:
            #image_aug = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = img.transpose(1,2,0) #chw->hwc
            #image_aug = np.array(image_aug, np.uint8)
            image_aug = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            image_aug = PIL.Image.fromarray(image_aug)  # array to image
            
            if random.random() < 0.8:
                image_aug = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_aug)
            image_aug = transforms.RandomGrayscale(p=0.2)(image_aug)
            image_aug = self._blur(image_aug)
            image_aug = np.array(image_aug)
            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
            #image_aug -=self.mean_bgr
            #image_aug = image_aug.transpose(2, 0, 1)  #hwc ->chw
            #image_aug = image_aug.astype(np.float32)
            image_aug = F.to_tensor(image_aug)
            #img = img.transpose(1,2,0)  #hwc ->chw
            #img= img.astype(np.float32)
            img = F.to_tensor(img)
            #img = PIL.Image.fromarray(np.array(img.numpy(), np.uint8))  # array to image
            #img = F.to_tensor(img)
            image_aug = F.normalize(image_aug, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            img = F.normalize(img, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            return img, image_aug, label
        #img = PIL.Image.fromarray(np.array(img, np.uint8))  # array to image
        #img = img.transpose(1,2,0)  #hwc ->chw
        #img= img.astype(np.float32)
        #img = F.to_tensor(img)
        #img = F.normalize(img, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        return img, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12DatasetMS(Dataset):
    def __init__(self, img_name_list_path, voc12_root, scales, train=True, transform=None, gen_attn=False, unit=1):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)

def build_dataset(is_train, args, gen_attn=False, strong_augment=False):
    if strong_augment:
         transform = build_transform(is_train, args, True)
    else:
         transform = build_transform(is_train, args, False)
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        dataset = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, strong_augment=strong_augment)
        nb_classes = 20
    elif args.data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform)
        # dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=[args.scales],
        #                        train=is_train, gen_attn=gen_attn, transform=transform)
        # dataset = VOC12DatasetMS(img_name_list_path='/home/ysy/cloudcc/ysy_data/VOCdevkit/VOC2012/ImageSets/Segmentation', voc12_root='/home/ysy/cloudcc/ysy_data/VOCdevkit/VOC2012/', scales=[1.0],
        #                        train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif args.data_set == 'COCO':
        dataset = COCOClsDataset(img_name_list_path=args.img_list, coco_root=args.data_path, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, strong_augment=strong_augment)
        nb_classes = 80
    elif args.data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(img_name_list_path=args.img_list, coco_root=args.data_path, scales=tuple(args.scales), label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80

    return dataset, nb_classes


def build_transform(is_train, args, use_prefetcher):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            use_prefetcher=use_prefetcher,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


