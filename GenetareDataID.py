import os
import numpy as np

root_path = '/data1/zbf_data/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation'

train_path = os.path.join(root_path, 'train_aug.txt')
save_train_path = os.path.join(root_path, 'train_aug_id.txt')

lines = open(train_path).readlines()
save_file = open(save_train_path, 'w')

for line in lines:
    line = line[-16:-5]
    save_file.write(line+'\n')
    print(line)
save_file.close()