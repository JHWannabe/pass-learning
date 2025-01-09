import numpy as np
import random
from torch.utils.data.dataset import Dataset
import torch

from cutmix.utils import onehot, rand_bbox


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        # print ('dataset elmts :', len(self.dataset[index]))
        # img , _ , lb , file_name , mask = self.dataset[index]
        img, mask  = self.dataset[index]
        # lb_onehot = onehot(self.num_class, lb)
        # print ('mask cutmix:', mask.shape)
        # lam_list = []


        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                # lam_list.append(1)
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            # img2 , mask2 , lb2, _ , _  = self.dataset[rand_index]
            img2 , mask2  = self.dataset[rand_index]
            # lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            mask[bbx1:bbx2, bby1:bby2] = mask2[bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2])) #implicitly implemented in the mask
            # lam_list.append(lam)
            # lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        # print ('lam_list :', lam_list)
        # print ('mask of cutmix :', mask.shape)
        # return img,lb, mask, file_name,  lb_onehot
        return img, mask #, lb_onehot, lam_list

    def __len__(self):
        return len(self.dataset)