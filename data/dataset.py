import cv2
import os
import numpy as np
from glob import glob
from einops import rearrange
import datetime

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa

from data import rand_perlin_2d_np
from typing import List, Tuple
import random

class Dataset(Dataset):
    def __init__(
        self, datadir: str, target: str, train: bool, to_memory: bool = False,
        resize: Tuple[int, int] = (224,224),
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        retraining: bool = False,
        retraining_period: int = 10000
    ):
        # mode
        self.train = train 
        self.to_memory = to_memory

        # load image file list
        self.datadir = datadir
        self.target = target
        match target:
            case "head" | "land":
                self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*.jpg' if train else 'test/*/*.jpg'))
            case "skirt" | "zumul":
                self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*/*.jpg' if train else 'test/*/*/*.jpg'))


        # load texture image file list
        if texture_source_dir:
            self.texture_source_file_list = glob(os.path.join(texture_source_dir,'*/*'))
            
        # synthetic anomaly
        if train:
            self.transparency_range = transparency_range
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            self.structure_grid_size = structure_grid_size
        
        # transform ndarray into tensor
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
        ])

        # sythetic anomaly switch
        self.anomaly_switch = True
        self.retraining =retraining
        self.retraining_period = retraining_period

        match target:
            case "head" | "land":
                self.notfound_img_list = glob(os.path.join('Z:/retraining_imgs', self.target, 'notfound/origin/*.jpg'))
                self.notfound_img_list_num = 0
                self.notfound_label_list = glob(os.path.join('Z:/retraining_imgs', self.target, 'notfound/label/*.jpg'))

                self.overkill_img_list = glob(os.path.join('Z:/retraining_imgs', self.target, 'overkill/origin/*.jpg'))
                self.overkill_img_list_num = 0

                self.all_list = glob(os.path.join('Z:/retraining_imgs', self.target, '*/origin/*.jpg'))
                self.all_list_num = 0

            case "skirt" | "zumul":
                self.notfound_img_list = glob(os.path.join('Z:/retraining_imgs', self.target, 'notfound/origin/*/*.jpg'))
                self.notfound_img_list_num = 0
                self.notfound_label_list = glob(os.path.join('Z:/retraining_imgs', self.target, 'notfound/label/*/*.jpg'))

                self.overkill_img_list = glob(os.path.join('Z:/retraining_imgs', self.target, 'overkill/origin/*/*.jpg'))
                self.overkill_img_list_num = 0

                self.all_list = glob(os.path.join('Z:/retraining_imgs', self.target, '*/origin/*/*.jpg'))
                self.all_list_num = 0

        random.shuffle(self.notfound_img_list)
        random.shuffle(self.overkill_img_list)
        random.shuffle(self.all_list)

        self.period_count = 0
        self.sub_period_count = 0
        self.img_count = 0

    def __getitem__(self, idx):

        if self.retraining == False: # 재학습 아닐때
            file_path = self.file_list[idx]

            if self.target in ['skirt', 'zumul']:
                if '_b' in file_path:
                    file_name = 'b' + os.path.splitext(os.path.basename(file_path))[0]
                elif '_a' in file_path:
                    file_name = 'a' + os.path.splitext(os.path.basename(file_path))[0]
            else:
                file_name = os.path.splitext(os.path.basename(file_path))[0]

            if "NG" in file_path:
                y_true = 0
            else:
                y_true = 1
            
            
            # image
            img = cv2.imread(file_path,cv2.IMREAD_COLOR)
            img_gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
            # mask
            mask = np.zeros(self.resize, dtype=np.float32)

            # anomaly source
            if not self.to_memory and self.train:
                if self.anomaly_switch:
                    img, mask = self.generate_anomaly(img=img)
                    self.anomaly_switch = False
                else:
                    self.anomaly_switch = True

            self.img_count = self.img_count + 1

            img = self.transform(img)
            mask = torch.Tensor(mask).to(torch.int64)

            if(self.train):
                return img, mask, y_true
            else:
                return img, mask, y_true, file_name, img_gray
        
        else: # 재학습 일때
            file_path = self.all_list[idx]

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))

            if 'overkill' in file_path:
                mask = np.zeros(self.resize, dtype=np.int_)

            else:
                label_path = file_path.replace('origin','label')
                mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, dsize=(self.resize[1], self.resize[0])).astype(np.bool_).astype(np.int_)

            img = self.transform(img)
            mask = torch.Tensor(mask).to(torch.int64)

            self.all_list_num = self.all_list_num + 1
            if self.all_list_num == len(self.all_list):
                self.all_list_num = 0

            self.period_count = self.period_count + 1

            return img, mask


            # if (self.period_count == self.retraining_period) and (self.sub_period_count == 0) :  # 미검 update
            #     ## img load
            #     file_path = self.notfound_img_list[self.notfound_img_list_num]
            #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
            #     ## label load
            #     label_path = file_path.replace('origin','label')
            #     mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            #     mask = cv2.resize(mask, dsize=(self.resize[1], self.resize[0])).astype(np.bool_).astype(np.int_)

            #     img = self.transform(img)
            #     mask = torch.Tensor(mask).to(torch.int64)

            #     self.notfound_img_list_num = self.notfound_img_list_num+1
            #     if self.notfound_img_list_num == len(self.notfound_img_list):
            #         self.notfound_img_list_num = 0
            #     self.sub_period_count = 1

            #     return img, mask

            # elif self.period_count == self.retraining_period and (self.sub_period_count == 1):  # 과검 update
            #     ## img load
            #     file_path = self.overkill_img_list[self.overkill_img_list_num]
            #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
            
            #     ## label load
            #     mask = np.zeros(self.resize, dtype=np.int_)
            
            #     img = self.transform(img)
            #     mask = torch.Tensor(mask).to(torch.int64)
            
            #     self.overkill_img_list_num = self.overkill_img_list_num+1
            #     if self.overkill_img_list_num == len(self.overkill_img_list):
            #         self.overkill_img_list_num = 0
            #     self.sub_period_count = 0
            
            #     return img, mask    
        
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-10, 10))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
    
    def generate_anomaly(self, img: np.ndarray, path = []) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask
            
        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation, 
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid 
            and randomly arranged to obtain the disordered image  𝐼 
            
        step 3. blending image and anomaly source
        '''
        
        # step 1. generate mask
        ## target foreground mask
        if path != []:
            target_foreground_mask = self.generate_target_foreground_mask2(img = img ,path = path)
        else :
            target_foreground_mask = self.generate_target_foreground_mask(img=img)  # 뭔가 이상함
        
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()

        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        # step 2. generate texture or structure anomaly
        ## anomaly source
        anomaly_source_img = self.anomaly_source(img=img)

        ## mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img

        return (anomaly_source_img.astype(np.uint8), mask)

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV)

        target_background_mask = target_background_mask.astype(np.bool_).astype(np.int_)

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)
        
        return target_foreground_mask

    def generate_target_foreground_mask2(self, img,path) -> np.ndarray:
        # convert RGB into GRAY scale
        path = path.replace('train','mask')
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.resize(img_gray, dsize=(self.resize[1], self.resize[0]))

        # generate binary mask of gray scale image
        # _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        target_background_mask = target_background_mask.astype(np.bool_).astype(np.int_)

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)

        return target_foreground_mask
 
    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray) -> np.ndarray:
        p = np.random.uniform()
        if p < 0.5:
            # TODO: None texture_source_file_list
            anomaly_source_img = self._texture_source()
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self) -> np.ndarray:
        idx = np.random.choice(len(self.texture_source_file_list))
        texture_source_img = cv2.imread(self.texture_source_file_list[idx])
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=(self.resize[1], self.resize[0])).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        assert self.resize[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = self.resize[1] // self.structure_grid_size
        grid_h = self.resize[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )

        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        if self.retraining == True:
            return len(self.all_list)
        elif self.retraining == False or self.train == False:
            return len(self.file_list)