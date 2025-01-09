import cv2
import os
import numpy as np
from glob import glob
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa

from data import rand_perlin_2d_np
from data.concat import concatenate_channels, concatenate_4channels
import albumentations
import random
from typing import List, Tuple

aug = albumentations.Compose([
                    albumentations.ShiftScaleRotate(shift_limit=random.uniform(0,0.1), scale_limit=0, rotate_limit=random.uniform(-0.5,0.5), p=0.1)
                ], p=0.1)

class Dataset(Dataset):
    def __init__(
        self, datadir: str, target: str, train: bool, to_memory: bool = False,
        resize: Tuple[int, int] = (224,224),
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        dataset_path: list = [],
        retraining: bool = False,
        retraining_period: int = 10000
    ):
        # mode
        self.train = train 
        self.to_memory = to_memory

        # load image file list
        self.datadir = datadir
        self.target = target
        all_image_paths = []
        if not retraining:
            match target:
                case "head":
                    base_path = datadir
                    file_paths = dataset_path.split(';') 
                    for file_path in file_paths:
                        file_path = os.path.join(base_path, file_path)
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as file:
                                image_paths = file.readlines()
                            all_image_paths.extend([path.strip() for path in image_paths])

                    self.file_list = all_image_paths

                case "mold":
                    base_path = datadir
                    file_paths = dataset_path.split(';') 
                    for file_path in file_paths:
                        file_path = os.path.join(base_path, file_path)
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as file:
                                image_paths = file.readlines()
                            all_image_paths.extend([path.strip() for path in image_paths])

                    self.file_list = all_image_paths

                case "skirt":
                    base_path = datadir
                    file_paths = dataset_path.split(';') 
                    for file_path in file_paths:
                        file_path = os.path.join(base_path, file_path)
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as file:
                                image_paths = file.readlines()
                            all_image_paths.extend([path.strip() for path in image_paths])

                    self.file_list = all_image_paths

                case "land":
                    self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*.jpg' if train else 'test/*/*.jpg'))
                    self.file_list = self.file_list + glob(os.path.join(self.datadir, self.target, 'train/*.bmp' if train else 'test/*/*.bmp'))

                    # 새 폴더 경로 생성
                    base_dir = os.path.dirname(self.file_list[0])
                    new_dir = os.path.join(base_dir, 'divide')
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    if 'GOOD' in new_dir:
                        other_dir = new_dir.replace('GOOD', 'NG')
                    elif 'NG' in new_dir:
                        other_dir = new_dir.replace('NG', 'GOOD')
                    
                    # 새 폴더가 존재하고, 그 안의 파일 수가 원본 파일 수의 4배인지 확인
                    print(len(os.listdir(new_dir)) + len(os.listdir(other_dir)))
                    if os.path.exists(new_dir) and len(os.listdir(new_dir)) + len(os.listdir(other_dir))== len(self.file_list) * 4:
                        # 이미 모든 이미지가 분할되어 있으므로, 분할된 이미지 경로로 file_list 업데이트
                        self.file_list = [os.path.join(new_dir, f) for f in os.listdir(new_dir)] + [os.path.join(other_dir, f) for f in os.listdir(other_dir)]
                    else:
                        # 분할이 필요한 경우
                        new_file_list = []
                        for file_path in self.file_list:
                            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                            height, width = img.shape[:2]
                            slice_width = width // 4
                            
                            # 새 폴더 생성
                            os.makedirs(new_dir, exist_ok=True)
                            
                            # 이미지를 4등분하여 저장
                            for i in range(4):
                                start = i * slice_width
                                end = (i + 1) * slice_width if i < 3 else width
                                
                                # 새 파일 이름 생성
                                base_dir = os.path.dirname(file_path)
                                new_dir = os.path.join(base_dir, 'divide')
                                base_name = os.path.splitext(os.path.basename(file_path))[0]
                                new_file_name = f"{base_name}_{i+1}.bmp"
                                new_file_path = os.path.join(new_dir, new_file_name)
                                
                                # 파일이 이미 존재하는지 확인
                                if not os.path.exists(new_file_path):
                                    slice_img = img[:, start:end]
                                    cv2.imwrite(new_file_path, slice_img)
                                
                                new_file_list.append(new_file_path)
                        
                        # file_list 재정의
                        self.file_list = new_file_list
                
        else:
            # match target:
            base_path = datadir
            file_paths = dataset_path.split(';') 
            for file_path in file_paths:
                file_path = os.path.join(base_path, file_path)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        image_paths = file.readlines()
                    all_image_paths.extend([path.strip() for path in image_paths])

        # self.all_list = glob(os.path.join('D:/piston_image/retraining_imgs', self.target, '*/origin/*.jpg'))
        # self.all_list = self.all_list + glob(os.path.join('D:/piston_image/retraining_imgs', self.target, '*/origin/*.bmp'))
            self.all_list = all_image_paths
            self.all_list_num = 0

            # case "skirt" | "mold":
            #     self.all_list = glob(os.path.join('D:/piston_image/retraining_imgs', self.target, '*/origin/*/*.jpg'))
            #     self.all_list = self.all_list + glob(os.path.join('D:/piston_image/retraining_imgs', self.target, '*/*/origin/*.bmp'))
            #     self.all_list_num = 0

            random.shuffle(self.all_list)

        self.period_count = 0
        self.sub_period_count = 0
        self.img_count = 0

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

    def __getitem__(self, idx):
        if self.retraining == False: # 재학습 아닐때
            file_path = self.file_list[idx]

            if self.target in ['skirt', 'mold']:
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
            if self.target == 'head':
                rgh_path = file_path.replace('.bmp', '_RGH.bmp').replace('origin', 'rgh')
                rgv_path = file_path.replace('.bmp', '_RGV.bmp').replace('origin', 'rgv')

                img_inh = cv2.imread(file_path,cv2.IMREAD_COLOR)
                img_rgh = cv2.imread(rgh_path,cv2.IMREAD_COLOR)
                img_rgv = cv2.imread(rgv_path,cv2.IMREAD_COLOR)
                img_gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)

                img_inh = cv2.cvtColor(img_inh, cv2.COLOR_BGR2RGB)
                img_rgh = cv2.cvtColor(img_rgh, cv2.COLOR_BGR2RGB)
                img_rgv = cv2.cvtColor(img_rgv, cv2.COLOR_BGR2RGB)

                img_inh = cv2.resize(img_inh, dsize=(self.resize[1], self.resize[0]))
                img_rgh = cv2.resize(img_rgh, dsize=(self.resize[1], self.resize[0]))
                img_rgv = cv2.resize(img_rgv, dsize=(self.resize[1], self.resize[0]))

            elif self.target == 'skirt':                
                img_gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                _1_path = file_path.replace('_0', '_1')
                _2_path = file_path.replace('_0', '_2')
                _3_path = file_path.replace('_0', '_3')
                img = concatenate_4channels(file_path, _1_path, _2_path, _3_path)
                img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
                transform = transforms.ToTensor()
                img = transform(img)

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

                mask = torch.Tensor(mask).to(torch.int64)

                if(self.train):
                    return img, mask, y_true
                else:
                    return img, mask, y_true, file_name, img_gray

            else:
                img = cv2.imread(file_path,cv2.IMREAD_COLOR)
                img_gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
                img_gray = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))

                img = self.preprocessing(img)

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

            img_inh = self.transform(img_inh)
            img_rgh = self.transform(img_rgh)
            img_rgv = self.transform(img_rgv)
            mask = torch.Tensor(mask).to(torch.int64)

            if(self.train):
                return img, mask, y_true
            else:
                return img_inh, img_rgh, img_rgv, mask, y_true, file_name, img_gray
        
        else: # 재학습 일때
            if self.target == 'head':
                file_path = self.all_list[idx]
                rgh_path = file_path.replace('.bmp', '_RGH.bmp').replace('origin', 'rgh')
                rgv_path = file_path.replace('.bmp', '_RGV.bmp').replace('origin', 'rgv')

                img_inh = cv2.imread(file_path,cv2.IMREAD_COLOR)
                img_rgh = cv2.imread(rgh_path,cv2.IMREAD_COLOR)
                img_rgv = cv2.imread(rgv_path,cv2.IMREAD_COLOR)

                img_inh = cv2.cvtColor(img_inh, cv2.COLOR_BGR2RGB)
                img_rgh = cv2.cvtColor(img_rgh, cv2.COLOR_BGR2RGB)
                img_rgv = cv2.cvtColor(img_rgv, cv2.COLOR_BGR2RGB)

                img_inh = cv2.resize(img_inh, dsize=(self.resize[1], self.resize[0]))
                img_rgh = cv2.resize(img_rgh, dsize=(self.resize[1], self.resize[0]))
                img_rgv = cv2.resize(img_rgv, dsize=(self.resize[1], self.resize[0]))
                
                img_inh = self.transform(img_inh)
                img_rgh = self.transform(img_rgh)
                img_rgv = self.transform(img_rgv)

                if 'overkill' in file_path:
                    mask = np.zeros(self.resize, dtype=np.int_)
                else:
                    label_path = file_path.replace('origin','label')
                    if(file_path.find('bmp')):
                        label_path = label_path.replace('bmp','jpg')
                    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, dsize=(self.resize[1], self.resize[0])).astype(np.bool_).astype(np.int_)

                mask = torch.Tensor(mask).to(torch.int64)

                self.all_list_num = self.all_list_num + 1
                if self.all_list_num == len(self.all_list):
                    self.all_list_num = 0

                self.period_count = self.period_count + 1

                return img_inh, img_rgh, img_rgv, mask

            
            elif self.target == 'skirt':
                file_path = self.all_list[idx]
                _1_path = file_path.replace('_0', '_1')
                _2_path = file_path.replace('_0', '_2')
                _3_path = file_path.replace('_0', '_3')
                img = concatenate_4channels(file_path, _1_path, _2_path, _3_path)
                img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
                transform = transforms.ToTensor()
                img = transform(img)

                if 'overkill' in file_path:
                    mask = np.zeros(self.resize, dtype=np.int_)
                else:
                    label_path = file_path.replace('origin','label')
                    if(file_path.find('_0.bmp')):
                        label_path = label_path.replace('_0.bmp','.jpg')
                    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, dsize=(self.resize[1], self.resize[0])).astype(np.bool_).astype(np.int_)

                mask = torch.Tensor(mask).to(torch.int64)

                self.all_list_num = self.all_list_num + 1
                if self.all_list_num == len(self.all_list):
                    self.all_list_num = 0

                self.period_count = self.period_count + 1

                return img, mask
            
    def preprocessing(self, img):
        match self.target:
            case "head":
                gamma = 1.8
            case "land":
                gamma = 3.0
            case _:
                return img

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_channel = hsv_img[:,:,2]

        gamma_corr_v = np.power(v_channel, gamma)
        # since we are applying a power operation, we need to scale back all the values to 0-255
        gamma_max = gamma_corr_v.max()
        gamma_corr_v = gamma_corr_v * 255 / gamma_max
        
        #update the v channel
        v_channel = gamma_corr_v[:,:]
        #update teh v_channel in the hsv image
        hsv_img[:,:,2] = v_channel[:,:]
        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)      

        return img     
        
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