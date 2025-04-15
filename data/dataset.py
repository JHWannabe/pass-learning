import cv2
import os
import numpy as np
from glob import glob
from einops import rearrange

from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa

from data import rand_perlin_2d_np
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
            base_path = datadir
            file_paths = dataset_path.split(';') 
            for file_path in file_paths:
                file_path = os.path.join(base_path, file_path)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        image_paths = file.readlines()
                    all_image_paths.extend([path.strip() for path in image_paths])

            self.file_list = all_image_paths
                
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
            if retraining == False:
                self.pipe = self.load_model('cuda')
            self.mode = 0
        
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
            img = cv2.imread(file_path,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
            mask = np.zeros(self.resize, dtype=np.float32)

            # anomaly source
            if not self.to_memory and self.train:
                if self.anomaly_switch:
                    img, mask = self.generate_anomaly(img=img)
                    # cv2.imwrite("result/anomaly_source.jpg", img)
                    # cv2.imwrite("result/mask.jpg", mask*255)
                    self.anomaly_switch = False
                else:
                    self.anomaly_switch = True

            self.img_count = self.img_count + 1

            img = self.transform(img)
            mask = torch.Tensor(mask).to(torch.int64)

            if(self.train):
                return img, mask, y_true
            else:
                return img, mask, y_true, file_name            
        
        else: # 재학습 일때
            if self.target == 'head':
                file_path = self.all_list[idx]
                img_inh = cv2.imread(file_path,cv2.IMREAD_COLOR)
                img_inh = cv2.cvtColor(img_inh, cv2.COLOR_BGR2RGB)
                img_inh = cv2.resize(img_inh, dsize=(self.resize[1], self.resize[0]))
                mask = np.zeros(self.resize, dtype=np.int_)
                
                img_inh = self.transform(img_inh)

                if 'notfound' in file_path:
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

                return img, mask
        
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
        # anomaly_source_img = self.stable_diffusion(img=img)

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
    
    def load_model(self, device):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
        ).to(device)

        return pipe

    def stable_diffusion(self, img: np.ndarray) -> np.ndarray:
        prompt = "Torn, contaminated, scratch"
        negative_prompt = "clean, smooth, flawless"

        img = img.astype(np.uint8)  # uint8 타입으로 변환
        init_image = Image.fromarray(img)
        init_image = init_image.resize((512, 512))

        anomaly_source_img = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,  # 0.0~1.0 (값이 높을수록 원본에서 멀어짐)
            guidance_scale=7.5,  # 프롬프트 반영 강도 (1~20, 기본값: 7.5)
            negative_prompt=negative_prompt,
        ).images[0]

        anomaly_source_img = anomaly_source_img.resize((img.shape[1], img.shape[0]))
            
        return np.array(anomaly_source_img)
    
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