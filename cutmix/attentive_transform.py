import torchvision.models as models

import torchvision.models as models
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image

class AttentiveTargetTransform(object):
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def __call__(self, target):
        target = [target, self.placeholder['rand_target']]
        return torch.tensor(target)

class AttentiveInputTransform(object):
    def __init__(self, dataset, placeholder, k = 10):
        model = models.resnet50(pretrained=True)
        # print('dataset',dataset)
        # self.dataname = dataname
        self.dataset = dataset
        self.placeholder = placeholder
        self.k = k
        self.grid_size = np.random.randint(1,128)
        temp_model = nn.Sequential(*list(model.children())[:-2])
        self.model = temp_model.cuda()
    
    def __call__(self, image, mask, rand_indices, attentive_regions, gridd, num):
        
        # rand_index = np.random.randint(0, len(self.dataset))
        # rand_indices = np.random.randint(0, len(self.dataset), size=image.shape[0])

        # rand_img, rand_target, _ = self.dataset[rand_index]
        rand_img = []
        rand_target = []

        for idx in rand_indices:
            img, rgh, rgv, target = self.dataset[idx]  # Adjust based on your dataset structure
            
            if num == 1:
                img = rgh
            elif num == 2:
                img = rgv
            rand_img.append(img)
            rand_target.append(target)

        rand_img = torch.stack(rand_img)
        rand_target = torch.stack(rand_target)
        self.placeholder['rand_target'] = rand_target

        # ori_size = image.shape

        # image = image.resize((224, 224))
        # rand_img = rand_img.resize((224, 224))
        # print('rand_img in call ',rand_img.shape)

        # attentive_regions = self._get_attentive_regions(image)
        rand_img,rand_target = self._replace_attentive_regions(rand_img, image, rand_target, mask, attentive_regions, gridd)

        # array = rand_img.squeeze(0).mean(dim=0).numpy()
        # array = (array * 255).astype(np.uint8)
        # image = Image.fromarray(array)
        # image.save("output_image.png")

        # return rand_img.resize(ori_size)
        return rand_img.to('cuda'),rand_target.to('cuda')
    
    def _replace_attentive_regions(self, rand_img, image,rand_mask, mask, attentive_regions, gridd):
        """
        rand_img: the img to be replaced
        image: where the 'patches' come from
        attentive_regions: an array contains the coordinates of attentive regions
        """
        # np_rand_img, np_img = np.array(rand_img), np.array(image)
        for attentive_region in attentive_regions:
            # self._replace_attentive_region(np_rand_img, np_img, attentive_region)
            self._replace_attentive_region(rand_img, image, rand_mask, mask, attentive_region, gridd)
        # return Image.fromarray(np_rand_img)
        return rand_img,rand_mask

    def _replace_attentive_region(self, np_rand_img, np_img, np_rand_mask, np_mask, attentive_region, gridd):
        x, y = attentive_region

        # min_size = min(self.grid_size, min(np_img.shape[2] , np_img.shape[3]))
        # grid_size = np.random.randint(1, min_size)
        # print ('grid size : ', grid_size)
        #gridd = np.random.randint(32,128)
        # print ('grid size : ', gridd)

        
        x1, x2, y1, y2 = gridd * x, gridd * (x+1), gridd * y, gridd * (y+1)
        # x1, x2, y1, y2 = self.grid_size * x, self.grid_size * (x+2), self.grid_size * y, self.grid_size * (y+2)

        # print ('np_img : ',np_img.shape)
        # print ('np_rand_img : ',np_rand_img.shape)
        region = np_img[:,:,x1:x2, y1: y2]
        region_mask = np_mask[:,x1:x2, y1: y2]
        # print ('region : ',region.shape)
        # print ('region_mask : ',region_mask.shape)

        np_rand_img[:,:,x1:x2, y1:y2] = region
        np_rand_mask[:,x1:x2, y1:y2] = region_mask
        # print ('np_rand_mask : ',np_rand_mask.shape)

    def _top_k(self, a):
        k = self.k
        idx = np.argpartition(a.ravel(),a.size-k)[-k:]
        return np.column_stack(np.unravel_index(idx, a.shape))

        
    def _get_attentive_regions(self, image):
    # def _get_attentive_regions(model,_top_k, image):
        """
        CIFAR return top k from 8x8
        ImageNet return top k from 7x7
        """
        # x = TF.to_tensor(image).unsqueeze_(0).cuda()
        x = image
        output = self.model(x)
        # output = model(x)
        last_feature_map = output[0][-1].detach().cpu().numpy()
        # print ('last_feature_map',last_feature_map.shape)
        # print ('self._top_k(last_feature_map) : ',self._top_k(last_feature_map))
        return self._top_k(last_feature_map)
        # return _top_k(last_feature_map)

class GetAttentiveRegions:
    def __init__(self):
        model = models.resnet50(pretrained=True)
        temp_model = nn.Sequential(*list(model.children())[:-2])
        self.model = temp_model.cuda()
        self.k = 10

    def __call__(self, image):
    # def _get_attentive_regions(model,_top_k, image):
        """
        CIFAR return top k from 8x8
        ImageNet return top k from 7x7
        """
        # x = TF.to_tensor(image).unsqueeze_(0).cuda()
        x = image
        output = self.model(x)
        # output = model(x)
        last_feature_map = output[0][-1].detach().cpu().numpy()
        # print ('last_feature_map',last_feature_map.shape)
        # print ('self._top_k(last_feature_map) : ',self._top_k(last_feature_map))
        return self._top_k(last_feature_map)
        # return _top_k(last_feature_map)

    def _top_k(self, a):
        k = self.k
        idx = np.argpartition(a.ravel(),a.size-k)[-k:]
        return np.column_stack(np.unravel_index(idx, a.shape))