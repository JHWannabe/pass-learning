import torch.nn as nn
from .decoder import Decoder
import numpy as np
import torch
from torch.nn import functional as F


def diffmap(fs_list, ft_list):

    mse_loss = torch.nn.MSELoss(reduction='none')
    diff_list = []
    for i in range(len(ft_list)):
        fs_norm = F.normalize(fs_list[i],p=2)
        ft_norm = F.normalize(ft_list[i],p=2)

        diff_features = mse_loss(fs_norm, ft_norm)
        diff_list.append(diff_features)
    return diff_list

def loss_fucntion(a, b, train,img_size):
    # mse_loss = torch.nn.MSELoss(reduction='none') # 테스트시 사용 해야함
    if train == True :
        mse_loss = torch.nn.MSELoss()
        loss = 0
    else :
        mse_loss = torch.nn.MSELoss(reduction='none')
        anomaly_map = np.zeros([img_size[0], img_size[1]])

    # cos_loss = torch.nn.CosineSimilarity()

    for item in range(len(a)):

        a_norm = F.normalize(a[item],p=2)
        b_norm = F.normalize(b[item],p=2)

        if train == True :
            loss += mse_loss(a_norm, b_norm)
        else :
            a_map = mse_loss(a_norm,b_norm).mean(dim=[1])
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=[img_size[0], img_size[1]], mode='bilinear', align_corners=True)
            a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
            anomaly_map += a_map

    if train == True :
        return loss
    else :
        return anomaly_map

class Supervised(nn.Module):
    def __init__(self, feature_extractor):
        super(Supervised, self).__init__()

        self.feature_extractor = feature_extractor
        self.decoder = Decoder()

    def forward(self, img):
        # extract features
        features = self.feature_extractor(img)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        skip_outputs = f_ii
        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + skip_outputs
        )

        return predicted_mask
    

class Unsupervised(nn.Module):
    def __init__(self, RD4AD_bn, RD4AD_decoder, feature_extractor):
        super(Unsupervised, self).__init__()

        self.feature_extractor = feature_extractor
        self.RD4AD_bn = RD4AD_bn
        self.RD4AD_decoder = RD4AD_decoder
        self.decoder = Decoder()

    def forward(self, normal_img,train):

        inputs = self.feature_extractor(normal_img)   ## resnet share

        f_in = inputs[0]
        f_out = inputs[-1]

        inputs = inputs[1:4]

        output1 = self.RD4AD_bn(inputs[0], inputs[1], inputs[2])
        RD4AD_outputs = self.RD4AD_decoder(output1)

        distill_loss_or_map = loss_fucntion(inputs,RD4AD_outputs, train, [768,1024])

        # ####extract concatenated information(CI)
        difference_features = diffmap(fs_list=inputs,ft_list=RD4AD_outputs)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + difference_features
        )


        return predicted_mask,distill_loss_or_map


class Unsupervised_for_test(nn.Module):
    def __init__(self, RD4AD_bn, RD4AD_decoder, feature_extractor):
        super(Unsupervised_for_test, self).__init__()

        self.feature_extractor = feature_extractor
        self.RD4AD_bn = RD4AD_bn
        self.RD4AD_decoder = RD4AD_decoder
        self.decoder = Decoder()

    def forward(self, normal_img):
        inputs = self.feature_extractor(normal_img)  ## resnet share

        f_in = inputs[0]
        f_out = inputs[-1]

        inputs = inputs[1:4]

        output1 = self.RD4AD_bn(inputs[0], inputs[1], inputs[2])
        RD4AD_outputs = self.RD4AD_decoder(output1)


        # ####extract concatenated information(CI)
        difference_features = diffmap(fs_list=inputs, ft_list=RD4AD_outputs)

        # decoder
        predicted_mask = self.decoder(
            encoder_output=f_out,
            concat_features=[f_in] + difference_features
        )

        return predicted_mask