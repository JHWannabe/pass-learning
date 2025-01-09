import logging
import torch
import torch.nn as nn
from configparser import ConfigParser

from data import create_dataset, create_dataloader
from models import Supervised
from test import test_skirt
from log import setup_default_logging
from utils import torch_seed
from RD4AD import resnet
import time
import warnings
import os

_logger = logging.getLogger('test')


def run_test(cfg):
    # setting seed and device
    setup_default_logging()
    #torch_seed(cfg['SEED']['seed'])

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build datasets
    testset = create_dataset(
        datadir                = cfg['DataSet']['data_dir'],
        target                 = cfg['DataSet']['target'],
        train                  = False,
        resize                 = (int(cfg['DataSet']['resize(h)']), int(cfg['DataSet']['resize(w)'])),
        texture_source_dir     = cfg['DataSet']['texture_source_dir'],
        structure_grid_size    = int(cfg['DataSet']['structure_grid_size']),
        transparency_range     = [float(cfg['DataSet']['transparency_range_under_bound']), float(cfg['DataSet']['transparency_range_upper_bound'])],
        perlin_scale           = int(cfg['DataSet']['perlin_scale']), 
        min_perlin_scale       = int(cfg['DataSet']['min_perlin_scale']), 
        perlin_noise_threshold = float(cfg['DataSet']['perlin_noise_threshold']),
        dataset_path           = cfg['Inference']['DataSet']
    )

    # build dataloader
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = 1,
        num_workers = int(cfg['DataLoader']['num_workers'])
    )

    # load RD4AD network
    RD4AD_encoder, RD4AD_bn = resnet.resnet18(pretrained=False)
    RD4AD_encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    RD4AD_encoder = RD4AD_encoder.to(device)

    for name, param in RD4AD_encoder.named_parameters():
        if 'layer1' in name:
            param.requires_grad = False
        if 'layer2' in name:
            param.requires_grad = False
        if 'layer3' in name:
            param.requires_grad = False


    RD4AD_encoder.eval()

    supervised_model = Supervised(feature_extractor = RD4AD_encoder).to(device)

    for j in range(1, 300, 2):
        epoch = (j+1)
        date = '0106'
        #file_path = cfg['Inference']['weight_dir']
        file_path = f'D:/JHChun/model_weight/skirt/raw_4ch/supervised_model_{date}_{epoch}.pth'
        folder_path = os.path.join(cfg['Inference']['results_dir'], cfg['DataSet']['target'], date+'_raw_4ch')
        supervised_model = torch.jit.load(file_path).to(device)
        supervised_model.eval()

        test_skirt(
        supervised_model=supervised_model,
        dataloader=testloader,
        folder_path=folder_path,
        num=epoch,
        target=cfg['DataSet']['target'],
        device=device
        )  



if __name__=='__main__':
    warnings.filterwarnings('ignore')
    setup_default_logging()
    config = ConfigParser()
    # D:\DeepLearningStudio\common\bin\x64\config\skirt_config.ini
    exe_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리 경로
    parent_path = os.path.abspath(os.path.join(exe_path, "../../"))
    file_path = os.path.join(parent_path, "common", "bin", "x64", "config", "skirt_config.ini")
    file_path = 'D:\JHChun\DeepLearningStudio\python\code\configs\skirt_config.ini'
    _logger.info('test')
    _logger.info(f'config path : {file_path}')
    config.read(file_path)
    time.sleep(90)
    run_test(config)