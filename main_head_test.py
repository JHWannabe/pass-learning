import logging
import torch
from configparser import ConfigParser

from data import create_dataset, create_dataloader
from models import Supervised
from test import test_head_only, test_individual
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

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
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
    RD4AD_encoder, RD4AD_bn = resnet.resnet18(pretrained=True)
    RD4AD_encoder = RD4AD_encoder.to(device)
    RD4AD_encoder.eval()

    supervised_model1 = Supervised(feature_extractor = RD4AD_encoder).to(device)
    supervised_model2 = Supervised(feature_extractor = RD4AD_encoder).to(device)
    supervised_model3 = Supervised(feature_extractor = RD4AD_encoder).to(device)

    for j in range(2, 100, 3):
        epoch = (j+1)
        date = '0109'
        #file_path = cfg['Inference']['weight_dir']
        file_path = f'D:/JHChun/model_weight/head/loss_a_cutmix_ensemble/inh/supervised_model_{epoch}.pth'
        folder_path = os.path.join(cfg['Inference']['results_dir'], cfg['DataSet']['target'], date+'_dynamic_loss_a_cutmix_ensemble')
        supervised_model1 = torch.jit.load(file_path).to(device)
        supervised_model1.eval()

        file_path = file_path.replace('inh', 'rgh')
        supervised_model2 = torch.jit.load(file_path).to(device)
        supervised_model2.eval()

        file_path = file_path.replace('rgh', 'rgv')
        supervised_model3 = torch.jit.load(file_path).to(device)
        supervised_model3.eval()

        test_head_only(
        supervised_model1=supervised_model1,
        supervised_model2=supervised_model2,
        supervised_model3=supervised_model3,
        dataloader=testloader,
        folder_path=folder_path,
        num=epoch,
        target=cfg['DataSet']['target'],
        device=device
        )
        
        #time.sleep(60)



if __name__=='__main__':
    warnings.filterwarnings('ignore')
    setup_default_logging()
    config = ConfigParser()
    # D:\DeepLearningStudio\common\bin\x64\config\head_config.ini
    exe_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리 경로
    parent_path = os.path.abspath(os.path.join(exe_path, "../../"))
    file_path = os.path.join(parent_path, "common", "bin", "x64", "config", "head_config.ini")
    
    _logger.info('test')
    _logger.info(f'config path : {file_path}')
    config.read(file_path)

    time.sleep(40)
    
    run_test(config)