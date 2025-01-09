import logging
import os
import torch
import torch.nn as nn

from data import create_dataset, create_dataloader
from models import Supervised
from focal_loss import FocalLoss
from train import training_supervise, training_4ch_supervise, write_to_memory_mapped_file
from log import setup_default_logging
from scheduler import CosineAnnealingWarmupRestarts
from RD4AD import resnet
import warnings
from configparser import ConfigParser

_logger = logging.getLogger('supervise')

def run_training(cfg):
    # setting seed and device
    gpu_num = cfg['Train']['Device_GPU']

    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    savedir = cfg['Result']['save_dir']
    
    os.makedirs(savedir, exist_ok=True)

    # build datasets
    trainset = create_dataset(
        datadir                = cfg['DataSet']['data_dir'],
        target                 = cfg['DataSet']['target'],
        train                  = True,
        resize                 = (int(cfg['DataSet']['resize(h)']), int(cfg['DataSet']['resize(w)'])),
        texture_source_dir     = cfg['DataSet']['texture_source_dir'],
        structure_grid_size    = int(cfg['DataSet']['structure_grid_size']),
        transparency_range     = [float(cfg['DataSet']['transparency_range_under_bound']), float(cfg['DataSet']['transparency_range_upper_bound'])],
        perlin_scale           = int(cfg['DataSet']['perlin_scale']), 
        min_perlin_scale       = int(cfg['DataSet']['min_perlin_scale']), 
        perlin_noise_threshold = float(cfg['DataSet']['perlin_noise_threshold']),
        dataset_path           = cfg['Train']['DataSet'],
        retraining = True,
        retraining_period = 0
    )

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
        dataset_path           = cfg['Train']['DataSet']
    )

    # build dataloader
    trainloader = create_dataloader(
        dataset     = trainset,
        train       = True,
        batch_size  = int(cfg['DataLoader']['batch_size']),
        num_workers = int(cfg['DataLoader']['num_workers'])
    )
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = int(cfg['DataLoader']['batch_size']),
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


    RD4AD_encoder.train()

    supervised_model = Supervised(feature_extractor = RD4AD_encoder).to(device)
    #print(supervised_model)

    # Transfer Learning
    if cfg['Train']['transfer'] == 'True':
        transfer_learning_model = cfg['Train']['transfer_learning_dir']
        supervised_model = torch.jit.load(transfer_learning_model).to(device)

    # Set training
    l1_criterion = nn.L1Loss()
    f_criterion = FocalLoss(
        gamma = float(cfg['Train']['focal_gamma']),
        alpha = float(cfg['Train']['focal_alpha'])
    )

    optimizer = torch.optim.AdamW(
        params       = filter(lambda p: p.requires_grad, supervised_model.parameters()),
        lr           = float(cfg['Optimizer']['learning_rate']),
        weight_decay = float(cfg['Optimizer']['weight_decay'])
    )

    if cfg['Scheduler']['use_scheduler']:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps = int(cfg['Train']['epochs']),
            max_lr = float(cfg['Optimizer']['learning_rate']),
            min_lr = float(cfg['Scheduler']['min_learning_rate']),
            warmup_steps   = int(int(cfg['Train']['epochs']) * float(cfg['Scheduler']['warmup_ratio']))
        )
    else:
        scheduler = None

    # Fitting model
    training_4ch_supervise(
        supervised_model = supervised_model,
        num_training_steps = int(cfg['Train']['epochs']),
        trainloader        = trainloader, 
        validloader        = testloader, 
        criterion          = [l1_criterion, f_criterion], 
        loss_weights       = [float(cfg['Train']['l1_weight']), float(cfg['Train']['focal_weight'])],
        resize             = (int(cfg['DataSet']['resize(h)']), int(cfg['DataSet']['resize(w)'])),
        optimizer          = optimizer,
        scheduler          = scheduler,
        log_interval       = int(cfg['Log']['log_interval']),
        eval_interval      = int(cfg['Log']['eval_interval']),
        savedir            = savedir,
        target             = cfg['DataSet']['target'],
        device             = device
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
    _logger.info('train')
    _logger.info(f'config path : {file_path}')
    config.read(file_path)
    write_to_memory_mapped_file(0, 0.0, 0.0, 0.0, 0.0)
    run_training(config)
    # pyinstaller .\main_skirt_train_supervise.py   ("../../")