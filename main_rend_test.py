import logging
import torch
from configparser import ConfigParser

from data import create_dataset, create_dataloader
from models import Supervised
from test import test_only
from log import setup_default_logging
from RD4AD import resnet
_logger = logging.getLogger('train')


def run_test(cfg):
    # setting seed and device
    setup_default_logging()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
        perlin_noise_threshold = float(cfg['DataSet']['perlin_noise_threshold'])
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


    supervised_model = Supervised(feature_extractor = RD4AD_encoder).to(device)

    # Fitting model
    for j in range(0,250):
        print((j+1)*20 ,"epoch model testing...")
        file_path = f'D:/DYP/model_weight/rend/super/supervised_model_{(j+1)*20}.pth'
        folder_path = cfg['Test']['result_dir']
        supervised_model = torch.jit.load(file_path).to(device)
        supervised_model.eval()

        # test_only(
        # supervised_model=supervised_model,
        # dataloader=testloader,
        # folder_path=folder_path,
        # num=(j+1)*20,
        # target=cfg['DataSet']['target'],
        # device=device
        # )

if __name__=='__main__':
    config = ConfigParser()
    config.read('configs/rend_config.ini')
    run_test(config)