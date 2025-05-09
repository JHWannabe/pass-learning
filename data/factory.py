from torch.utils.data import DataLoader
from typing import Tuple, List
from .dataset import Dataset
from cutmix.cutmix import CutMix

def create_dataset(
    datadir: str, target: str, train: bool, to_memory: bool = False,
    resize: Tuple[int, int] = (224,224),
    texture_source_dir: str = None, structure_grid_size: str = 8,
    transparency_range: List[float] = [0.15, 1.],
    perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
    dataset_path: str = None,
    retraining : bool = False,
    retraining_period : int = 10
):
    dataset = Dataset(
        datadir                = datadir,
        target                 = target, 
        train                  = train,
        to_memory              = to_memory,
        resize                 = resize,
        texture_source_dir     = texture_source_dir, 
        structure_grid_size    = structure_grid_size,
        transparency_range     = transparency_range,
        perlin_scale           = perlin_scale, 
        min_perlin_scale       = min_perlin_scale, 
        perlin_noise_threshold = perlin_noise_threshold,
        dataset_path = dataset_path,
        retraining = retraining,
        retraining_period = retraining_period
    )

    return dataset


def create_dataloader(dataset, train: bool, batch_size: int = 16, num_workers: int = 1):
    dataloader = DataLoader(
        #CutMix(dataset, 2, beta=1.0, prob=0.5 , num_mix= 3),
        dataset,
        shuffle     = train,
        batch_size  = batch_size,
        num_workers = num_workers
    )

    return dataloader