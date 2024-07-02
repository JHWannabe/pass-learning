import time
import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from datetime import datetime

import mmap
import struct
from multiprocessing import Lock

_logger = logging.getLogger('train')
import cv2

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

file_path = "./configs/example.dat"
file_size = 1024  # 1KB
lock = Lock()

def write_to_memory_mapped_file(epoch, loss, data_time, batch_time, learning_rate):
    # 모든 데이터 유형에 필요한 총 크기 계산
    total_size = struct.calcsize('iffff')
     # 데이터를 이진 형식으로 패킹
    packed_data = struct.pack('iffff', epoch, loss, data_time, batch_time, learning_rate)
    with open(file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), length=total_size, access=mmap.ACCESS_WRITE)
        lock.acquire()
        try:
            # 메모리 맵 파일에 패킹된 데이터 쓰기
            mm.write(packed_data)
            mm.flush()
        finally:
            lock.release()
        mm.close()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def training_unsupervise(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4],
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, target: str = None,
             device: str = 'cpu') -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    s_losses_m = AverageMeter()
    s_l1_losses_m = AverageMeter()
    s_focal_losses_m = AverageMeter()

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set train mode
    supervised_model.train()
    # set optimizer
    optimizer.zero_grad()

    # training
    step = 0
    train_mode = True
    while train_mode:

        end = time.time()
        for inputs, masks, success in trainloader:
            # batch
            inputs, masks = inputs.to(device), masks.to(device)

            # predict 1
            s_outputs = supervised_model(inputs)
            s_outputs = F.sigmoid(s_outputs)

            s_l1_loss = l1_criterion(s_outputs.squeeze(1), masks)

            scores = torch.cat([1 - s_outputs, s_outputs], dim=1)
            s_focal_loss = focal_criterion(scores, masks)

            s_loss = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

            s_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            s_losses_m.update(s_loss.item())
            s_l1_losses_m.update(s_l1_loss.item())
            s_focal_losses_m.update(s_focal_loss.item())
            batch_time_m.update(time.time() - end)

            if (step + 1) % log_interval == 0 or step == 0:
                _logger.info('TRAIN [{:>4d}/{}] '
                             's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                             's_L1 Loss: {s_l1_loss.val:>.3e} ({s_l1_loss.avg:>.3e}) '
                             's_Focal Loss: {s_focal_loss.val:>.3e} ({s_focal_loss.avg:>.3e}) '
                             'LR: {lr:.3e} '
                             'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                             'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    step + 1, num_training_steps,
                    s_loss=s_losses_m,
                    s_l1_loss=s_l1_losses_m,
                    s_focal_loss=s_focal_losses_m,
                    lr=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time_m,
                    rate=inputs.size(0) / batch_time_m.val,
                    rate_avg=inputs.size(0) / batch_time_m.avg,
                    data_time=data_time_m))

            if ((step + 1) % eval_interval == 0 and step != 0) or (step + 1) == num_training_steps:
                os.makedirs(os.path.join(savedir, f'{target}/unsuper'), exist_ok=True)
                tt = torch.jit.trace(supervised_model, inputs)
                current_date = datetime.now().strftime('%m%d')
                model_path = f'{target}/unsuper/unsupervised_model_{current_date}_{(step + 5001)}.pth'
                tt.save(os.path.join(savedir, model_path))

            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break

def training_supervise(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4],
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, target: str = None,
             device: str = 'cpu'):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    s_losses_m = AverageMeter()
    s_l1_losses_m = AverageMeter()
    s_focal_losses_m = AverageMeter()

    # set train mode
    supervised_model.train()

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set optimizer
    optimizer.zero_grad()

    # training
    step = 0
    train_mode = True
    while train_mode:

        end = time.time()
        for inputs, masks in trainloader:
            # batch
            inputs, masks = inputs.to(device), masks.to(device)

            data_time_m.update(time.time() - end)

            # predict 1
            s_outputs = supervised_model(inputs)
            s_outputs = F.sigmoid(s_outputs)

            match target:
                case "land" | "zumul" | "head":
                    s_l1_loss = l1_criterion(s_outputs.squeeze(1), masks)

                    scores = torch.cat([1 - s_outputs, s_outputs], dim=1) 
                    s_focal_loss = focal_criterion(scores, masks)
                    s_loss = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

                    s_loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    # log loss
                    s_losses_m.update(s_loss.item())
                    s_l1_losses_m.update(s_l1_loss.item())
                    s_focal_losses_m.update(s_focal_loss.item())
                    batch_time_m.update(time.time() - end)

                case "skirt":
                    s_l1_loss = l1_criterion(s_outputs.squeeze(1), masks)

                    scores = torch.cat([1 - s_outputs, s_outputs], dim=1)
                    s_focal_loss = focal_criterion(scores, masks)

                    s_loss = s_focal_loss
                    s_loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    # log loss
                    s_losses_m.update(s_loss.item())
                    batch_time_m.update(time.time() - end)


            if (step + 1) % log_interval == 0 or step == 0:
                _logger.info('TRAIN [{:>4d}/{}] '
                             's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                             'LR: {lr:.3e} '
                             'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                             'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    step + 1, num_training_steps,
                    s_loss=s_losses_m,
                    lr=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time_m,
                    rate=inputs.size(0) / batch_time_m.val,
                    rate_avg=inputs.size(0) / batch_time_m.avg,
                    data_time=data_time_m))
                
            # memory map test
            write_to_memory_mapped_file(step+1, s_losses_m.val, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

            if ((step + 1) % eval_interval == 0 and step != 0) or (step + 1) == num_training_steps:
                os.makedirs(os.path.join(savedir, f'{target}/super'), exist_ok=True)
                tt = torch.jit.trace(supervised_model, inputs)
                current_date = datetime.now().strftime('%m%d')
                model_path = f'{target}/super/supervised_model_{current_date}_{(step + 1)}.pth'
                tt.save(os.path.join(savedir, model_path))

            # schedulerS
            if scheduler:
                scheduler.step(step)

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break