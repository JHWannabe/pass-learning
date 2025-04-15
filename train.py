import time
import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image

import mmap
import struct
from multiprocessing import Lock
from ctypes import windll
import ctypes
import onnx
import onnxruntime
from torchvision.transforms import ToPILImage


_logger = logging.getLogger('train')
import cv2

import torchvision.transforms as T
from PIL import Image
import os

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

#file_path = "D:/DeepLearningStudio/common/bin/x64/config/train_mmf.dat"
exe_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리 경로
parent_path = os.path.abspath(os.path.join(exe_path))
file_path = os.path.join(parent_path, "configs", "train_mmf.dat")
_logger.info(file_path)
file_size = 1024  # 1KB
mutex_name = "Global\\memorymapmutex"

def write_to_memory_mapped_file(epoch, loss, data_time, batch_time, learning_rate):
    try:
        # Mutex 열기
        mutex = create_or_open_mutex(mutex_name)
        # Mutex 대기
        wait_for_mutex(mutex)

        # 모든 데이터 유형에 필요한 총 크기 계산
        total_size = struct.calcsize('iffff')
        # 데이터를 이진 형식으로 패킹
        packed_data = struct.pack('iffff', epoch, loss, data_time, batch_time, learning_rate)
        with open(file_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), length=total_size, access=mmap.ACCESS_WRITE)
            mm.seek(0)
            mm.write(packed_data)
            mm.flush()
            mm.close()
    # Mutex 해제
    finally:
        release_mutex(mutex)

# Mutex 생성 및 대기 함수
def create_or_open_mutex(name):
    kernel32 = windll.kernel32
    # Mutex 생성
    mutex = kernel32.CreateMutexA(None, False, name.encode('utf-8'))
    if not mutex:
        raise ValueError("Mutex 생성에 실패했습니다.")
    return mutex

def wait_for_mutex(mutex):
    kernel32 = windll.kernel32
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 0x00000102
    WAIT_FAILED = 0xFFFFFFFF
    INFINITE = 0xFFFFFFFF
    
    result = kernel32.WaitForSingleObject(mutex, INFINITE)
    if result == WAIT_FAILED:
        raise ctypes.WinError(ctypes.get_last_error())
    elif result != WAIT_OBJECT_0:
        raise RuntimeError("Failed to wait for mutex, result code: {}".format(result))
    
def release_mutex(mutex):
    kernel32 = windll.kernel32
    if not kernel32.ReleaseMutex(mutex):
        raise ctypes.WinError(ctypes.get_last_error())
    
def calculate_mIoU(preds, targets, num_classes=2):
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    preds = preds.argmax(dim=1).flatten()
    targets = targets.flatten()
    
    for cls in range(num_classes):
        intersection[cls] = torch.sum((preds == cls) & (targets == cls))
        union[cls] = torch.sum((preds == cls) | (targets == cls))
    
    iou = intersection / (union + 1e-6)
    return torch.mean(iou).item()

def calculate_pixel_accuracy(preds, targets):
    preds = preds.argmax(dim=1).flatten()
    targets = targets.flatten()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

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
                tt = torch.jit.trace(supervised_model, inputs).to('cuda:0')
                current_date = datetime.now().strftime('%m%d')
                model_path = f'{target}/unsuper/unsupervised_model_{current_date}_{(step + 1)}.pth'
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
    try:
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
                    case "skirt" | "rend":
                        s_l1_loss = l1_criterion(s_outputs.squeeze(1), masks)
                        scores = torch.cat([1 - s_outputs, s_outputs], dim=1)
                        s_focal_loss = focal_criterion(scores, masks)
                        s_loss = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)
                    case "head" | "mold":
                        scores = torch.cat([1 - s_outputs, s_outputs], dim=1)
                        s_loss = focal_criterion(scores, masks)
                
                s_loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                # log loss
                s_losses_m.update(s_loss.item())
                if target == 'skirt' or target == 'rend':
                    s_l1_losses_m.update(s_l1_loss.item())
                    s_focal_losses_m.update(s_focal_loss.item())
                batch_time_m.update(time.time() - end)

                if (step + 1) % log_interval == 0 or step == 0:
                    _logger.info('TRAIN [{:>4d}/{}] '
                                's_Loss: {s_loss.val:>6.4f} ({s_loss.avg:>6.4f}) '
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
                    tt.save(os.path.join(savedir, f'{target}/super/supervised_model_{(step+1)}.pth'))

                # scheduler
                if scheduler:
                    scheduler.step()

                end = time.time()

                step += 1

                if step == num_training_steps:
                    train_mode = False
                    break
    except Exception as e:
        _logger.error(f'Error loading batch: {e}')

def training_iter_supervise(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], resize: Tuple[int, int] = (224,224),
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
    valid_loss = 0.0
    for i in range(num_training_steps):
        end = time.time()
        s_losses_m.reset()

        supervised_model = supervised_model.to(device)
        
        # Use tqdm for the inner loop as well
        for inputs, masks in tqdm(trainloader, desc=f"Epoch {i+1}", leave=False):
            # batch
            inputs, masks = inputs.to(device), masks.to(device)

            # predict 1
            s_outputs = supervised_model(inputs).to(device)
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

            valid_loss += s_loss.item()

            # scheduler
            if scheduler:
                scheduler.step(step)

        data_time_m.update(time.time() - end)

        _logger.info('\nTRAIN [{:>4d}/{}] '
                        's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            i + 1, num_training_steps,
            s_loss=s_losses_m,
            lr=optimizer.param_groups[0]['lr'],
            batch_time=batch_time_m,
            rate=inputs.size(0) / batch_time_m.val,
            rate_avg=inputs.size(0) / batch_time_m.avg,
            data_time=data_time_m))
                
        step += 1
        
        # memory map test
        write_to_memory_mapped_file(i+1, s_losses_m.avg, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

        device_after = 'cuda:0'
        inputs, masks = inputs.to(device_after), masks.to(device_after)
        supervised_model = supervised_model.to(device_after)

        current_date = datetime.now().strftime('%m%d')
        #current_date = '1111'
        os.makedirs(os.path.join(savedir, f'{target}/{current_date}'), exist_ok=True)
        tt = torch.jit.trace(supervised_model, example_inputs=inputs)
        model_path = f'{target}/{current_date}/supervised_model_{current_date}_{(i + 1)}.pth'
        tt.save(os.path.join(savedir, model_path))

        onnx_path = os.path.splitext(os.path.join(savedir, model_path))[0] + '.onnx'

        input_image = torch.randn(1, 3, resize[0], resize[1]).to(device_after)

        torch.onnx.export(supervised_model, input_image, onnx_path, 
                  export_params=True,
                  opset_version=12,
                  verbose=False,
                  do_constant_folding=True if device == 'cpu' else False, 
				  input_names = ['input'],
                  output_names = ['output'])

def training_output_ensemble(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], resize: Tuple[int, int] = (224,224),
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, target: str = None,
             device: str = 'cpu'):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    s_losses_m = AverageMeter()
    s_l1_losses_m = AverageMeter()
    s_focal_losses_m = AverageMeter()

    # set train mode
    supervised_model1 = supervised_model[0]
    supervised_model2 = supervised_model[1]
    supervised_model3 = supervised_model[2]

    supervised_model1.train()
    supervised_model2.train()
    supervised_model3.train()
    

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set optimizer
    optimizer.zero_grad()

    # training
    step = 0
    valid_loss = 0.0
    for i in range(num_training_steps):
        end = time.time()
        s_losses_m.reset()

        supervised_model1 = supervised_model1.to(device)
        supervised_model2 = supervised_model2.to(device)
        supervised_model3 = supervised_model3.to(device)
        
        # Use tqdm for the inner loop as well
        for inh, rgh, rgv, masks in tqdm(trainloader, desc=f"Epoch {i+1}", leave=False):
            # batch
            inh, rgh, rgv, masks = inh.to(device), rgh.to(device), rgv.to(device), masks.to(device)

            # predict 1
            s_output1 = supervised_model1(inh).to(device)
            s_output2 = supervised_model2(rgh).to(device)
            s_output3 = supervised_model3(rgv).to(device)

            total_output = torch.mean(torch.stack([s_output1, s_output2, s_output3]), dim=0)

            total_output = F.sigmoid(total_output)
            s_l1_loss = l1_criterion(total_output.squeeze(1), masks)

            scores = torch.cat([1 - total_output, total_output], dim=1) 
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

            valid_loss += s_loss.item()

            # scheduler
            if scheduler:
                scheduler.step(step)

            #img_count += 1
        
        data_time_m.update(time.time() - end)

        _logger.info('TRAIN [{:>4d}/{}] '
                        's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            i + 1, num_training_steps,
            s_loss=s_losses_m,
            lr=optimizer.param_groups[0]['lr'],
            batch_time=batch_time_m,
            rate=inh.size(0) / batch_time_m.val,
            rate_avg=inh.size(0) / batch_time_m.avg,
            data_time=data_time_m))
                
        step += 1
        
        # memory map test
        write_to_memory_mapped_file(i+1, s_losses_m.avg, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

        device_after = 'cuda:0'
        inh, rgh, rgv, masks = inh.to(device_after), rgh.to(device_after), rgv.to(device_after), masks.to(device_after)
        supervised_model1 = supervised_model1.to(device_after)
        supervised_model2 = supervised_model2.to(device_after)
        supervised_model3 = supervised_model3.to(device_after)

        current_date = datetime.now().strftime('%m%d')
        dir_path = savedir + f'{target}/output_ensemble/inh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model1, example_inputs=inh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/output_ensemble/rgh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model2, example_inputs=rgh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/output_ensemble/rgv'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model3, example_inputs=rgv)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

def training_loss_ensemble(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], resize: Tuple[int, int] = (224,224),
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, target: str = None,
             device: str = 'cpu'):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    s_losses_m = AverageMeter()
    s_l1_losses_m = AverageMeter()
    s_focal_losses_m = AverageMeter()

    # set train mode
    supervised_model1 = supervised_model[0]
    supervised_model2 = supervised_model[1]
    supervised_model3 = supervised_model[2]

    supervised_model1.train()
    supervised_model2.train()
    supervised_model3.train()
    

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set optimizer
    optimizer.zero_grad()

    # training
    step = 0
    valid_loss = 0.0
    for i in range(num_training_steps):
        end = time.time()
        s_losses_m.reset()

        supervised_model1 = supervised_model1.to(device)
        supervised_model2 = supervised_model2.to(device)
        supervised_model3 = supervised_model3.to(device)
        
        # Use tqdm for the inner loop as well
        for inh, rgh, rgv, masks in tqdm(trainloader, desc=f"Epoch {i+1}", leave=False):
            # batch
            inh, rgh, rgv, masks = inh.to(device), rgh.to(device), rgv.to(device), masks.to(device)

            # predict 1
            s_output1 = supervised_model1(inh).to(device)
            s_output2 = supervised_model2(rgh).to(device)
            s_output3 = supervised_model3(rgv).to(device)

            s_output1 = F.sigmoid(s_output1)
            s_l1_loss = l1_criterion(s_output1.squeeze(1), masks)
            scores = torch.cat([1 - s_output1, s_output1], dim=1) 
            s_focal_loss = focal_criterion(scores, masks)
            s_loss1 = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

            s_output2 = F.sigmoid(s_output2)
            s_l1_loss = l1_criterion(s_output2.squeeze(1), masks)
            scores = torch.cat([1 - s_output2, s_output2], dim=1) 
            s_focal_loss = focal_criterion(scores, masks)
            s_loss2 = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

            s_output3 = F.sigmoid(s_output3)
            s_l1_loss = l1_criterion(s_output3.squeeze(1), masks)
            scores = torch.cat([1 - s_output3, s_output3], dim=1) 
            s_focal_loss = focal_criterion(scores, masks)
            s_loss3 = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

            s_loss = s_loss1 + s_loss2 + s_loss3

            s_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # log loss
            s_losses_m.update(s_loss.item())
            s_l1_losses_m.update(s_l1_loss.item())
            s_focal_losses_m.update(s_focal_loss.item())
            batch_time_m.update(time.time() - end)

            valid_loss += s_loss.item()

            # scheduler
            if scheduler:
                scheduler.step(step)

            #img_count += 1
        
        data_time_m.update(time.time() - end)

        _logger.info('TRAIN [{:>4d}/{}] '
                        's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            i + 1, num_training_steps,
            s_loss=s_losses_m,
            lr=optimizer.param_groups[0]['lr'],
            batch_time=batch_time_m,
            rate=inh.size(0) / batch_time_m.val,
            rate_avg=inh.size(0) / batch_time_m.avg,
            data_time=data_time_m))
                
        step += 1
        
        # memory map test
        write_to_memory_mapped_file(i+1, s_losses_m.avg, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

        device_after = 'cuda:0'
        inh, rgh, rgv, masks = inh.to(device_after), rgh.to(device_after), rgv.to(device_after), masks.to(device_after)
        supervised_model1 = supervised_model1.to(device_after)
        supervised_model2 = supervised_model2.to(device_after)
        supervised_model3 = supervised_model3.to(device_after)

        current_date = datetime.now().strftime('%m%d')
        dir_path = savedir + f'{target}/loss_ensemble/inh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model1, example_inputs=inh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 501)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/loss_ensemble/rgh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model2, example_inputs=rgh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 501)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/loss_ensemble/rgv'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model3, example_inputs=rgv)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 501)}.pth')
        tt.save(model_path)

def training_4ch_supervise(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], resize: Tuple[int, int] = (224,224),
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
    valid_loss = 0.0
    for i in range(num_training_steps):
        end = time.time()
        s_losses_m.reset()
        #img_count = 1

        supervised_model = supervised_model.to(device)
        
        # Use tqdm for the inner loop as well
        for inputs, masks in tqdm(trainloader, desc=f"Epoch {i+1}", leave=False):
            # batch
            inputs, masks = inputs.to(device), masks.to(device)

            # predict 1
            s_outputs = supervised_model(inputs).to(device)
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

            valid_loss += s_loss.item()

            # scheduler
            if scheduler:
                scheduler.step(step)

            #img_count += 1
        
        data_time_m.update(time.time() - end)

        _logger.info('TRAIN [{:>4d}/{}] '
                        's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            i + 1, num_training_steps,
            s_loss=s_losses_m,
            lr=optimizer.param_groups[0]['lr'],
            batch_time=batch_time_m,
            rate=inputs.size(0) / batch_time_m.val,
            rate_avg=inputs.size(0) / batch_time_m.avg,
            data_time=data_time_m))
                
        step += 1
        
        # memory map test
        write_to_memory_mapped_file(i+1, s_losses_m.avg, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

        device_after = 'cuda:0'
        inputs, masks = inputs.to(device_after), masks.to(device_after)
        supervised_model = supervised_model.to(device_after)

        current_date = datetime.now().strftime('%m%d')
        #current_date = '1111'
        os.makedirs(os.path.join(savedir, f'{target}/raw_4ch'), exist_ok=True)
        tt = torch.jit.trace(supervised_model, example_inputs=inputs)
        model_path = f'{target}/raw_4ch/supervised_model_{(i + 1)}.pth'
        tt.save(os.path.join(savedir, model_path))

        # onnx_path = os.path.splitext(os.path.join(savedir, model_path))[0] + '.onnx'

        # input_image = torch.randn(1, 3, resize[0], resize[1]).to(device_after)

        # torch.onnx.export(supervised_model, input_image, onnx_path, 
        #           export_params=True,
        #           opset_version=12,
        #           verbose=False,
        #           do_constant_folding=True if device == 'cpu' else False, 
		# 		  input_names = ['input'],
        #           output_names = ['output'])

def training_output_a_cutmix_ensemble(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], resize: Tuple[int, int] = (224,224),
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, target: str = None,
             device: str = 'cpu'):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    s_losses_m = AverageMeter()
    s_l1_losses_m = AverageMeter()
    s_focal_losses_m = AverageMeter()

    # set train mode
    supervised_model1 = supervised_model[0]
    supervised_model2 = supervised_model[1]
    supervised_model3 = supervised_model[2]

    supervised_model1.train()
    supervised_model2.train()
    supervised_model3.train()
    
    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set optimizer
    optimizer.zero_grad()

    # training
    step = 0
    valid_loss = 0.0
    for i in range(num_training_steps):
        end = time.time()
        s_losses_m.reset()

        supervised_model1 = supervised_model1.to(device)
        supervised_model2 = supervised_model2.to(device)
        supervised_model3 = supervised_model3.to(device)
        
        # Use tqdm for the inner loop as well
        for inh, rgh, rgv, masks in tqdm(trainloader, desc=f"Epoch {i+1}", leave=False):
            rand_indices = np.random.randint(0, len(trainloader), size=inh.shape[0])
            gridd = np.random.randint(32,128)
            # batch
            inh, rgh, rgv, masks = inh.to(device), rgh.to(device), rgv.to(device), masks.to(device)
            
            # predict 1
            s_output1 = supervised_model1(inh).to(device)
            s_output2 = supervised_model2(rgh).to(device)
            s_output3 = supervised_model3(rgv).to(device)

            total_output = torch.mean(torch.stack([s_output1, s_output2, s_output3]), dim=0)

            total_output = F.sigmoid(total_output)
            s_l1_loss = l1_criterion(total_output.squeeze(1), masks)

            scores = torch.cat([1 - total_output, total_output], dim=1) 
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

            valid_loss += s_loss.item()

            # scheduler
            if scheduler:
                scheduler.step(step)

            #img_count += 1
        
        data_time_m.update(time.time() - end)

        _logger.info('TRAIN [{:>4d}/{}] '
                        's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            i + 1, num_training_steps,
            s_loss=s_losses_m,
            lr=optimizer.param_groups[0]['lr'],
            batch_time=batch_time_m,
            rate=inh.size(0) / batch_time_m.val,
            rate_avg=inh.size(0) / batch_time_m.avg,
            data_time=data_time_m))
                
        step += 1
        
        # memory map test
        write_to_memory_mapped_file(i+1, s_losses_m.avg, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

        device_after = 'cuda:0'
        inh, rgh, rgv, masks = inh.to(device_after), rgh.to(device_after), rgv.to(device_after), masks.to(device_after)
        supervised_model1 = supervised_model1.to(device_after)
        supervised_model2 = supervised_model2.to(device_after)
        supervised_model3 = supervised_model3.to(device_after)

        current_date = datetime.now().strftime('%m%d')
        dir_path = savedir + f'{target}/output_a_cutmix_ensemble/inh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model1, example_inputs=inh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/output_a_cutmix_ensemble/rgh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model2, example_inputs=rgh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/output_a_cutmix_ensemble/rgv'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model3, example_inputs=rgv)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

def training_loss_a_cutmix_ensemble(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], resize: Tuple[int, int] = (224,224),
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, target: str = None,
             device: str = 'cpu'):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    s_losses_m = AverageMeter()

    # set train mode
    supervised_model1 = supervised_model[0]
    supervised_model2 = supervised_model[1]
    supervised_model3 = supervised_model[2]

    supervised_model1.train()
    supervised_model2.train()
    supervised_model3.train()

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set optimizer
    optimizer.zero_grad()

    # training
    step = 0
    valid_loss = 0.0
    for i in range(num_training_steps):
        end = time.time()
        s_losses_m.reset()

        supervised_model1 = supervised_model1.to(device)
        supervised_model2 = supervised_model2.to(device)
        supervised_model3 = supervised_model3.to(device)
        
        # Use tqdm for the inner loop as well
        for inh, rgh, rgv, masks in tqdm(trainloader, desc=f"Epoch {i+1}", leave=False):
            rand_indices = np.random.randint(0, len(trainloader), size=inh.shape[0])
            gridd = np.random.randint(32,128)
            # batch
            inh, rgh, rgv, masks = inh.to(device), rgh.to(device), rgv.to(device), masks.to(device)
            
            # predict 1
            s_output1 = supervised_model1(inh).to(device)
            s_output2 = supervised_model2(rgh).to(device)
            s_output3 = supervised_model3(rgv).to(device)

            # predict 1
            s_output1 = F.sigmoid(s_output1)
            s_l1_loss = l1_criterion(s_output1.squeeze(1), masks)
            scores = torch.cat([1 - s_output1, s_output1], dim=1) 
            s_focal_loss = focal_criterion(scores, masks)
            s_loss1 = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

            s_output2 = F.sigmoid(s_output2)
            s_l1_loss = l1_criterion(s_output2.squeeze(1), masks)
            scores = torch.cat([1 - s_output2, s_output2], dim=1) 
            s_focal_loss = focal_criterion(scores, masks)
            s_loss2 = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

            s_output3 = F.sigmoid(s_output3)
            s_l1_loss = l1_criterion(s_output3.squeeze(1), masks)
            scores = torch.cat([1 - s_output3, s_output3], dim=1) 
            s_focal_loss = focal_criterion(scores, masks)
            s_loss3 = (l1_weight * s_l1_loss) + (focal_weight * s_focal_loss)

            s_loss = s_loss1 + s_loss2 + s_loss3

            s_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # log loss
            s_losses_m.update(s_loss.item())
            batch_time_m.update(time.time() - end)

            valid_loss += s_loss.item()

            # scheduler
            if scheduler:
                scheduler.step(step)

            #img_count += 1
        
        data_time_m.update(time.time() - end)

        _logger.info('TRAIN [{:>4d}/{}] '
                        's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            i + 1, num_training_steps,
            s_loss=s_losses_m,
            lr=optimizer.param_groups[0]['lr'],
            batch_time=batch_time_m,
            rate=inh.size(0) / batch_time_m.val,
            rate_avg=inh.size(0) / batch_time_m.avg,
            data_time=data_time_m))
                
        step += 1
        
        # memory map test
        write_to_memory_mapped_file(i+1, s_losses_m.avg, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

        device_after = 'cuda:0'
        inh, rgh, rgv, masks = inh.to(device_after), rgh.to(device_after), rgv.to(device_after), masks.to(device_after)
        supervised_model1 = supervised_model1.to(device_after)
        supervised_model2 = supervised_model2.to(device_after)
        supervised_model3 = supervised_model3.to(device_after)

        dir_path = savedir + f'{target}/loss_a_cutmix_ensemble/inh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model1, example_inputs=inh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/loss_a_cutmix_ensemble/rgh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model2, example_inputs=rgh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/loss_a_cutmix_ensemble/rgv'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model3, example_inputs=rgv)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)


def training_dynamic_loss_a_cutmix_ensemble(supervised_model, trainloader, validloader, criterion, optimizer,
             scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], resize: Tuple[int, int] = (224,224),
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, target: str = None,
             device: str = 'cpu'):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    s_losses_m = AverageMeter()
    miou_meter = AverageMeter()  # mIoU 기록
    pa_meter = AverageMeter()    # PA 기록

    # set train mode
    supervised_model1 = supervised_model[0]
    supervised_model2 = supervised_model[1]
    supervised_model3 = supervised_model[2]

    supervised_model1.train()
    supervised_model2.train()
    supervised_model3.train()

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set optimizer
    optimizer.zero_grad()

    # training
    step = 0
    valid_loss = 0.0
    for i in range(num_training_steps):
        end = time.time()
        s_losses_m.reset()

        supervised_model1 = supervised_model1.to(device)
        supervised_model2 = supervised_model2.to(device)
        supervised_model3 = supervised_model3.to(device)
        
        # Use tqdm for the inner loop as well
        for inh, rgh, rgv, masks in tqdm(trainloader, desc=f"Epoch {i+1}", leave=False):
            rand_indices = np.random.randint(0, len(trainloader), size=inh.shape[0])
            gridd = np.random.randint(32, 128)
            # Batch 데이터 전처리
            inh, rgh, rgv, masks = inh.to(device), rgh.to(device), rgv.to(device), masks.to(device)

            # 예측 및 손실 계산
            outputs = []
            losses = []
            for model, input_tensor in zip([supervised_model1, supervised_model2, supervised_model3], [inh, rgh, rgv]):
                output = model(input_tensor).to(device)
                output = F.sigmoid(output)
                l1_loss = l1_criterion(output.squeeze(1), masks)
                scores = torch.cat([1 - output, output], dim=1)
                focal_loss = focal_criterion(scores, masks)
                total_loss = (l1_weight * l1_loss) + (focal_weight * focal_loss)
                outputs.append(output)
                losses.append(total_loss)

            # 손실 차이에 따른 가중치 조정
            total_losses = torch.tensor(losses, device=device)
            normalized_losses = total_losses / total_losses.sum()  # 손실 비율 계산
            dynamic_weights = 1.0 / (normalized_losses + 1e-6)  # 손실이 클수록 높은 가중치 부여
            dynamic_weights = dynamic_weights / dynamic_weights.sum()  # 가중치 정규화

            # 최종 손실 계산 (동적 가중치 적용)
            weighted_loss = sum(w * loss for w, loss in zip(dynamic_weights, losses))

            combined_output = torch.mean(torch.stack(outputs), dim=0)
            # mIoU와 PA 계산
            miou = calculate_mIoU(combined_output, masks)
            pa = calculate_pixel_accuracy(combined_output, masks)

            # 역전파
            weighted_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 로그 기록
            s_losses_m.update(weighted_loss.item())
            miou_meter.update(miou)
            pa_meter.update(pa)
            batch_time_m.update(time.time() - end)
            valid_loss += weighted_loss.item()


            # scheduler
            if scheduler:
                scheduler.step(step)

            #img_count += 1
        
        data_time_m.update(time.time() - end)

        _logger.info('TRAIN [{:>4d}/{}] '
                        's_Loss: {s_loss.val:>.3e} ({s_loss.avg:>.3e}) '
                        'LR: {lr:.3e} '
                        'mIoU: {miou:.4f} '
                        'PA: {pa:.4f} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            i + 1, num_training_steps,
            s_loss=s_losses_m,
            lr=optimizer.param_groups[0]['lr'],
            miou=miou_meter.avg, 
            pa=pa_meter.avg, 
            batch_time=batch_time_m,
            rate=inh.size(0) / batch_time_m.val,
            rate_avg=inh.size(0) / batch_time_m.avg,
            data_time=data_time_m))
                
        step += 1
        
        # memory map test
        write_to_memory_mapped_file(i+1, s_losses_m.avg, data_time_m.val, batch_time_m.val, optimizer.param_groups[0]['lr'])

        device_after = 'cuda:0'
        inh, rgh, rgv, masks = inh.to(device_after), rgh.to(device_after), rgv.to(device_after), masks.to(device_after)
        supervised_model1 = supervised_model1.to(device_after)
        supervised_model2 = supervised_model2.to(device_after)
        supervised_model3 = supervised_model3.to(device_after)

        dir_path = savedir + f'{target}/dynamic_loss_a_cutmix_ensemble/inh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model1, example_inputs=inh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/dynamic_loss_a_cutmix_ensemble/rgh'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model2, example_inputs=rgh)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)

        dir_path = savedir + f'{target}/dynamic_loss_a_cutmix_ensemble/rgv'
        os.makedirs(dir_path, exist_ok=True)
        tt = torch.jit.trace(supervised_model3, example_inputs=rgv)
        model_path = os.path.join(dir_path, f'supervised_model_{(i + 1)}.pth')
        tt.save(model_path)