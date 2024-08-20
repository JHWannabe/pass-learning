import os
import numpy as nppirun
from torch import nn
import torch
from models import Supervised
from RD4AD import resnet

import onnx
import onnxruntime

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 랜덤 텐서 생성
# random_tensor = torch.randn(batch_size, channels, height, width)
input_image_0 = torch.randn(1, 3, 1280, 1280).to(device)
model_path = './saved_model/head/supervised_model_0813_12.pth'
# model_path = './saved_model/rend/supervised_model_0727_29.pth'
onnx_path = os.path.splitext(model_path)[0] + '.onnx'
# load RD4AD network
RD4AD_encoder, RD4AD_bn = resnet.resnet18(pretrained=True)
RD4AD_encoder = RD4AD_encoder.to(device)
RD4AD_encoder.eval()
supervised_model = Supervised(feature_extractor = RD4AD_encoder).to(device)

# .pt 일 때
# state_dict = torch.load(model_path)
# supervised_model.load_state_dict(state_dict,strict=False)
# .pth일 때
supervised_model = torch.jit.load(model_path)

supervised_model.to(device)
supervised_model.eval()

torch.onnx.export(supervised_model, input_image_0, onnx_path, 
                  export_params=True, 
                  opset_version=12,
                  verbose=False,
                  do_constant_folding=True if device == 'cpu' else False, 
				  input_names = ['input'],
                  output_names = ['output'])

onnx_model = onnx.load(onnx_path)
assert onnx.checker.check_model(onnx_model) == None

providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = onnxruntime.SessionOptions()
ort_session = onnxruntime.InferenceSession(onnx_path)

# pyinstaller D:\Pass_Learning_Python\main_head_train_unsupervise.py