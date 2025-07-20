## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##--------------------------------------------------------------
##------- Demo file to test Restormer on your own images---------
## Example usage on directory containing several images:   python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
## Example usage on a image directly: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'
## Example usage with tile option on a large image: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/' --tile 720 --tile_overlap 32
##--------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count

import os
from runpy import run_path
from natsort import natsorted
from glob import glob
import argparse
import numpy as np
import time 


def get_weights_and_parameters(task, parameters):
    if task == 'UnderWater':
        weights = os.path.join('Under_Water', 'pretrained_models', 'model.pth')
    return weights, parameters

# Get model weights and parameters
task = 'UnderWater'
parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 32, 'ffn_expansion_factor': 2, 'stages': 2, 'LayerNorm_type': 'WithBias'}
weights, parameters = get_weights_and_parameters(task, parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'underwater_arch.py'))
model = load_arch['UWNet'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()
model = model.cpu()
print(f"\n ==> Running {task} with weights {weights}\n ")
inputs = torch.randn((1, 3, 256, 256))
with torch.cuda.device(0):  # hoặc bỏ nếu không dùng GPU
    flops = FlopCountAnalysis(model, inputs)
    params = parameter_count(model)

# In ra kết quả
print(f"FLOPs: {flops.total() / 1e9:.2f} GFlops")
print(f"Params: {params[''] / 1e6:.2f} M")