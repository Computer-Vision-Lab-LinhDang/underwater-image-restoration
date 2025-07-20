"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
import torch
## local libs
from basicsr.metrics.uiqm_utils import getUIQM
from basicsr.metrics.metric_util import reorder_image



def calculate_uiqm(img,
                   input_order='HWC'):
    if type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        img = img.detach().cpu().numpy().transpose(1,2,0)
    img = reorder_image(img, input_order=input_order)
    uiqm = getUIQM(img)
    return uiqm
