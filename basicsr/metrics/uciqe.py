import cv2
import numpy as np
import torch
import kornia.color as color
import math
from basicsr.metrics.metric_util import reorder_image


def calculate_uciqe(image):
    # RGB转为HSV
    if len(image.shape) == 4:
        image = image.squeeze(0)
    hsv = color.rgb_to_hsv(image)  
    H, S, V = torch.chunk(hsv, 3)

    # 色度的标准差
    delta = torch.std(H) / (2 * math.pi)
    
    # 饱和度的平均值
    mu = torch.mean(S)  
    
    # 求亮度对比值
    n, m = V.shape[1], V.shape[2]
    number = math.floor(n * m / 100)
    v = V.flatten()
    v, _ = v.sort()
    bottom = torch.sum(v[:number]) / number
    v = -v
    v, _ = v.sort()
    v = -v
    top = torch.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe