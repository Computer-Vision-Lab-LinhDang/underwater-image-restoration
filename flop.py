from calflops import calculate_flops
from models.waternet import WaterNet
from models.raune_net import RauneNet

import torch

model = RauneNet(3, 3, 30, 2)
flops, macs_netG_HR, params_netG_HR = calculate_flops(model=model,input_shape=(1, 3, 256, 256), print_results=False)
print(flops, macs_netG_HR, params_netG_HR)
print("FLOPS, INFER TIME, PARAMS done!")