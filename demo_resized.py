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
import torchvision.transforms as transforms
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np
import time 

parser = argparse.ArgumentParser(description='Test Restormer on your own images')
parser.add_argument('--input_dir', default='./demo/degraded/', type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./demo/restored/', type=str, help='Directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=["UnderWater"])
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(task, parameters):
    if task == 'UnderWater':
        weights = os.path.join('Under_Water', 'pretrained_models', 'model.pth')
    return weights, parameters

task    = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir, task)

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
    files = natsorted(files)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 32,'ffn_expansion_factor': 2, 'stages': 2, 'LayerNorm_type': 'WithBias'}
weights, parameters = get_weights_and_parameters(task, parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'underwater_arch.py'))
model = load_arch['UWNet'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")

fps_list = []
inference_times = []

with torch.no_grad():
    length = len(files)
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        import time

        img = load_img(file_)



        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        input_ = transforms.Resize((256, 256))(input_)
                ## Testing on the original resolution image
                
        start = time.time()
        restored = model(input_)
        elapsed = time.time() - start
        fps = 1 / elapsed
        fps_list.append(fps)
        inference_times.append(elapsed)
        
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        # stx()
        
        save_img((os.path.join(out_dir, f+'.png')), restored)
    print(f"\nAverage FPS: {np.mean(fps_list):.2f}")
    print(f"Average Inference Time: {np.mean(inference_times):.4f} seconds")
    print(f"\nRestored images are saved at {out_dir}")
