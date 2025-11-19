# Installation

This repository is built in PyTorch 11.8.1 and tested on Ubuntu 16.04 environment (Python3.10, CUDA12.3, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/Computer-Vision-Lab-LinhDang/underwater-image-restoration.git
cd underwater-image-restoration
```

2. Make conda environment
```
conda create -n underwater python=3.10
conda activate underwater
```

3. Install dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
pip install pytorch_wavelets
pip install PyWavelets
pip install calflops
pip install transformer
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```

### Datasets

Datasets 