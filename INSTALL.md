# Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/swz30/Restormer.git
cd Restormer
```

2. Make conda environment
```
conda create -n pytorch181 python=3.7
conda activate pytorch181
```

3. Install dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
pip install pytorch_wavelets
pip install PyWavelets
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```

### Download datasets from Google Drive

