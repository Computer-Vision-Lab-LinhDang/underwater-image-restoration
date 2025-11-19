import ptwt
import torch
from pytorch_wavelets import DWTForward, DWTInverse

xfm = DWTForward(J=1, mode='zero', wave='haar').to('cuda')   # DWT DWT có tham số không
ifm = DWTInverse(mode='zero', wave='haar').to('cuda')
import time

x = torch.rand((1, 64, 128, 128)).to('cuda')

start = time.time()
with torch.no_grad():  # Tắt gradient cho inference
    coeffs = xfm(x)
end = time.time()

print(end-start)

with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    coeffs = xfm(x)
    end = time.time()
print("DWTForward time:", end - start)
# torch.backends.cudnn.benchmark = False  # Tăng tốc conv ops
start = time.time()
with torch.no_grad():  # Tắt gradient cho inference
    coeffs = xfm(x)
end = time.time()
print("DWTForward time:", end - start)

with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    coeffs = xfm(x)
    end = time.time()
print("DWTForward time:", end - start)

with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    coeffs = xfm(x)
    end = time.time()
print("DWTForward time:", end - start)

with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    coeffs = xfm(x)
    end = time.time()
print("DWTForward time:", end - start)

with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    coeffs = xfm(x)
    end = time.time()
print("DWTForward time:", end - start)

with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    coeffs = xfm(x)
    end = time.time()
print("DWTForward time:", end - start)

with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    c = ifm(coeffs)
    end = time.time()
print("DWTForward time:", end - start)


with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    c = ifm(coeffs)
    end = time.time()
print("DWTForward time:", end - start)


with torch.no_grad():  # Tắt gradient cho inference
    start = time.time()
    c = ifm(coeffs)
    end = time.time()
print("DWTForward time:", end - start)