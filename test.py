import numpy as np
import torch
import ctypes
from ctypes import *
from random import random

import math
import io
import matplotlib.pyplot as plt
import os
from pandas import DataFrame
import seaborn as sns

from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_msssim import ms_ssim
from compressai.zoo import bmshj2018_factorized

# # create example tensors on GPU
# input_tensor_gpu = torch.tensor([random() for i in range(1024 * 1024)], dtype=torch.float32).cuda()

# compression and decompression round trip
def gpulz_compress():
    dll = ctypes.CDLL('./gpulz.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.runCompression
    func.argtypes = [POINTER(c_uint32), c_uint32]
    return func

def run_gpulz(input_tensor, file_size):
    # get input GPU pointer
    input_gpu_ptr = input_tensor.data_ptr()
    input_gpu_ptr = cast(input_gpu_ptr, ctypes.POINTER(c_uint32))

    file_size_c = c_uint32(file_size)

    gpulz_comp = gpulz_compress()
    gpulz_comp(input_gpu_ptr, file_size_c)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = bmshj2018_factorized(quality=1, pretrained=True).eval().to(device)

    img = Image.open('/home/bozhan/result/microscopy/2023_09_20/STO_GE_2.png').convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    y = net.g_a(x)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("compression"):
            y_quantization = y.to(torch.int32)

            file_size = y_quantization.shape[1] * y_quantization.shape[2] * y_quantization.shape[3] * 4
            run_gpulz(y_quantization, file_size)
    
    prof.export_chrome_trace("gpulz.json")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("compression"):
            y_strings = net.entropy_bottleneck.compress(y)
    
    prof.export_chrome_trace("compressai.json")