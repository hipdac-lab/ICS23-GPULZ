import numpy as np
import torch
import ctypes
from ctypes import *
from random import random
from math import floor

# compression and decompression round trip
def compressFunc():
    dll = ctypes.CDLL('./gpulz.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cCompress
    func.argtypes = [POINTER(c_uint32), POINTER(c_uint8), c_uint32, c_int, c_void_p]
    return func

def decompressFunc():
    dll = ctypes.CDLL('./gpulz.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cDecompress
    func.argtypes = [POINTER(c_uint8), POINTER(c_uint32), c_uint32, c_int, c_void_p]
    return func

def pCompress(inputTensor, outputTensor, fileSize, gpuIdx, stream):
    # get input GPU pointer
    input_gpu_ptr = inputTensor.data_ptr()
    input_gpu_ptr = cast(input_gpu_ptr, ctypes.POINTER(c_uint32))

    compressed_gpu_ptr = outputTensor.data_ptr()
    compressed_gpu_ptr = cast(compressed_gpu_ptr, ctypes.POINTER(c_uint8))

    fileSize_c = c_uint32(fileSize)

    stream_ptr = stream.cuda_stream
    stream_ptr = cast(stream_ptr, ctypes.c_void_p)

    compressor = compressFunc()
    compressor(input_gpu_ptr, compressed_gpu_ptr, fileSize_c, c_int(gpuIdx), stream_ptr)

def pDecompress(inputTensor, outputTensor, fileSize, gpuIdx, stream):
    input_gpu_ptr = inputTensor.data_ptr()
    input_gpu_ptr = cast(input_gpu_ptr, ctypes.POINTER(c_uint8))

    output_gpu_ptr = outputTensor.data_ptr()
    output_gpu_ptr = cast(output_gpu_ptr, ctypes.POINTER(c_uint32))

    fileSize_c = c_uint32(fileSize)

    stream_ptr = stream.cuda_stream
    stream_ptr = cast(stream_ptr, ctypes.c_void_p)

    decompressor = decompressFunc()
    decompressor(input_gpu_ptr, output_gpu_ptr, fileSize_c, c_int(gpuIdx), stream_ptr)

if __name__ == '__main__':
    stream = torch.cuda.Stream()

    # create example tensors on GPU
    inputTensor_gpu = torch.tensor([2 for i in range(1024 * 1024)], dtype=torch.int32).cuda()
    compressedTensor_gpu = torch.tensor([0 for i in range(1024 * 1024)], dtype=torch.int32).cuda()
    output_tensor_gpu = torch.tensor([0 for i in range(1024 * 1024)], dtype=torch.int32).cuda()

    pCompress(inputTensor_gpu, compressedTensor_gpu, 1024 * 1024 * 4, 0, stream)

    pDecompress(compressedTensor_gpu, output_tensor_gpu, 1024 * 1024 * 4, 0, stream)

    are_equal = torch.equal(inputTensor_gpu, output_tensor_gpu)
    if are_equal:
        print('original and reconstructed are equal')
    else:
        print('original and reconstructed are NOT equal!!!!!')
