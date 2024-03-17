# GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on Modern GPUs

GPULZ is a highly efficient LZSS compression solution for multi-byte data on modern GPUs. It optimizes the pattern-matching approach for multi-byte symbols to reduce computational complexity and discover longer repeated patterns. This results in higher performance than state-of-the-art solutions. At present, GPULZ performs compression and decompression simultaneously; however, we plan to offer options for performing these tasks separately in future iterations.

(C) 2023 by Indiana University and Argonne National Laboratory.

- Developer: Boyuan Zhang
- Contributors (alphabetic): Dingwen Tao, Franck Cappello, Jiannan Tian, Sheng Di, Xiaodong Yu

## Recommended Environment
- Linux OS with NVIDIA GPUs
- GCC (>= 7.3.0)
- CUDA (>= 11.0)

## Compile
Please use the following command to compile GPULZ. You will get the executable ```gpulz```.
```
make -j
```

## Configuration
Please modify the following code in ```gpulz.cu``` for different configurations.
```
#define BLOCK_SIZE 2048     // in unit of byte, the size of one data block
#define WINDOW_SIZE 32      // in unit of datatype, maximum 255, the size of the sliding window, so as the maximum match length
#define INPUT_TYPE uint32_t // define input type as c++ doesn't support runtime data type definition
```

## Download Data
Please use ```get_sample_data.sh``` to download the sample data. If you want to download more sample data, please remove the commented lines from the script.

```
./get_sample_data.sh
```

## Run GPULZ
Please use the below command to run ```gpulz``` on a float32 data.
```
./gpulz -i [input data path]
```

For example,
```
./gpulz -i ./02_HURR_zyx_100x500x500=25000000/QCLOUDf48.bin.f32.errctrl.1e-3
```

Finally, you can observe the output including compression ratio and compression/decompression end-to-end throughputs.
```
compression ratio: 11.5429
compression e2e throughput: 20.9908 GB/s
decompression e2e throughput: 34.4679 GB/s
```

To obtain more accurate timing for the compression kernel, please use ```nsys``` before the execution command, like
```
nsys profile --stats=true ./gpulz -i ./02_HURR_zyx_100x500x500=25000000/QCLOUDf48.bin.f32.errctrl.1e-3
```

You will observe the time for each kernel, i.e., compressKernelI (compression kernel I), compressKernelIII (compression kernel III), cub::DeviceScanKernel (CUB prefix sum kernel), and decompressKernel (decompression kernel).

## Citing GPULZ
**ICS '23: GPULZ** ([local copy](ICS23-GPULZ.pdf), [via ACM](https://dl.acm.org/doi/10.1145/3577193.3593706), or [via arXiv](https://arxiv.org/abs/2304.07342v2))

```bibtex
@inproceedings{gpulz2023zhang,
      title = {GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on Modern GPUs},
     author = {Zhang, Boyuan and Tian, Jiannan and Di, Sheng and Yu, Xiaodong and Swany, Martin and Tao, Dingwen and Cappello, Franck},
       year = {2023},
       isbn = {979-8-4007-0056-9/23/06},
  publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
	url = {https://dl.acm.org/doi/10.1145/3577193.3593706},
        doi = {10.1145/3577193.3593706},
  booktitle = {2023 International Conference on Supercomputing},
   numpages = {12},
   keywords = {Lossless compression; LZSS; GPU; performance},
   location = {Orlando, FL, USA},
     series = {ICS '23}
}
```

## Acknowledgements
This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants OAC2003709, OAC-2104023, OAC-2303064, OAC-2247080, and OAC2312673.
