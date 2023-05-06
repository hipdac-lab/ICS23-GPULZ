# GPULZ: Optimizing LZSS Lossless Compression for Multi-byte Data on Modern GPUs

Currently, GPULZ performs compression and decompression together, but we plan to provide options for performing compression and decompression separately in the future.

(C) 2023 by Indiana University and Argonne National Laboratory.

- Developers: Boyuan Zhang
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
#define INPUT_TYPE uint32_t // define input type, since c++ doesn't support runtime data type defination
```

## Download Data
Please use ```get_sample_data.sh``` to download the sample data.

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
./gpulz -i QCLOUDf48.bin.f32.errctrl.1e-3
```

Finally, you can observe the output including compression ratio and compression/decompression end-to-end throughputs.
```
compression ratio: 11.5429
compression e2e throughput: 20.9908 GB/s
decompression e2e throughput: 34.4679 GB/s
```

To obtain more accurate timing for the compression kernel, please use ```nsys``` before the execution command, like
```
nsys profile --stats=true ./gpulz -i QCLOUDf48.bin.f32.errctrl.1e-3
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
This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants OAC-2003709/2303064, OAC-2104023/2247080, and OAC-2312673.
