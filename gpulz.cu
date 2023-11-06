#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cmath>
#include <cub/cub.cuh>

#define BLOCK_SIZE 2048     // in unit of vector, the size of one data block
#define THREAD_SIZE 128     // in unit of vector, the size of the thread block
#define WINDOW_SIZE 32      // in unit of datatype, maximum 255, the size of the sliding window, so as the maximum match length
#define INPUT_TYPE uint32_t // define input type, since c++ doesn't support runtime data type defination
#define VECTOR_SIZE 16      // in unit of datatype (uint32_t), the size of the vector
// #define DEBUG

// each thread block handles VECTOR_SIZE * THREAD_SIZE = 2048 data points

__device__ bool vectorComparison(INPUT_TYPE *input, uint32_t vectorSize, uint32_t bufferPosition, uint32_t windowPosition)
{
    for (int tmpIdx = 0; tmpIdx < vectorSize; tmpIdx++)
    {
        if (input[bufferPosition + tmpIdx] != input[windowPosition + tmpIdx])
        {
            return false;
        }
    }
    return true;
}

// Define the compress match kernel functions
template <typename T>
__global__ void compressKernelI(T *input, uint32_t numOfBlocks, uint32_t *flagArrSizeGlobal, uint32_t *compressedDataSizeGlobal, uint8_t *tmpFlagArrGlobal, uint8_t *tmpCompressedDataGlobal, int minEncodeLength)
{
    // Block size in uint of datatype
    const uint32_t blockSize = BLOCK_SIZE;

    // Window size in uint of datatype
    const uint32_t threadSize = THREAD_SIZE;

    const uint32_t vectorSize = VECTOR_SIZE;

    // Allocate shared memory for the lookahead buffer information
    __shared__ uint8_t lengthBuffer[blockSize];
    __shared__ uint8_t offsetBuffer[blockSize];
    __shared__ uint8_t byteFlagArr[(blockSize / 8)];
    __shared__ uint32_t prefixBuffer[blockSize + 1];

    // initialize the start position, in unit of vector
    int startPosision = blockIdx.x * blockSize;

    int vectorIdx = 0;

    // find match for every data point
    for (int iteration = 0; iteration < (int)(blockSize / threadSize); iteration++)
    {
        // Initialize the lookahead buffer and the sliding window pointers
        // initialize the vectorIdx, in unit of vector
        vectorIdx = threadIdx.x + iteration * threadSize;
        int bufferStart = vectorIdx;
        int bufferPointer = bufferStart;
        int windowStart = bufferStart - int(WINDOW_SIZE) < 0 ? 0 : bufferStart - WINDOW_SIZE;
        int windowPointer = windowStart;

        uint8_t maxLen = 0;
        uint8_t maxOffset = 0;
        uint8_t len = 0;
        uint8_t offset = 0;

        while (windowPointer < bufferStart && bufferPointer < blockSize)
        {
            if (vectorComparison(input, vectorSize, (startPosision + bufferPointer) * vectorSize, (startPosision + windowPointer) * vectorSize))
            {
                if (offset == 0)
                {
                    offset = bufferPointer - windowPointer;
                }
                len++;
                bufferPointer++;
            }
            else
            {
                if (len > maxLen)
                {
                    maxLen = len;
                    maxOffset = offset;
                }
                len = 0;
                offset = 0;
                bufferPointer = bufferStart;
            }
            windowPointer++;
        }
        if (len > maxLen)
        {
            maxLen = len;
            maxOffset = offset;
        }

        lengthBuffer[vectorIdx] = maxLen;
        offsetBuffer[vectorIdx] = maxOffset;

        // initialize array as 0
        prefixBuffer[vectorIdx] = 0;
    }
    __syncthreads();

    // find encode information
    uint32_t flagCount = 0;

    if (threadIdx.x == 0)
    {
        uint8_t flagPosition = 0x01;
        uint8_t byteFlag = 0;

        int encodeIndex = 0;

        while (encodeIndex < blockSize)
        {
            // if length < minEncodeLength, no match is found
            if (lengthBuffer[encodeIndex] < minEncodeLength)
            {
                prefixBuffer[encodeIndex] = vectorSize * sizeof(T);
                encodeIndex++;
            }
            // if length > minEncodeLength, match is found
            else
            {
                prefixBuffer[encodeIndex] = 2;
                encodeIndex += lengthBuffer[encodeIndex];
                byteFlag |= flagPosition;
            }
            // store the flag if there are 8 bits already
            if (flagPosition == 0x80)
            {
                byteFlagArr[flagCount] = byteFlag;
                flagCount++;
                flagPosition = 0x01;
                byteFlag = 0;
                continue;
            }
            flagPosition <<= 1;
        }
        if (flagPosition != 0x01)
        {
            byteFlagArr[flagCount] = byteFlag;
            flagCount++;
        }
    }
    __syncthreads();

    // prefix summation, up-sweep
    int prefixSumOffset = 1;
    for (uint32_t d = blockSize >> 1; d > 0; d = d >> 1)
    {
        for (int iteration = 0; iteration < (int)(blockSize / threadSize); iteration++)
        {
            vectorIdx = threadIdx.x + iteration * threadSize;
            if (vectorIdx < d)
            {
                int ai = prefixSumOffset * (2 * vectorIdx + 1) - 1;
                int bi = prefixSumOffset * (2 * vectorIdx + 2) - 1;
                prefixBuffer[bi] += prefixBuffer[ai];
            }
            __syncthreads();
        }
        prefixSumOffset *= 2;
    }

    // clear the last element
    if (threadIdx.x == 0)
    {
        // printf("block size: %d flag array size: %d\n", prefixBuffer[blockSize - 1], flagCount);
        compressedDataSizeGlobal[blockIdx.x] = prefixBuffer[blockSize - 1];
        flagArrSizeGlobal[blockIdx.x] = flagCount;
        prefixBuffer[blockSize] = prefixBuffer[blockSize - 1];
        prefixBuffer[blockSize - 1] = 0;
    }
    __syncthreads();

    // prefix summation, down-sweep
    for (int d = 1; d < blockSize; d *= 2)
    {
        prefixSumOffset >>= 1;
        for (int iteration = 0; iteration < (int)(blockSize / threadSize);
             iteration++)
        {
            vectorIdx = threadIdx.x + iteration * threadSize;

            if (vectorIdx < d)
            {
                int ai = prefixSumOffset * (2 * vectorIdx + 1) - 1;
                int bi = prefixSumOffset * (2 * vectorIdx + 2) - 1;

                uint32_t t = prefixBuffer[ai];
                prefixBuffer[ai] = prefixBuffer[bi];
                prefixBuffer[bi] += t;
            }
            __syncthreads();
        }
    }

    // encoding phase one
    int blockOffset = blockSize * blockIdx.x;

    for (int iteration = 0; iteration < (int)(blockSize / threadSize); iteration++)
    {
        vectorIdx = threadIdx.x + iteration * threadSize;
        if (prefixBuffer[vectorIdx + 1] != prefixBuffer[vectorIdx])
        {
            if (lengthBuffer[vectorIdx] < minEncodeLength)
            {
                uint32_t tmpOffset = prefixBuffer[vectorIdx];
                // uint8_t *bytePtr = (uint8_t *)&buffer[vectorIdx];
                uint8_t *bytePtr = (uint8_t *)&input[(blockOffset + vectorIdx) * vectorSize];
                for (int tmpIndex = 0; tmpIndex < vectorSize * sizeof(T); tmpIndex++)
                {
                    tmpCompressedDataGlobal[blockOffset * vectorSize * sizeof(T) + tmpOffset + tmpIndex] = *(bytePtr + tmpIndex);
                }
            }
            else
            {
                uint32_t tmpOffset = prefixBuffer[vectorIdx];
                tmpCompressedDataGlobal[blockOffset * vectorSize * sizeof(T) + tmpOffset] = lengthBuffer[vectorIdx];
                tmpCompressedDataGlobal[blockOffset * vectorSize * sizeof(T) + tmpOffset + 1] = offsetBuffer[vectorIdx];
            }
        }
    }

    // Copy the memeory back
    if (threadIdx.x == 0)
    {
        for (int flagArrIndex = 0; flagArrIndex < flagCount; flagArrIndex++)
        {
            tmpFlagArrGlobal[blockSize / 8 * blockIdx.x + flagArrIndex] = byteFlagArr[flagArrIndex];
        }
    }
}

// Define the compress Encode kernel functions
template <typename T>
__global__ void compressKernelIII(uint32_t numOfBlocks, uint32_t *flagArrOffsetGlobal, uint32_t *compressedDataOffsetGlobal, uint8_t *tmpFlagArrGlobal, uint8_t *tmpCompressedDataGlobal, uint8_t *flagArrGlobal, uint8_t *compressedDataGlobal)
{
    // Block size in uint of bytes
    const int blockSize = BLOCK_SIZE;

    // Window size in uint of bytes
    const int threadSize = THREAD_SIZE;

    const uint32_t vectorSize = VECTOR_SIZE;

    // find block index
    int blockIndex = blockIdx.x;

    int flagArrOffset = flagArrOffsetGlobal[blockIndex];
    int flagArrSize = flagArrOffsetGlobal[blockIndex + 1] - flagArrOffsetGlobal[blockIndex];

    int compressedDataOffset = compressedDataOffsetGlobal[blockIndex];
    int compressedDataSize = compressedDataOffsetGlobal[blockIndex + 1] - compressedDataOffsetGlobal[blockIndex];

    int tid = threadIdx.x;

    while (tid < flagArrSize)
    {
        flagArrGlobal[flagArrOffset + tid] = tmpFlagArrGlobal[blockSize / 8 * blockIndex + tid];
        tid += threadSize;
    }

    tid = threadIdx.x;

    while (tid < compressedDataSize)
    {
        compressedDataGlobal[compressedDataOffset + tid] = tmpCompressedDataGlobal[blockSize * blockIndex * sizeof(T) * vectorSize + tid];
        tid += threadSize;
    }
}

// Define the decompress kernel functions
template <typename T>
__global__ void decompressKernel(T *output, uint32_t numOfBlocks, uint32_t *flagArrOffsetGlobal, uint32_t *compressedDataOffsetGlobal, uint8_t *flagArrGlobal, uint8_t *compressedDataGlobal)
{
    // Block size in unit of datatype
    const uint32_t blockSize = BLOCK_SIZE;

    const uint32_t vectorSize = VECTOR_SIZE;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numOfBlocks)
    {
        int flagArrOffset = flagArrOffsetGlobal[tid];
        int flagArrSize = flagArrOffsetGlobal[tid + 1] - flagArrOffsetGlobal[tid];

        int compressedDataOffset = compressedDataOffsetGlobal[tid];

        uint32_t vectorIdx = 0;
        uint32_t compressedDataIndex = 0;

        uint8_t byteFlag;

        for (int flagArrayIndex = 0; flagArrayIndex < flagArrSize; flagArrayIndex++)
        {
            byteFlag = flagArrGlobal[flagArrOffset + flagArrayIndex];

            for (int bitCount = 0; bitCount < 8; bitCount++)
            {
                int matchFlag = (byteFlag >> bitCount) & 0x1;
                if (matchFlag == 1)
                {
                    int length = compressedDataGlobal[compressedDataOffset + compressedDataIndex];
                    int offset = compressedDataGlobal[compressedDataOffset + compressedDataIndex + 1];
                    compressedDataIndex += 2;
                    int vectorStart = vectorIdx;
                    for (int tmpDecompIndex = 0; tmpDecompIndex < length; tmpDecompIndex++)
                    {
                        for (int tmpSecondLevelIdx = 0; tmpSecondLevelIdx < vectorSize; tmpSecondLevelIdx++)
                        {
                            output[(tid * blockSize + vectorIdx) * vectorSize + tmpSecondLevelIdx] = output[(tid * blockSize + vectorStart - offset + tmpDecompIndex) * vectorSize + tmpSecondLevelIdx];
                        }
                        vectorIdx++;
                    }
                }
                else
                {
                    uint8_t *tmpPtr = (uint8_t *)&output[(tid * blockSize + vectorIdx) * vectorSize];
                    for (int tmpDecompIndex = 0; tmpDecompIndex < sizeof(T) * vectorSize; tmpDecompIndex++)
                    {
                        *(tmpPtr + tmpDecompIndex) = compressedDataGlobal[compressedDataOffset + compressedDataIndex + tmpDecompIndex];
                    }

                    compressedDataIndex += sizeof(T) * vectorSize;
                    vectorIdx++;
                }
                if (vectorIdx >= blockSize)
                {
                    return;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    std::string inputFileName;
    int decomp = 0;
    int opt;

    /* parse command line */
    while ((opt = getopt(argc, argv, "i:dh")) != -1)
    {
        switch (opt)
        {
        case 'i': /* input file name */
            inputFileName = optarg;
            break;

        case 'd': /* decompression */
            decomp = 1;
            break;

        case 'h': /* help */
            printf(" Usage for compression and decompression: ./gpulz -i {inputfile}\n");
            return 0;
        }
    }

    INPUT_TYPE *hostArray = io::read_binary_to_new_array<INPUT_TYPE>(inputFileName);

#ifdef DEBUG
    int debugOffset = 0;

    printf("print the first 1024 elements:\n");
    for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
    {
        std::cout << hostArray[tmpIndex + debugOffset] << "\t";
    }
    printf("\n");
#endif

    INPUT_TYPE *deviceArray;
    INPUT_TYPE *deviceOutput;
    uint32_t fileSize = io::FileSize(inputFileName);

    uint32_t *flagArrSizeGlobal;
    uint32_t *flagArrOffsetGlobal;
    uint32_t *compressedDataSizeGlobal;
    uint32_t *compressedDataOffsetGlobal;
    uint8_t *tmpFlagArrGlobal;
    uint8_t *tmpCompressedDataGlobal;
    uint8_t *flagArrGlobal;
    uint8_t *compressedDataGlobal;

    // calculate the padding size, unit in bytes
    uint32_t minimumChunkSize = BLOCK_SIZE * VECTOR_SIZE * sizeof(INPUT_TYPE);
    uint32_t paddingSize = fileSize % minimumChunkSize == 0 ? 0 : minimumChunkSize - fileSize % minimumChunkSize;

    // calculate the datatype size, unit in vector
    uint32_t inputVectorSize = static_cast<uint32_t>((fileSize + paddingSize) / sizeof(INPUT_TYPE) / VECTOR_SIZE);

    uint32_t numOfBlocks = inputVectorSize / BLOCK_SIZE;

    INPUT_TYPE *hostOutput = (INPUT_TYPE *)malloc(inputVectorSize * sizeof(INPUT_TYPE) * VECTOR_SIZE);

    // malloc the device buffer and set it as 0
    cudaMalloc((void **)&deviceArray, fileSize + paddingSize);
    cudaMalloc((void **)&deviceOutput, fileSize + paddingSize);

    cudaMalloc((void **)&flagArrSizeGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&flagArrOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&compressedDataSizeGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&compressedDataOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&tmpFlagArrGlobal, sizeof(uint8_t) * inputVectorSize / 8);
    cudaMalloc((void **)&tmpCompressedDataGlobal, sizeof(INPUT_TYPE) * inputVectorSize * VECTOR_SIZE);
    cudaMalloc((void **)&flagArrGlobal, sizeof(uint8_t) * inputVectorSize / 8);
    cudaMalloc((void **)&compressedDataGlobal, sizeof(INPUT_TYPE) * inputVectorSize * VECTOR_SIZE);

    // initialize the mem as 0
    cudaMemset(deviceArray, 0, fileSize + paddingSize);
    cudaMemset(deviceOutput, 0, fileSize + paddingSize);
    cudaMemset(flagArrSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(flagArrOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(compressedDataSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(compressedDataOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(tmpFlagArrGlobal, 0, sizeof(uint8_t) * inputVectorSize / 8);
    cudaMemset(tmpCompressedDataGlobal, 0, sizeof(INPUT_TYPE) * inputVectorSize * VECTOR_SIZE);

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // copy host memory to device
    cudaMemcpy(deviceArray, hostArray, fileSize, cudaMemcpyHostToDevice);

    // printf("num of blocks: %d\nfile size: %d\npadding size: %d\n data type length: %d\n", numOfBlocks, fileSize, paddingSize, inputVectorSize);

    dim3 gridDim(numOfBlocks);
    dim3 blockDim(THREAD_SIZE);

    dim3 deGridDim(ceil(float(numOfBlocks) / 32));
    dim3 deBlockDim(32);

    uint32_t *flagArrOffsetGlobalHost;
    uint32_t *compressedDataOffsetGlobalHost;
    uint8_t *tmpFlagArrGlobalHost;
    uint8_t *tmpCompressedDataGlobalHost;
    uint8_t *flagArrGlobalHost;
    uint8_t *compressedDataGlobalHost;

    flagArrOffsetGlobalHost = (uint32_t *)malloc(sizeof(uint32_t) * (numOfBlocks + 1));
    compressedDataOffsetGlobalHost = (uint32_t *)malloc(sizeof(uint32_t) * (numOfBlocks + 1));
    tmpFlagArrGlobalHost = (uint8_t *)malloc(sizeof(uint8_t) * inputVectorSize / 8);
    tmpCompressedDataGlobalHost = (uint8_t *)malloc(sizeof(INPUT_TYPE) * inputVectorSize * VECTOR_SIZE);
    flagArrGlobalHost = (uint8_t *)malloc(sizeof(uint8_t) * inputVectorSize / 8);
    compressedDataGlobalHost = (uint8_t *)malloc(sizeof(INPUT_TYPE) * inputVectorSize * VECTOR_SIZE);

    cudaEvent_t compStart, compStop, decompStart, decompStop;
    cudaEventCreate(&compStart);
    cudaEventCreate(&compStop);
    cudaEventCreate(&decompStart);
    cudaEventCreate(&decompStop);

    cudaEventRecord(compStart);

    int minEncodeLength = 1;

    // launch kernels
    compressKernelI<INPUT_TYPE><<<gridDim, blockDim>>>(deviceArray, numOfBlocks, flagArrSizeGlobal, compressedDataSizeGlobal, tmpFlagArrGlobal, tmpCompressedDataGlobal, minEncodeLength);

    // Determine temporary device storage requirements
    void *flag_d_temp_storage = NULL;
    size_t flag_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(flag_d_temp_storage, flag_temp_storage_bytes, flagArrSizeGlobal, flagArrOffsetGlobal, numOfBlocks + 1);

    // Allocate temporary storage
    cudaMalloc(&flag_d_temp_storage, flag_temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(flag_d_temp_storage, flag_temp_storage_bytes, flagArrSizeGlobal, flagArrOffsetGlobal, numOfBlocks + 1);

    // Determine temporary device storage requirements
    void *data_d_temp_storage = NULL;
    size_t data_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(data_d_temp_storage, data_temp_storage_bytes, compressedDataSizeGlobal, compressedDataOffsetGlobal, numOfBlocks + 1);

    // Allocate temporary storage
    cudaMalloc(&data_d_temp_storage, data_temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(data_d_temp_storage, data_temp_storage_bytes, compressedDataSizeGlobal, compressedDataOffsetGlobal, numOfBlocks + 1);

    compressKernelIII<INPUT_TYPE><<<gridDim, blockDim>>>(numOfBlocks, flagArrOffsetGlobal, compressedDataOffsetGlobal, tmpFlagArrGlobal, tmpCompressedDataGlobal, flagArrGlobal, compressedDataGlobal);

    cudaEventRecord(compStop);

    cudaEventRecord(decompStart);

    decompressKernel<INPUT_TYPE><<<deGridDim, deBlockDim>>>(deviceOutput, numOfBlocks, flagArrOffsetGlobal, compressedDataOffsetGlobal, flagArrGlobal, compressedDataGlobal);
    cudaEventRecord(decompStop);

    // copy the memory back to global
    cudaMemcpy(flagArrOffsetGlobalHost, flagArrOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(compressedDataOffsetGlobalHost, compressedDataOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpFlagArrGlobalHost, tmpFlagArrGlobal, sizeof(uint8_t) * inputVectorSize / 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpCompressedDataGlobalHost, tmpCompressedDataGlobal, sizeof(INPUT_TYPE) * inputVectorSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(flagArrGlobalHost, flagArrGlobal, sizeof(uint8_t) * inputVectorSize / 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(compressedDataGlobalHost, compressedDataGlobal, sizeof(INPUT_TYPE) * inputVectorSize, cudaMemcpyDeviceToHost);

    cudaMemcpy(hostOutput, deviceOutput, fileSize, cudaMemcpyDeviceToHost);

#ifdef DEBUG
    printf("print the first 1024 flag array offset elements:\n");
    for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
    {
        printf("%d\t", flagArrOffsetGlobalHost[tmpIndex]);
    }
    printf("\n");

    printf("print the first 1024 compressed data offset elements:\n");
    for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
    {
        printf("%d\t", compressedDataOffsetGlobalHost[tmpIndex]);
    }
    printf("\n");

    printf("print the first 1024 flag array elements:\n");
    for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
    {
        printf("%d\t", flagArrGlobalHost[tmpIndex]);
    }
    printf("\n");

    printf("print the first 1024 compressed data elements:\n");
    for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
    {
        printf("%d\t", compressedDataGlobalHost[tmpIndex]);
    }
    printf("\n");

    // printf("print the first 1024 tmp flag array elements:\n");
    // for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
    // {
    //   printf("%d\t", tmpFlagArrGlobalHost[tmpIndex]);
    // }
    // printf("\n");
#endif

    // verify the final output
    for (int verifyIndex = 0; verifyIndex < fileSize / sizeof(INPUT_TYPE); verifyIndex++)
    {
        if (hostArray[verifyIndex] != hostOutput[verifyIndex])
        {
            printf("verification failed!!! Index %d is wrong\n", verifyIndex);
            std::cout << "hostArray: " << hostArray[verifyIndex] << ", hostOutput: " << hostOutput[verifyIndex] << std::endl;
            break;
        }
    }

    float originalSize = fileSize;
    float compressedSize = sizeof(uint32_t) * (numOfBlocks + 1) * 2 + flagArrOffsetGlobalHost[numOfBlocks] + compressedDataOffsetGlobalHost[numOfBlocks];
    float compressionRatio = originalSize / compressedSize;
    std::cout << "compression ratio: " << compressionRatio << std::endl;

    cudaEventSynchronize(decompStop);
    float compTime = 0;
    float decompTime = 0;
    cudaEventElapsedTime(&compTime, compStart, compStop);
    cudaEventElapsedTime(&decompTime, decompStart, decompStop);
    float compTp = float(fileSize) / 1024 / 1024 / compTime;
    float decompTp = float(fileSize) / 1024 / 1024 / decompTime;
    std::cout << "compression e2e throughput: " << compTp << " GB/s" << std::endl;
    std::cout << "decompression e2e throughput: " << decompTp << " GB/s" << std::endl;

    // free dynamic arrays
    free(flagArrOffsetGlobalHost);
    free(compressedDataOffsetGlobalHost);
    free(tmpFlagArrGlobalHost);
    free(tmpCompressedDataGlobalHost);
    free(flagArrGlobalHost);
    free(compressedDataGlobalHost);

    cudaFree(deviceArray);
    cudaFree(deviceOutput);

    cudaFree(flagArrSizeGlobal);
    cudaFree(flagArrOffsetGlobal);
    cudaFree(compressedDataSizeGlobal);
    cudaFree(compressedDataOffsetGlobal);
    cudaFree(tmpFlagArrGlobal);
    cudaFree(tmpCompressedDataGlobal);
    cudaFree(flagArrGlobal);
    cudaFree(compressedDataGlobal);

    cudaFree(flag_d_temp_storage);
    cudaFree(data_d_temp_storage);

    free(hostOutput);

    delete hostArray;

    return 0;
}