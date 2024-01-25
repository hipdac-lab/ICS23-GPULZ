#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <errno.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "cufile.h"

namespace io
{
    size_t FileSize(const std::string filename)
    {
        std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
        return static_cast<size_t>(in.tellg());
    }

    template <typename T>
    T *read_binary_to_new_array(const std::string &fname)
    {
        std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
        if (not ifs.is_open())
        {
            std::cerr << "fail to open " << fname << std::endl;
            exit(1);
        }
        size_t dtype_len = FileSize(fname) / sizeof(T);
        auto _a = new T[dtype_len]();
        ifs.read(reinterpret_cast<char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ifs.close();
        return _a;
    }

    template <typename T>
    void read_binary_to_array(const std::string &fname, T *_a, size_t dtype_len)
    {
        std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
        if (not ifs.is_open())
        {
            std::cerr << "fail to open " << fname << std::endl;
            exit(1);
        }
        ifs.read(reinterpret_cast<char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ifs.close();
    }

    template <typename T>
    void write_array_to_binary(const std::string &fname, T *const _a, size_t const dtype_len)
    {
        std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
        if (not ofs.is_open())
            return;
        ofs.write(reinterpret_cast<const char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ofs.close();
    }

    void cufileWrite(const char *dir, void *inputPtrDevice, const size_t size)
    {
        CUfileDescr_t cf_descr;
        CUfileError_t status;
        CUfileHandle_t cf_handle;
        ssize_t ret = -1;
        int fd;

        // Write loaded data from GPU memory to a new file
        fd = open(dir, O_CREAT | O_RDWR | O_DIRECT, 0664);
        if (fd < 0)
        {
            std::cerr << "write file open error : " << std::strerror(errno) << std::endl;
            return;
        }

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS)
        {
            std::cerr << "file register error" << std::endl;
            close(fd);
            return;
        }

        ret = cuFileWrite(cf_handle, inputPtrDevice, size, 0, 0);
        if (ret < 0)
        {
            if (IS_CUFILE_ERR(ret))
                std::cerr << "write failed" << std::endl;
        }

        cuFileHandleDeregister(cf_handle);
        close(fd);
    }

    void cufileRead(const char *dir, void *outputPtrDevice, const size_t size)
    {
        CUfileDescr_t cf_descr;
        CUfileError_t status;
        CUfileHandle_t cf_handle;
        ssize_t ret = -1;
        int fd;

        // Write loaded data from GPU memory to a new file
        fd = open(dir, O_RDONLY | O_DIRECT);
        if (fd < 0)
        {
            std::cerr << "read file open error : " << dir << std::strerror(errno) << std::endl;
            return;
        }

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS)
        {
            std::cerr << "file register error" << std::endl;
            close(fd);
            return;
        }

        ret = cuFileRead(cf_handle, outputPtrDevice, size, 0, 0);
        if (ret < 0)
        {
            if (IS_CUFILE_ERR(ret))
                std::cerr << "read failed" << std::endl;
        }

        cuFileHandleDeregister(cf_handle);
        close(fd);
    }

}

#endif // UTILS_H