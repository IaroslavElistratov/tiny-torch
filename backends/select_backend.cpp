#include "../nn.h"

// // declare and define pointers to functions
// void (*COPY_TO_DEVICE)(tensor*) = NULL;
// tensor* (*COPY_FROM_DEVICE)(tensor*) = NULL;

// include this unconditionally, because used by both cuda and cpu
#include "cpu/move_data.cpp"

#include "common/asserts.cpp"

#if DEVICE == CPU
    #include "cpu/kernels.cpp"
    #include "cpu/kernels_conv.cpp"
    #define set_backend_device() set_backend_cpu()

    // this workaround is needed for compiling for python API (when main isn't called):
    // the two lines below are equivalent to calling set_backend_device -- but
    // bc in C a fn can only be called from main -- I can't just call that fn here
    void (*COPY_TO_DEVICE)(tensor*) = copy_to_cpu;
    tensor* (*COPY_FROM_DEVICE)(tensor*) = copy_from_cpu;

    // todo-low: cleanup
    // otherwise with CPU backend, linker error because tensor.free_tensor uses that fn name;
    // actual function definition
    void checkCudaErrors(cudaError_t err) {};

#elif DEVICE == CUDA
    #include "cuda/move_data.cu"
    #include "cuda/kernels.cu"
    #include "cuda/kernels_conv.cu"
    #include "cuda/kernels_reduce.cu"
    #define set_backend_device() set_backend_cuda()

    void (*COPY_TO_DEVICE)(tensor*) = copy_to_cuda;
    tensor* (*COPY_FROM_DEVICE)(tensor*) = copy_from_cuda;

#endif

#include "common/kernels.cpp"
