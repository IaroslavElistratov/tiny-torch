#include "../nn.h"

// declare and define pointers to functions
void (*COPY_TO_DEVICE)(tensor*) = NULL;
tensor* (*COPY_FROM_DEVICE)(tensor*) = NULL;

// include this unconditionally, because used by both cuda and cpu
#include "cpu/move_data.cpp"

#if DEVICE == CPU
    #include "cpu/kernels.cpp"
    #include "cpu/kernels_conv.cpp"
    #define set_backend_device() set_backend_cpu()
#elif DEVICE == CUDA
    #include "cuda/move_data.cu"
    #include "cuda/kernels.cu"
    #include "cuda/kernels_conv.cu"
    #include "cuda/kernels_reduce.cu"
    #define set_backend_device() set_backend_cuda()
#endif

#include "common.cpp"
