#include "../nn.h"

// declare and define pointers to functions
void (*COPY_TO_DEVICE)(tensor*) = NULL;
tensor* (*COPY_FROM_DEVICE)(tensor*) = NULL;

// include this unconditionally, because used by both cuda and cpu
#include "cpu/move_data.cpp"

#if DEVICE == CPU
    #include "cpu/kernels.cpp"
    #include "cpu/kernels_conv.cpp"
#elif DEVICE == CUDA
    #include "cuda/move_data.cpp"
    #include "cuda/kernels.cu"
    #include "cuda/kernels_conv.cu"
#endif

#include "common.cpp"

#define set_backend_device() ((DEVICE==CUDA) ? set_backend_cuda() : set_backend_cpu())
