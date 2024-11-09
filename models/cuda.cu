#include <iostream> // todo: use C only
using namespace std;

#define DEVICE CUDA
#define print(a) ((DEVICE == CUDA) ? cuda_print_2d(a) : print_2d(a))

#include "../tensor.cpp"
#include "../ops.cpp"
#include "../utils.cpp"
#include "../print.cpp"


void cuda_print_2d(tensor* t)
{
    // todo: can just define a macro for print to call 4 lines below and then call the orignal print2d (no need for cuda_print_2d)
    cudaDeviceSynchronize();
    int size = t->size * sizeof(float);
    float* host_data = (float*)malloc(size);
    cudaError_t err = cudaMemcpy(host_data, t->data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("[cuda memcopy] error: %s",  cudaGetErrorString(err));
    }

    sprint_2d(t);

    int y = t->shape[0];
    int z = t->shape[1];

    for (int yi=0; yi<y; yi++){
        printf("[");
        for (int zi=0; zi<z; zi++){
            // todo: doens't make sense to use cpu strides when acessing contigious copy data
            // int idx = index_2d(t, yi, zi);
            int idx = yi * t->shape[1] + zi;
            printf("%8.4f, ", host_data[idx]);
        }
        printf("],\n");
    }
    printf("\n");
}


// todo-now:
// for cuda, use same operation abstractions (as for cpu), but make these abstractions call host stubs for cuda kernels (instead of my kernles for cpu) -- this will preserve my graph building functionality
//   - re-use this from ops.cpp: use same names for kernels, just make them refer to different impls (cpu, cuda) depending on wether device is CUDA or not -- this will reduce code duplication needed to copy paste ops

// todo-now:
// and "matmul_bwd", "batched_matmul_bwd", "div_bwd", "pow_bwd", "reduce_sum_bwd" can also be re-used!

int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int N = 16;
    int M = 8;
    int D = 4;

    tensor* x = CudaTensor(N, M);
    set_name(x, "x"); print(x);

    tensor* w1 = CudaTensor(M, D);
    set_name(w1, "w1"); print(w1);

    // x(N, M) @ w1(M, D) = out1(N, D)
    tensor* out = matmul(x, w1);
    set_name(out, "out"); print(out);



    x = CudaTensor(N, M);
    set_name(x, "x"); print(x);

    out = transpose(x);
    set_name(out, "out"); print(out);

    return 0;
}
