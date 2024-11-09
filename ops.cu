#include "nn.h"
#include "indexing.cpp"

#define NUM_THREADS 32
#define CUDA_DEBUG true

void input_checks(tensor*, tensor*);


// a(N, M) @ b(M, D) = out(N, D)
__global__ void MatMulKernel(float* a, float* b, float* out, int N, int M, int D){

    // (block idx * num threads per block) + threadidx
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if ((n<N) && (d<D)){
        float curr_out = 0.0;
        for (int m=0; m<M; m++){
            // todo-low: calling a __host__ function("index_2d") from a __global__ function("MatMulKernel") is not allowed
            //   But, it doesn't really make sense to use index_2d when you're accessing contiguous cuda memory
            //   (I belive it's contiguous as this is ouput of cuda-malloc called from inside CudaTensor constructor)
            // out += a->data[index_2d(a, n, k)] * b->data[index_2d(b, k, d)];
            curr_out += a[n*M + m] * b[m*D + d];
        }
        out[n*D + d] = curr_out;
    }
}

// a(N, M) @ b(M, D) = out(N, D)
tensor* MatMul(tensor* a, tensor* b){

    input_checks(a, b);

    int N = a->shape[0], M = a->shape[1], D = b->shape[1];
    // todo: fill w 0
    tensor* out = CudaTensor(N, D);

    // important to have it float to avoid int division
    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(N/num_threads), ceil(D/num_threads), 1);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda MatMul] grid: (%f, %f, 1)\n", ceil(N/num_threads), ceil(D/num_threads));
        printf("[cuda MatMul] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    // todo: cp tensor structs to cuda and pass them to the kernel?
    MatMulKernel<<<dimGrid, dimBlock>>>(a->data, b->data, out->data, N, M, D);

    return out;
}

void input_checks(tensor* a, tensor* b){
    if (a->device!=CUDA || b->device!=CUDA){
        printf("[cuda MatMul] Error: expect device cuda");
        return;
    }
    if (a->num_dims!=2 || b->num_dims!=2){
        printf("[cuda MatMul] Error: expect 2-dim inputs");
        return;
    }
    if (a->shape[1] != b->shape[0]){
        printf("[cuda MatMul] Error: inner dim doesn't match");
        return;
    }
}
