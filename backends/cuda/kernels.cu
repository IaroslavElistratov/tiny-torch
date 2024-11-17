#include "../../nn.h"

#define NUM_THREADS 32
#define CUDA_DEBUG true


tensor* transpose_k(tensor*);
// tensor* Tensor(...);


void unary_input_checks(tensor* a){
    if (a->device!=CUDA){
        printf("[cuda kernel] Error: expected device cuda\n");
        exit(1);
    }
    if (a->num_dims!=2){
        printf("[cuda kernel] Error: expected 2-dim inputs\n");
        exit(1);
    }
}

void binary_input_checks(tensor* a, tensor* b){
    unary_input_checks(a);
    unary_input_checks(b);
}

void binary_elsementwise_input_checks(tensor* a, tensor* b){
    binary_input_checks(a, b);
    if (a->shape[0]!=b->shape[0] || a->shape[1]!=b->shape[1]){
        printf("[cuda kernel] Error: expected shapes to match\n");
        printf("a.shape=(%i, %i), b.shape=(%i, %i)\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        exit(1);
    }
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backward defined in common ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



// binary elementwise


typedef void (*BinaryElementwiseKernel)(float* a, float* b, float* out, int size);

// comment: I think unnecessary to launch 2d blocks for binary/unary ops -- since data is contigious can just launch 1d blocks
tensor* _launch_binary_elsementwise(BinaryElementwiseKernel kernel, tensor* a, tensor* b, tensor* out){
    binary_elsementwise_input_checks(a, b);

    // added out arg to _launch_binary_elsementwise, so that this fn can be re-used in add_k_
    if (!out){
        out = TensorLikeFill(a, 0.0);
    }

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(out->size/num_threads), 1, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda binary_elsementwise] grid: (%f, 1, 1)\n", ceil(a->size/num_threads));
        printf("[cuda binary_elsementwise] block: (%f, 1, 1)\n", num_threads);
    }

    kernel<<<dimGrid, dimBlock>>>(a->data, b->data, out->data, out->size);

    return out;
}


// todo-now: unify this as well
__global__ void AddKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] + b[idx];
    }
}

tensor* add_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[add_k]\n");
    return _launch_binary_elsementwise(AddKernel, a, b, NULL);
}

// need the below bc add_k_ is used in backends/common div_bwd
// AddKernel is semantically similar to cpu's add_k_, except the
// latter expects tensors (not *floats)
// Maybe in the future change my cuda kernels to expect tensors to
// get rid of this, other other hand add_k_ and mul_k_ are useful in
// impls of some bwd funcs
tensor* add_k_(tensor* a, tensor* b, tensor* c){
    if (CUDA_DEBUG) printf("[add_k_]\n");
    return _launch_binary_elsementwise(AddKernel, a, b, c);
}


__global__ void SubKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] - b[idx];
    }
}

tensor* sub_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[sub_k]\n");
    return _launch_binary_elsementwise(SubKernel, a, b, NULL);
}


__global__ void MulKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] * b[idx];
    }
}

tensor* mul_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[mul_k]\n");
    return _launch_binary_elsementwise(MulKernel, a, b, NULL);
}


__global__ void DivKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] / b[idx];
    }
}

tensor* div_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[div_k]\n");
    return _launch_binary_elsementwise(DivKernel, a, b, NULL);
}


// binary


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
            //   (I belive it's contiguous as this is ouput of cuda-malloc called from inside Tensor constructor)
            // out += a->data[index_2d(a, n, k)] * b->data[index_2d(b, k, d)];
            curr_out += a[n*M + m] * b[m*D + d];
        }
        out[n*D + d] = curr_out;
    }
}

// todo: this _k naming is maintain parity between names of the cpu kernels and the cuda kernels (so that both can be used polimorphically in ops)
//     but the below isn't "kernel" in the sense of this word, instead it's a stub that calls the actual kernel (MatMulKernel)

// a(N, M) @ b(M, D) = out(N, D)
tensor* matmul_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[matmul_k]\n");
    binary_input_checks(a, b);
    if (a->shape[1] != b->shape[0]){
        printf("[cuda MatMul] Error: inner dim doesn't match\n");
        exit(1);
    }

    int N = a->shape[0], M = a->shape[1], D = b->shape[1];
    // todo: fill w 0
    tensor* out = Tensor(N, D);

    // todo: unify 7 lines below into a fn (e.g. compute_launch_shapes), re-use acorss all stubs
    // important to have it float to avoid int division
    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(N/num_threads), ceil(D/num_threads), 1);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda MatMul] grid: (%f, %f, 1)\n", ceil(N/num_threads), ceil(D/num_threads));
        printf("[cuda MatMul] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    // todo: to avoid passing shapes, cp tensor structs to cuda and pass them to the kernel?
    MatMulKernel<<<dimGrid, dimBlock>>>(a->data, b->data, out->data, N, M, D);

    return out;
}


// unary


typedef void (*UnaryKernel)(float* a, float* out, int size);

tensor* _launch_unary_elsementwise(UnaryKernel kernel, tensor* a){
    unary_input_checks(a);

    tensor* out = TensorLikeFill(a, 0.0);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(out->size/num_threads), 1, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda unary] grid: (%f, 1, 1)\n", ceil(a->size/num_threads));
        printf("[cuda unary] block: (%f, 1, 1)\n", num_threads);
    }

    kernel<<<dimGrid, dimBlock>>>(a->data, out->data, out->size);

    return out;
}


__global__ void PowKernel(float* a, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pow_exponent = 2;
    if (idx<size){
        out[idx] = (float)pow(a[idx], pow_exponent);
    }
}

tensor* pow_k(tensor* a, int exponent){
    if (CUDA_DEBUG) printf("[pow_k]\n");
    // cpu's pow_k expects exponent as arg, but here because of standardized _launch_unary_elsementwise interface I hardcode it
    // todo: pass it via global argument
    // pow_exponent = exponent;
    if (exponent!=2){
        printf("[cuda pow_k] currently this kernel only supports exponent=2\n");
        exit(1);
    }
    return _launch_unary_elsementwise(PowKernel, a);
}


__global__ void NegKernel(float* a, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = -a[idx];
    }
}

tensor* neg_k(tensor* a){
    if (CUDA_DEBUG) printf("[neg_k]\n");
    return _launch_unary_elsementwise(NegKernel, a);
}


// todo: for transpose, launch 1d blocks so that it can re-use _launch_unary_elsementwise?
// a(N, M) -> out(M, N)
__global__ void TransposeKernel(float* a, float* out, int M, int N){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if ((m<M) && (n<N)){
        out[m*N + n] = a[n*M + m];
    }
}

tensor* transpose_k(tensor* a){
    unary_input_checks(a);

    int N = a->shape[0], M = a->shape[1];
    // todo: allocate empty
    tensor* out = Tensor(M, N);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(M/num_threads), ceil(N/num_threads), 1);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda Transpose] grid: (%f, %f, 1)\n", ceil(M/num_threads), ceil(N/num_threads));
        printf("[cuda Transpose] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    TransposeKernel<<<dimGrid, dimBlock>>>(a->data, out->data, M, N);

    return out;
}


// void repeat_bwd(tensor* upstream, tensor* out)

// void pow_bwd(tensor* upstream, tensor* out)

// void reduce_sum_bwd(tensor* upstream, tensor* out)

// void batched_matmul_bwd(tensor* upstream, tensor* out) {



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backward NOT defined in common ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



