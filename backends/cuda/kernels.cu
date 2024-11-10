#define NUM_THREADS 32
#define CUDA_DEBUG true


tensor* transpose_k(tensor*);
void matmul_input_checks(tensor*, tensor*);
void transpose_input_checks(tensor*);

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
    matmul_input_checks(a, b);

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

void matmul_input_checks(tensor* a, tensor* b){
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



// a(N, M) -> out(M, N)
__global__ void TransposeKernel(float* a, float* out, int M, int N){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if ((m<M) && (n<N)){
        out[m*N + n] = a[n*M + m];
    }
}

tensor* transpose_k(tensor* a){
    transpose_input_checks(a);

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

void transpose_input_checks(tensor* a){
    if (a->device!=CUDA){
        printf("[cuda Transpose] Error: expect device cuda");
        return;
    }
    if (a->num_dims!=2){
        printf("[cuda Transpose] Error: expect 2-dim inputs");
        return;
    }
}
