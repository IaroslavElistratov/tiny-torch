#include "../../nn.h"



void assert_device(tensor* a){
    if (a->device!=CUDA){
        printf("[assert_device] Error: expected device cuda\n");
        exit(1);
    }
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backward defined in common ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



// binary elementwise


typedef void (*BinaryElementwiseKernel)(float* a, float* b, float* out, int size);

// comment: I think unnecessary to launch 2d blocks for binary/unary ops -- since data is contigious can just launch 1d blocks
tensor* _launch_binary_elementwise(BinaryElementwiseKernel kernel, tensor* a, tensor* b, tensor* out){

    assert_binary_elementwise(a, b);

    // added out arg to _launch_binary_elementwise, so that this fn can be re-used in add_k_
    if (!out){
        out = TensorLikeFill(a, 0.0);
    }

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(out->size/num_threads), 1, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda binary_elementwise] grid: (%f, 1, 1)\n", ceil(a->size/num_threads));
        printf("[cuda binary_elementwise] block: (%f, 1, 1)\n", num_threads);
    }

    kernel<<<dimGrid, dimBlock>>>(a->data, b->data, out->data, out->size);
    return out;
}


// todo: unify this as well
__global__ void AddKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] + b[idx];
    }
}

tensor* add_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[add_k]\n");
    return _launch_binary_elementwise(AddKernel, a, b, NULL);
}

// need the below bc add_k_ is used in backends/common div_bwd
// AddKernel is semantically similar to cpu's add_k_, except the
// latter expects tensors (not *floats)
// Maybe in the future change my cuda kernels to expect tensors to
// get rid of this, other other hand add_k_ and mul_k_ are useful in
// impls of some bwd funcs
tensor* add_k_(tensor* a, tensor* b, tensor* c){
    if (CUDA_DEBUG) printf("[add_k_]\n");
    return _launch_binary_elementwise(AddKernel, a, b, c);
}

// does not verify input shape similarity, only verifies input size similarity
// another way to name this fn is "unsafe_add_k_"
tensor* unsafe_add_k_(tensor* a, tensor* b, tensor* c){
    if (CUDA_DEBUG) printf("[unsafe_add_k_]\n");
    IS_INPUT_DIM_CHECK = false;
    tensor* out = _launch_binary_elementwise(AddKernel, a, b, c);
    IS_INPUT_DIM_CHECK = true;
    return out;
}

__global__ void SubKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] - b[idx];
    }
}

tensor* sub_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[sub_k]\n");
    return _launch_binary_elementwise(SubKernel, a, b, NULL);
}


__global__ void MulKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] * b[idx];
    }
}

tensor* mul_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[mul_k]\n");
    return _launch_binary_elementwise(MulKernel, a, b, NULL);
}

// used in exp_bwd, log_bwd
tensor* mul_k_(tensor* a, tensor* b, tensor* c){
    if (CUDA_DEBUG) printf("[mul_k_]\n");
    return _launch_binary_elementwise(MulKernel, a, b, c);
}


__global__ void DivKernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = a[idx] / b[idx];
    }
}

tensor* div_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[div_k]\n");
    return _launch_binary_elementwise(DivKernel, a, b, NULL);
}


// binary


// a(?B, N, M) @ b(?B, M, D) = out(?B, N, D)
__global__ void MatMulKernel(float* a, float* b, float* out, int N, int M, int D, bool is_batched){
    // (block idx * num threads per block) + threadidx
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    // cancels out when not batched
    int batch = blockIdx.z * is_batched;

    if ((n<N) && (d<D)){
        float curr_out = 0.0;
        for (int m=0; m<M; m++){
            // todo-low: calling a __host__ function("index_2d") from a __global__ function("MatMulKernel") is not allowed
            //   But, it doesn't really make sense to use index_2d when you're accessing contiguous cuda memory
            //   (I belive it's contiguous as this is ouput of cuda-malloc called from inside Tensor constructor)
            // out += a->data[index_2d(a, n, k)] * b->data[index_2d(b, k, d)];
            curr_out += a[batch*N*M + n*M + m] * b[batch*M*D + m*D + d];
        }
        out[batch*N*D + n*D + d] = curr_out;
    }
}

// todo: this _k naming is maintain parity between names of the cpu kernels and the cuda kernels (so that both can be used polimorphically in ops)
//     but the below isn't "kernel" in the sense of this word, instead it's a stub that calls the actual kernel (MatMulKernel)

// a(N, M) @ b(M, D) = out(N, D)
tensor* matmul_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[matmul_k]\n");
    assert_input(a, 2);
    assert_input(b, 2);
    if (a->shape[1] != b->shape[0]){
        printf("[cuda MatMul] Error: inner dim doesn't match, saw: a(%i, %i) b(%i, %i)\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        exit(1);
    }

    int N = a->shape[0], M = a->shape[1], D = b->shape[1];
    // todo: fill w 0, here and in other stubs
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
    MatMulKernel<<<dimGrid, dimBlock>>>(a->data, b->data, out->data, N, M, D, false);
    return out;
}

// a(B, N, M) @ b(B, M, D) = out(B, N, D)
tensor* batched_matmul_k(tensor* a, tensor* b){
    if (CUDA_DEBUG) printf("[batched_matmul_k]\n");
    assert_input(a, 3);
    assert_input(b, 3);
    if (a->shape[2] != b->shape[1]){
        printf("[cuda BatchedMatMul] Error: inner dim doesn't match\n");
        exit(1);
    }

    int B = a->shape[0], N = a->shape[1], M = a->shape[2], D = b->shape[2];
    tensor* out = Tensor(B, N, D);

    // important to have it float to avoid int division
    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(N/num_threads), ceil(D/num_threads), B);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda BatchedMatMul] grid: (%f, %f, %i)\n", ceil(N/num_threads), ceil(D/num_threads), B);
        printf("[cuda BatchedMatMul] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    MatMulKernel<<<dimGrid, dimBlock>>>(a->data, b->data, out->data, N, M, D, true);
    return out;
}


// unary


typedef void (*UnaryKernel)(float* a, float* out, int size);

tensor* _launch_unary_elementwise(UnaryKernel kernel, tensor* a){

    // don't assert n_dims == 2, bc in conv_net.cu 4d input is fed to relu kernel, which calls _launch_unary_elementwise;
    assert_device(a);
    assert_contiguous(a);

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
    // cpu's pow_k expects exponent as arg, but here because of standardized _launch_unary_elementwise interface I hardcode it
    // todo: pass it via global argument
    // pow_exponent = exponent;
    if (exponent!=2){
        printf("[cuda pow_k] currently this kernel only supports exponent=2\n");
        exit(1);
    }
    return _launch_unary_elementwise(PowKernel, a);
}


__global__ void ExpKernel(float* a, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = expf(a[idx]);
    }
}

tensor* exp_k(tensor* a){
    if (CUDA_DEBUG) printf("[exp_k]\n");
    return _launch_unary_elementwise(ExpKernel, a);
}


__global__ void LogKernel(float* a, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = logf(a[idx]);
    }
}

tensor* log_k(tensor* a){
    if (CUDA_DEBUG) printf("[log_k]\n");
    return _launch_unary_elementwise(LogKernel, a);
}


__global__ void NegKernel(float* a, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = -a[idx];
    }
}

tensor* neg_k(tensor* a){
    if (CUDA_DEBUG) printf("[neg_k]\n");
    return _launch_unary_elementwise(NegKernel, a);
}


__global__ void ReluKernel(float* a, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = (a[idx] < 0.0) ? 0.0 : a[idx];
    }
}

tensor* relu_k(tensor* a){
    if (CUDA_DEBUG) printf("[relu_k]\n");
    return _launch_unary_elementwise(ReluKernel, a);
}


__global__ void ReluBwdKernel(float* a, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size){
        out[idx] = (a[idx] > 0.0) ? 1.0 : 0.0;
    }
}

void relu_bwd(tensor* upstream, tensor* out) {
    if (CUDA_DEBUG) printf("[relu_bwd]\n");
    tensor* a = out->inputs[0];
    tensor* local = _launch_unary_elementwise(ReluBwdKernel, a);
    a->grad = mul_k(local, upstream);
    // free(local);
}



// todo: for transpose, launch 1d blocks so that it can re-use _launch_unary_elementwise?
// todo-high: this kernel (TransposeKernel, BatchedTransposeKernel) basically does: swap strides + "contigify" -- can I get rid of this kernel when
//   - support non-contigious data in my cuda kernel (which means when use "at" instead of t[idx] to index into tensors)
//   - which implies changing kernels to input tensors not floats, to access strides
// a(?B, N, M) -> out(?B, M, N)
__global__ void TransposeKernel(float* a, float* out, int M, int N, bool is_batched){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z * is_batched;

    if (m<M && n<N){
        out[batch*M*N + m*N + n] = a[batch*N*M + n*M + m];
    }
}

tensor* transpose_k(tensor* a){
    if (CUDA_DEBUG) printf("[transpose_k]\n");
    assert_input(a, 2);

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

    TransposeKernel<<<dimGrid, dimBlock>>>(a->data, out->data, M, N, false);
    return out;
}




// a(B, 1) -> out(B, N)
__global__ void RepeatKernel(float* a, float* out, int num_repeats, int B){
    // (block idx * num threads per block) + thread idx
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    // this is repeat_idx (0-num_repeats) that this thread represents
    int i = blockIdx.y; // * blockDim.y + threadIdx.y;
    if (b<B && i<num_repeats){
        out[b*num_repeats + i] = a[b];
    }
}

// question-now: or use this kernel?
// // a(B, 1) -> out(B, N)
// __global__ void RepeatKernel(float* a, float* out, int num_repeats, int B){
//     // (block idx * num threads per block) + thread idx
//     int b = blockIdx.x * blockDim.x + threadIdx.x;
//     printf("[kernel] b=%i\n", b);
//     if (b<B){
//         for (int i=0; i<num_repeats; i++){
//             // Indexing into out: since out(B, num_repeats), to get to the next batch element
//             // (IOW out->stride[0]) need to skip "num_repeats" locations in memory;
//             // Indexing into a: since a(B, 1), stride a->stride[0] is just 1 so can omit it
//             out[b*num_repeats + i] = a[b];
//         }
//     }
// }

tensor* repeat_k(tensor* a, int num_repeats){
    if (CUDA_DEBUG) printf("[repeat_k]\n");
    assert_input(a, 2);
    if (a->shape[1]!=1){
        printf("[CUDA RepeatKernel] Shape error\n");
        exit(1);
    }

    int B = a->shape[0];
    tensor* out = Tensor(B, num_repeats);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(B/num_threads), num_repeats, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda RepeatKernel] grid: (%f, 1, 1)\n", ceil(B/num_threads));
        printf("[cuda RepeatKernel] block: (%f, 1, 1)\n", num_threads);
    }

    RepeatKernel<<<dimGrid, dimBlock>>>(a->data, out->data, num_repeats, B);
    return out;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backward NOT defined in common ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__global__ void SelectKernel(float* input, float* idx, int B, int N, float* out);
__global__ void SelectSetKernel(float* input, float* idx, int B, int N, float value);

// using pointer to value, so that I can pass NULL from select_k (to be able re-use this fn)
tensor* _launch_select(tensor* a, tensor* idx, float value){
    if (CUDA_DEBUG) printf("[_launch_select]\n");
    assert_input(a, 2);
    assert_input(idx, 2);
    if (idx->shape[1]!=1 || idx->shape[0]!=a->shape[0]) {
        printf("[_launch_select] Error shape\n");
        exit(1);
    }

    int B = a->shape[0], N = a->shape[1];

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(B/num_threads), 1, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda _launch_select] grid: (%f, 1, 1)\n", ceil(B/num_threads));
        printf("[cuda _launch_select] block: (%f, 1, 1)\n", num_threads);
    }

    if (value==-1){
        tensor* out = Tensor(B, 1);
        SelectKernel<<<dimGrid, dimBlock>>>(a->data, idx->data, B, N, out->data);
        return out;
    } else {
        SelectSetKernel<<<dimGrid, dimBlock>>>(a->data, idx->data, B, N, value);
        return a;
    }
}

// input(s1, s2), idx(s1, 1) -> out(s1, 1)
__global__ void SelectKernel(float* input, float* idx, int B, int N, float* out){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b<B){
        int input_idx = idx[b];
        // bc out and idx are (B, 1), can simply index into each of them with arr[b]
        out[b] = input[b*N + input_idx];
    }
}
tensor* select_k(tensor* a, tensor* idx){
    return _launch_select(a, idx, -1);
}

__global__ void SelectSetKernel(float* input, float* idx, int B, int N, float value){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b<B){
        int input_idx = idx[b];
        input[b*N + input_idx] = value;
    }
}
tensor* select_set_(tensor* a, tensor* idx, float value){
    return _launch_select(a, idx, value);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ kernels that do not have op wrappers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tensor* batched_transpose_k(tensor* a){
    if (CUDA_DEBUG) printf("[batched_transpose_k]\n");
    assert_input(a, 3);

    int B = a->shape[0], N = a->shape[1], M = a->shape[2];
    // todo: allocate empty
    tensor* out = Tensor(B, M, N);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(M/num_threads), ceil(N/num_threads), B);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda Transpose] grid: (%f, %f, %i)\n", ceil(M/num_threads), ceil(N/num_threads), B);
        printf("[cuda Transpose] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    TransposeKernel<<<dimGrid, dimBlock>>>(a->data, out->data, M, N, true);
    return out;
}
