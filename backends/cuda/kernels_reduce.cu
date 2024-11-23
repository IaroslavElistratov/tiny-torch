#include "../../nn.h"


// Reduce kernel isn't as straightforward as it seems, because naively it
// requires a lot of atomic operations (poor perf -- worse than CPU), so
// use parallel reduction pattern


// https://github.com/treecode/Bonsai/blob/581fa8e70501ce85660c7eac0d61c0e5c5bece4a/runtime/profiling/derived_atomic_functions.h#L199C1-L209C2
__device__ float atomicMax(float *address, float val){
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret)){
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}


// todo-high: rm
__device__ float add(float a, float b){
    return a+b;
}

/*
more generic form to support different reduce funcs with the same kernel

Prior to that tried pointers to the kernel:
    __global__ void ReductionKernel(ReductionFn reduction_fn, AtomicFn atomic_fn, float* input, float* out)

    _launch_reduction_kernel(...)
        ReductionKernel<<<dimGrid, dimBlock>>>(reduction_fn, atomic_fn, input->data, out->data)

    tensor* reduce_max_k(tensor* a)
        return _launch_reduction_kernel(deviceMax, atomicMax, a);

Resulted in err below, likely because the function pointers passed were allocated on CPU (all inputs to the kernel should be allocated on the device)
    CUDA Exception: Warp Invalid PC; Error: Could not read error PC  (dev=0, sm=0, wp=0), error=CUDBG_ERROR_UNKNOWN_FUNCTION(0x3).
    (cuda-gdb)  Program terminated with signal CUDA_EXCEPTION_8, Warp Invalid PC.
*/
typedef float (*OpFn)(float, float);
typedef float (*AtomicFn)(float *, float);
__device__ OpFn op_fns[3] = {max, add};
__device__ AtomicFn atomic_fns[3] = {atomicMax, atomicAdd};


// features: convergent (active threads are close to each other), with privatization (shared memory), segmented (multi-block)
__global__ void ReductionKernel(int op_idx, float* input, float* out){

    // variable t is the start in the shared array, variable "start" is the start in the input array
    unsigned int t = threadIdx.x;

    // *** privatization ***

    // each thread block takes 2*BlockDim.x input elements
    // note: Dynamic allocations happen at kernel invocation
    // __shared__ float partialSum[2*blockDim.x];
    __shared__ float partialSum[2*NUM_THREADS];
    // thus when we multiply the size of each input segment by blockIdx.x
    // of a block, we have the starting location of the segment to be
    // processed by the block
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // each thread loads 2 elements into shared memory
    //  set thread_idx in the shared array to start_idx+thread_idx in the global array
    partialSum[t] = input[start+t];
    partialSum[t + blockDim.x] = input[start+t + blockDim.x];

    // *** convergent reduction kernel ***

    // in each step, one of the inputs comes from an increasing distance away
    //  much better for SM's warp utilization to stride/=2 rather than stride/=2
    for (unsigned int stride=blockDim.x; stride>0; stride/=2){
        // after each step, half of the threads are no longer needed;
        // each thread is responsible for an  index location in the partial sum vector ("location of responsibility");
        // tests for wether the thread should be active
        if (t < stride){
            partialSum[t] = op_fns[op_idx](partialSum[t], partialSum[t+stride]);
        }
        __syncthreads();
    }

    // once we finish execution of the for loop for all the threads, we have entire
    // section (2*num_threads in this block) of the input vector reduced to one value,
    // so at the end of the kernel only need to have one thread to write the sum of the
    // thread block into a new vector (to be index with blockIdx.x);
    // In this new vector every element is a partial sum produced by one thread block

    /*
    An alternative way to get final value is instead of using Atomic functions at the end, can
    pass output array of shape num_blocks, then one thread from each block can write the partial
    sum (for the block) into this array indexing that array with its BlockIdx, then have a cpu
    short loop in the stub to sum these per-block partial sums into a final value.

    For now preferred having atomics as it simplifies things by limiting all computations to the
    kernel function (wt needing additional logic in the stub)
    */
    if (t==0){
        // question-now: float atomic add doesn't work for shared memory -- https://github.com/treecode/Bonsai/blob/581fa8e70501ce85660c7eac0d61c0e5c5bece4a/runtime/profiling/derived_atomic_functions.h#L14-L17
        atomic_fns[op_idx](out, partialSum[t]);
    }
}


tensor* _launch_reduction_kernel(int op_idx, tensor* input){
    // todo:
    // unary_input_checks(input);

    float fill_value;
    // todo-high: ugly
    if (op_idx==0){
        fill_value = -100.0;
    } else if (op_idx==1){
        fill_value = 0.0;
    } else {
        printf("[cuda reduction_kernel] unsupported op_idx\n");
        exit(1);
    }
    tensor* out = TensorScalarFill(fill_value);

    float num_threads = (float)NUM_THREADS;
    // each thread block consumes num_threads*2 inputs
    int num_blocks = ceil(input->size/(num_threads*2));
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda reduction_kernel] grid: (%i, 1, 1)\n", num_blocks);
        printf("[cuda reduction_kernel] block: (%f, 1, 1)\n", num_threads);
    }

    ReductionKernel<<<dimGrid, dimBlock>>>(op_idx, input->data, out->data);
    return out;
}


tensor* reduce_max_k(tensor* a){
    if (CUDA_DEBUG) printf("[reduce_max_k]\n");
    return _launch_reduction_kernel(0, a);
}

tensor* reduce_sum_k(tensor* a){
    if (CUDA_DEBUG) printf("[reduce_sum_k]\n");
    return _launch_reduction_kernel(1, a);
}




struct value {
    float val;
    unsigned int idx;
};

__device__ value max_with_idxs(value a, value b){
    return (a.val > b.val) ? a : b;
}

// mostly copy from ReductionKernel
__global__ void MaxBwdKernel(float* input, value* out){
    unsigned int t = threadIdx.x;
    __shared__ value partialSum[2*NUM_THREADS];
    unsigned int start = 2*blockIdx.x*blockDim.x;
    partialSum[t] = value{input[start+t], start+t};
    partialSum[t + blockDim.x] = value{input[start+t + blockDim.x], start+t + blockDim.x};

    for (unsigned int stride=blockDim.x; stride>0; stride/=2){
        if (t < stride){
            partialSum[t] = max_with_idxs(partialSum[t], partialSum[t+stride]);
        }
        __syncthreads();
    }
    // comment:
    // unclear how to make this atomic w idx -- can rm it and make a loop in the stub
    if (t==0){
        out[blockIdx.x] = partialSum[t];
    }
}

int _launch_max_bwd_kernel(tensor* input){
    // todo:
    // unary_input_checks(input);

    float num_threads = (float)NUM_THREADS;
    int num_blocks = ceil(input->size/(num_threads*2));
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda _launch_max_bwd_kernel] grid: (%i, 1, 1)\n", num_blocks);
        printf("[cuda _launch_max_bwd_kernel] block: (%f, 1, 1)\n", num_threads);
    }

    // comment:
    // copy to cuda, then copy out back to cpu -- maybe a better solution is to launch 2nd kernel (to do the work in reduce_max_bwd)/
    value* out_device;
    int size = num_blocks * sizeof(value);
    cudaError_t err = cudaMalloc((void**)&out_device, size);
    if (err != cudaSuccess){
        printf("[cuda malloc] error: %s\n",  cudaGetErrorString(err));
        exit(1);
    }

    MaxBwdKernel<<<dimGrid, dimBlock>>>(input->data, out_device);

    // copy out back to cpu
    value* out_host = (value*)malloc(size);
    err = cudaMemcpy(out_host, out_device, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("[cuda memcopy] error: %s\n",  cudaGetErrorString(err));
        exit(1);
    }

    // aggregate values from multiple blocks into a single value
    // use this because not sure how to use max_with_idxs with atomics
    // (to be used inside the kernel)
    value max = out_host[0];
    for (int i=1; i<num_blocks; i++){
        value curr = out_host[i];
        printf("CURR value: %f", curr.val);
        // can't re-use max_with_idxs since it's a device function, but this is host code (not inside the kernel)
        max = (max.val > curr.val) ? max : curr;
    }

    return max.idx;
}

// todo-high: too many copy_to/from
void reduce_max_bwd(tensor* upstream, tensor* out){
    tensor* a = out->inputs[0];
    if (CUDA_DEBUG) printf("[reduce_max_bwd_k]\n");
    int idx = _launch_max_bwd_kernel(a);
    // printf("IDX: %i", idx);
    tensor* local = TensorLikeFill(a, 0.0);

    // copy to cpu before accessing t->data
    tensor* upstream_host = COPY_FROM_DEVICE(upstream);
    tensor* broadcasted_upstream = TensorLikeFill(a, upstream_host->data[0]);

    tensor* local_host = COPY_FROM_DEVICE(local);
    local_host->data[idx] = 1.0;
    COPY_TO_DEVICE(local_host); // semantically this is "local"

    a->grad = mul_k(local_host, broadcasted_upstream);
}
