#include "../../nn.h"


// Reduce kernel isn't as straightforward as it seems, because naively it
// requires a lot of atomic operations (poor perf -- worse than CPU), so
// use parallel reduction pattern



// todo: the disadvantage of passing "value structs" around is that for reduce_sum idxs effectively do nothing, but do compute shared memory
// modified _launch_reduction_kernel to save idxs to be used for _bwd
struct __align__(8) value { // CUDA specific syntax needed to that this struct can be re-interpreted as ULL
    float val;
    int idx;
};

__device__ value max(value a, value b){
    return (a.val > b.val) ? a : b;
}

// for reduce_add, it doesn't make sense to propagate idxs (value->idx)
// through the reduction, but I still do it because want the code inside
// the ReductionKernel to work polymorphically (op_fns, atomic_fns) for
// any reduction op (so that _launch_reduction_kernel can be re-used for:
// reduce_max_k, reduce_sum_k, etc)
__device__ value add(value a, value b){
    // ues -1 as an invalid idx
    return value{a.val + b.val, -1};
}

typedef unsigned long long int atomic_t;

// Only compare old.val >= val.val instead of comparing the entire struct.
// When a larger value is found, the entire struct (including the idx) is atomically updated
__device__ value atomicMaxValue(value* address, value val) {
    bool done = false;
    value old;

    do {
        old = *address;

        // compare only the val fields
        if (old.val >= val.val) {
            // current value is larger or equal, no need to update
            break;
        }

        atomic_t* address_ull = (atomic_t*)address;
        atomic_t assumed_ull = *(atomic_t*)&old;
        atomic_t val_ull = *(atomic_t*)&val;

        atomic_t old_ull = atomicCAS(address_ull, assumed_ull, val_ull);
        done = (old_ull == assumed_ull);
        old = *(value*)&old_ull;

    } while (!done);

    // returns old by convention for atomic operations
    return old;
}

// mostly copy of atomicMaxValue
__device__ value atomicAddValue(value* address, value val) {
    bool done = false;
    value old;

    do {
        // whatever value at that address now
        old = *address;

        // if, at address_ull, find assumed_ull then overwrite it with val_ull
        // (which is the sum of assumed and val)
        val.val += old.val;

        atomic_t* address_ull = (atomic_t*)address;
        atomic_t assumed_ull = *(atomic_t*)&old;
        atomic_t val_ull = *(atomic_t*)&val;

        atomic_t old_ull = atomicCAS(address_ull, assumed_ull, val_ull);
        done = (old_ull == assumed_ull);

        // undo your modification to val if update failed
        // (update can fail because observed value was != assumed value)
        val.val -= old.val;

        old = *(value*)&old_ull;

    } while (!done);

    return old;
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
typedef value (*OpFn)(value, value);
typedef value (*AtomicFn)(value *, value);
__device__ OpFn op_fns[2] = {max, add};
__device__ AtomicFn atomic_fns[2] = {atomicMaxValue, atomicAddValue};


// features: convergent (active threads are close to each other), with privatization (shared memory), segmented (multi-block);
// scratch_space (used when ReductionKernel is used to compute reduce_max) to store idxs of max values, needed for the reduce_max_bwd and reduce_max_batched_bwd
__global__ void ReductionKernel(int op_idx, float* input, value* out, int stride_to_next_b){

    // recover is_batched from stride_to_next_b to avoid passing 2
    // arguments (is_batched, stride_to_next_b) which mostly mean the same thing
    bool is_batched = stride_to_next_b ? true : false;
    // cancels out when not batched
    unsigned int batch = blockIdx.y * is_batched;

    // variable t is the start in the shared array, variable "start" is the start in the input array
    unsigned int t = threadIdx.x;

    // todo-now:
    // Add guards to the kernel so that if the input is smaller than 2*NUM_THREADS, it should not access it's these locations;
    // Even non-batched version of this kernel has this bug; remove exit from the stub

    // *** privatization ***

    // each thread block takes 2*BlockDim.x input elements
    // note: Dynamic allocations happen at kernel invocation
    // __shared__ float partialSum[2*blockDim.x];
    __shared__ value partialSum[2*NUM_THREADS];
    // thus when we multiply the size of each input segment by blockIdx.x
    // of a block, we have the starting location of the segment to be
    // processed by the block
    unsigned int start = batch*stride_to_next_b + 2*blockIdx.x*blockDim.x;

    // each thread loads 2 elements into shared memory
    //  set thread_idx in the shared array to start_idx+thread_idx in the global array

    // initialize value struct, with the index of that value in the input tensor -- needs to be idx of input tensor not local tensor (partial_sum)
    partialSum[t] = {input[start+t], start+t};
    partialSum[t + blockDim.x] = {input[start+t + blockDim.x], start+t + blockDim.x};

    if (CUDA_DEBUG){
        printf("[ReductionKernel] batch: %i\n", batch);
        printf("[ReductionKernel blockIdx.y=%i] partialSum[t=%i] = input[(start + t)=%i];\n", blockIdx.y, t, start+t);
        printf("[ReductionKernel blockIdx.y=%i] partialSum[(t + blockDim.x)=%i] = input[(start+t + blockDim.x)=%i];\n", blockIdx.y, t + blockDim.x, start+t + blockDim.x);
    }

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

    But then extended to propagate value structs though the reduction kernel, and it's not clear
    how to use atomics to reduce two value structs. So ended up pass array of b*num_blocks to back
    to the stub and do the reduction across the blocks there. Perform atomic reductions on custom
    structures -- bc I want to update two adjacent 32-bit items, can use a generalized 64-bit atomic
    operation, treat the entire struct as a single 64-bit value (unsigned long long)
    */
    if (t==0){
        atomic_fns[op_idx](&out[batch], partialSum[t]);
    }
}

tensor* _launch_reduction_kernel(int op_idx, tensor* input, bool is_batched){
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

    float num_threads = (float)NUM_THREADS;
    int num_blocks, B, stride_to_next_b;
    tensor* out;
    if (!is_batched){
        B = 1;
        // used as (bool is_batch) inside the kernel
        stride_to_next_b = 0;

        out = TensorScalarFill(fill_value);
        // each thread block consumes num_threads*2 inputs
        num_blocks = ceil(input->size/(num_threads*2));
        if (input->size < num_threads*2){
            printf("[temporary] shape err: %i(input->size) < %i(num_threads*2)\n", input->size, (int)num_threads*2);
            exit(1);
        }
    } else {
        B = input->shape[0];
        stride_to_next_b = input->stride[0];

        // defining the second tensor is needed so that you can use Fill (currently I don't
        // conveniently support initializing non 1d tensor and filling it);
        // This isn't necessary bc you overwrite ->data inside the Kernel anyway
        // out = TensorLikeFill(Tensor(B, 1), fill_value);
        out = Tensor(B, 1);

        // In non-batched kernel each thread block consumes num_threads*2 inputs;
        // Additionally, divide by B so that each block doesn't see all the elements
        // (input->size), but instead it only sees elements in a single batch element (b);
        // And because you have B as additional dim of the grid -- these elements
        // will still be covered (but by blocks with different blockIdx.y)
        num_blocks = ceil(input->size/B/(num_threads*2));
        if (input->shape[1] < num_threads*2){
            printf("[temporary] shape err:  %i(input->shape[1]) < %i(num_threads*2)\n", input->shape[1], (int)num_threads*2);
            exit(1);
        }
    }

    dim3 dimGrid(num_blocks, B, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda reduction_kernel] grid: (%i, %i, 1)\n", num_blocks, B);
        printf("[cuda reduction_kernel] block: (%f, 1, 1)\n", num_threads);
        printf("stride_to_next_b: %i\n", stride_to_next_b);
    }

    // copy to cuda, then copy out back to cpu -- maybe a better solution is to launch 2nd kernel (to do the work in reduce_max_bwd)/
    value* out_device; // (B, 1)
    int size = B * sizeof(value);
    checkCudaErrors(cudaMalloc((void**)&out_device, size));

    ReductionKernel<<<dimGrid, dimBlock>>>(op_idx, input->data, out_device, stride_to_next_b);

    // copy out back to cpu
    value* out_host = (value*)malloc(size);
    checkCudaErrors(cudaMemcpy(out_host, out_device, size, cudaMemcpyDeviceToHost));

    // copy to host because below I modify its ->data attribute
    tensor* scratch = COPY_FROM_DEVICE(TensorLike(out));
    out = COPY_FROM_DEVICE(out);

    // separate members of the struct into the two tensors
    for (int b=0; b<B; b++){
        out->data[b] = out_host[b].val;
        // convert global input idx (in range 0-B*N) into per b idx (in range 0-N),
        // bc select_set_ in reduce_max_bwd expects each element of the idx to be in range 0-N
        scratch->data[b] = (float)(out_host[b].idx - b*input->stride[0]);
        // printf("[idx] b=%i, val=%f, idx=%i\n", b, out_host[b].val, out_host[b].idx);
    }

    COPY_TO_DEVICE(scratch);
    out->scratch_space[0] = scratch;

    COPY_TO_DEVICE(out);
    return out;
}

tensor* reduce_max_k(tensor* a){
    if (CUDA_DEBUG) printf("[reduce_max_k]\n");
    return _launch_reduction_kernel(0, a, false);
}

tensor* reduce_sum_k(tensor* a){
    if (CUDA_DEBUG) printf("[reduce_sum_k]\n");
    return _launch_reduction_kernel(1, a, false);
}

tensor* batched_reduce_max_k(tensor* a){
    if (CUDA_DEBUG) printf("[batched_reduce_max_k]\n");
    return _launch_reduction_kernel(0, a, true);
}

tensor* batched_reduce_sum_k(tensor* a){
    if (CUDA_DEBUG) printf("[batched_reduce_sum_k]\n");
    return _launch_reduction_kernel(1, a, true);
}
