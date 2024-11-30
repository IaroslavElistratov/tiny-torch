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
    value old, sum;

    do {
        // whatever value at that address now
        old = *address;

        // if, at address_ull, find assumed_ull then overwrite it with val_ull
        // (which is the sum of assumed and val)

        // has semantics of undoing modification to val if update failed
        // (update can fail because observed value was != assumed value)
        sum = add(old, val);

        atomic_t* address_ull = (atomic_t*)address;
        atomic_t assumed_ull = *(atomic_t*)&old;
        atomic_t val_ull = *(atomic_t*)&sum;

        atomic_t old_ull = atomicCAS(address_ull, assumed_ull, val_ull);
        done = (old_ull == assumed_ull);

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
__global__ void ReductionKernel(int op_idx, float* input, value* out, int stride_to_next_b, int single_example_len, int len_padding){

    // recover is_batched from stride_to_next_b to avoid passing 2
    // arguments (is_batched, stride_to_next_b) which mostly mean the same thing
    bool is_batched = stride_to_next_b ? true : false;
    // cancels out when not batched
    unsigned int batch = blockIdx.y * is_batched;

    // variable t is the start in the shared array, variable "start" is the start in the input array
    unsigned int t = threadIdx.x;

    // *** privatization ***

    // each thread block takes 2*BlockDim.x input elements
    // note: Dynamic allocations happen at kernel invocation
    // __shared__ float partialSum[2*blockDim.x];
    // answer-now: notice this uses the value of the macro, and not the actual runtime value for the number of threads which the kernel was launched for (see _launch_reduction_kernel) -- and the value of the NUM_THREADS macro is larger or equal to runtime value for the dimentinality of blocks, this means the shared memory array (partial sum) can be larger than needed, but that's ok becuase I have logic in this kernel which makes sure tha these values (larger than actual runtiem blockDim.x) is not used
    __shared__ value partialSum[2*NUM_THREADS];
    // thus when we multiply the size of each input segment by blockIdx.x
    // of a block, we have the starting location of the segment to be
    // processed by the block
    unsigned int start = batch*stride_to_next_b + 2*blockIdx.x*blockDim.x;

    // each thread loads 2 elements into shared memory
    //  set thread_idx in the shared array to start_idx+thread_idx in the global array

    // alternatively fill them with -inf (for max) and 0 (for sum) values
    float fill_value = op_idx == 0 ? -100.0 : 0.0;
    // initialize value struct, with the index of that value in the input tensor -- needs to
    // be idx of input tensor not local tensor (partial_sum);
    // no need to check for "&&(start+t)<next_power"
    if ((start+t)>=(batch*single_example_len + single_example_len)){
        // if (CUDA_DEBUG) printf("if! b: %i, start+t: %i\n", batch, start+t);
        partialSum[t] = {fill_value, -1};
    } else {
        // if (CUDA_DEBUG) printf("else! b: %i, start+t: %i\n", batch, start+t);
        partialSum[t] = {input[start+t], start+t};
    }
    // "+ single_example_len" because when batch=0, the max num elements should be "single_example_len",
    // and not 0 (which it would be if multiplied by batch=0)
    if ((start+t + blockDim.x)>=(batch*single_example_len + single_example_len)){
        // if (CUDA_DEBUG) printf("if! b: %i, t+blockDim.x: %i\n", batch,t  + blockDim.x);
        partialSum[t + blockDim.x] = {fill_value, -1};
    } else {
        // if (CUDA_DEBUG) printf("else! b: %i, t+blockDim.x: %i\n", batch,t  + blockDim.x);
        partialSum[t + blockDim.x] = {input[start+t + blockDim.x], start+t + blockDim.x};
    }

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
            if (CUDA_DEBUG) printf("(partialSum[%i] = partialSum[%i], partialSum[%i]) = %f\n", t, t, t+stride, partialSum[t].val);
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
        if (CUDA_DEBUG) printf("[t==0] partialSum[t].val: %f\n", partialSum[t].val);
        atomic_fns[op_idx](&out[batch], partialSum[t]);
        if (CUDA_DEBUG) printf("[t==0] out[batch=%i].val: %f\n", batch, out[batch].val);

    }
}

int round_to_power_of_2(int x){
    int power = 1;
    while (power < x){
        power *= 2;
    }
    return power;
}

tensor* _launch_reduction_kernel(int op_idx, tensor* input, bool is_batched){

    // regardless batched or not
    assert_input(input, 2);

    float fill_value;
    // todo-high: ugly, but the teaching kit says to initialize to the smallest possible value
    // on the system (mathematically:  -inf) -- use "-INFINITY"
    if (op_idx==0){
        fill_value = -100.0;
    } else if (op_idx==1){
        fill_value = 0.0;
    } else {
        printf("[cuda reduction_kernel] unsupported op_idx\n");
        exit(1);
    }

    // for cases where input is smaller than value of the NUM_THREADS macro
    int single_example_len = is_batched ? input->shape[1] : input->size;

    // one approach is to only support inputs of power of 2:
    // make sure input shape is a power of two (because in the kernel will be repeatedly dividing it by 2)
    // each of the strides should be even as well -- otherwise incorrect result
    // if (ceil(log2(single_example_len)) != floor(log2(single_example_len))){
    //     printf("[cuda reduce] Expected size of a single reduction array to be a power of 2, saw: %i\n", single_example_len);
    //     exit(1);
    // }

    // another approach is round to the nearest power of 2 to support e.g. (B, 10) inputs,
    // need a power of 2 bc in the kernel loop will be repeatedly dividing by 2
    int power = round_to_power_of_2(single_example_len);

    int num_threads_for_single_example = power/2;

    // todo: i think this line is unnecessary
    float num_threads = (float)min(NUM_THREADS, num_threads_for_single_example);
    int num_blocks, B, stride_to_next_b;
    tensor* out;
    if (!is_batched){
        B = 1;
        // used as (bool is_batch) inside the kernel
        stride_to_next_b = 0;

        out = TensorScalarFill(fill_value);
        // each thread block consumes num_threads*2 inputs
        //  e.g: ceil(16 / num_threads(16) * 2) = ceil(16 / 32) = ceil(0.5) = 1
        num_blocks = ceil(input->size/(num_threads*2));
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
    }
    int len_padding = power - single_example_len;

    dim3 dimGrid(num_blocks, B, 1);
    dim3 dimBlock(num_threads, 1, 1);

    if (CUDA_DEBUG){
        printf("[cuda reduction_kernel] grid: (%i, %i, 1)\n", num_blocks, B);
        printf("[cuda reduction_kernel] block: (%f, 1, 1)\n", num_threads);
        printf("stride_to_next_b: %i\n", stride_to_next_b);
    }

    // allocate output buffer

    value* out_device; // (B, 1)
    int size = B * sizeof(value);
    checkCudaErrors(cudaMalloc((void**)&out_device, size));
    // fill the out_value -- bc atomic function max checks to see if the value we're about to record is smaller than existing
    // value in out_device (and current existing value there is 0.0 -- so it can skip doing the atomic function entirely)
    //
    // fill the inputs with fill_values -- otherwise currently incorrect max of negative input values! Bc in atomicMaxValue,
    // I don't perform atomicMax if existing value at the output pointer is larger than the value of the input to atomicMax,
    // but by default out_device is initialized to 0.0, but smallest number in my tensor was -0.12 -- so it mistakenly just kept the
    // default value because it was smaller;
    value out_cpu[B];
    for (int i=0; i<B; i++){
        out_cpu[i].val = fill_value;
    }
    checkCudaErrors(cudaMemcpy(out_device, out_cpu, size, cudaMemcpyHostToDevice));

    // launch

    ReductionKernel<<<dimGrid, dimBlock>>>(op_idx, input->data, out_device, stride_to_next_b, single_example_len, len_padding);

    // separate members of the struct into the two tensors

    // copy to host because below I modify its ->data attribute
    // maybe a better solution is to launch 2nd kernel to do the work?
    value* out_host = (value*)malloc(size);
    checkCudaErrors(cudaMemcpy(out_host, out_device, size, cudaMemcpyDeviceToHost));

    tensor* scratch = COPY_FROM_DEVICE(TensorLike(out));
    out = COPY_FROM_DEVICE(out);

    for (int b=0; b<B; b++){
        out->data[b] = out_host[b].val;
        // convert global input idx (in range 0-B*N) into per b idx (in range 0-N),
        // bc select_set_ in reduce_max_bwd expects each element of the idx to be in range 0-N
        scratch->data[b] = (float)(out_host[b].idx - b*input->stride[0]);
        if (CUDA_DEBUG) printf("[idx] b=%i, val=%f, idx=%i\n", b, out_host[b].val, out_host[b].idx);
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
