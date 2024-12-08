#include "../../nn.h"



// the disadvantage of passing "value structs" around is that for reduce_sum idxs effectively do nothing, but do compute shared memory
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


typedef value (*OpFn)(value, value);
typedef value (*AtomicFn)(value *, value);
__device__ OpFn op_fns[2] = {max, add};
__device__ AtomicFn atomic_fns[2] = {atomicMaxValue, atomicAddValue};


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
    __shared__ value partialSum[2*NUM_THREADS];
    // thus when we multiply the size of each input segment by blockIdx.x
    // of a block, we have the starting location of the segment to be
    // processed by the block
    unsigned int start = batch*stride_to_next_b + 2*blockIdx.x*blockDim.x;

    // each thread loads 2 elements into shared memory
    //  set thread_idx in the shared array to start_idx+thread_idx in the global array

    // todo : fill them with -inf (for max) and 0 (for sum) values
    float fill_value = op_idx == 0 ? -100.0 : 0.0;

    // initialize value struct, with the index of that value in the input tensor
    if ((start+t)>=(batch*single_example_len + single_example_len)){
        partialSum[t] = {fill_value, -1};
    } else {
        partialSum[t] = {input[start+t], start+t};
    }

    if ((start+t + blockDim.x)>=(batch*single_example_len + single_example_len)){
        partialSum[t + blockDim.x] = {fill_value, -1};
    } else {
        partialSum[t + blockDim.x] = {input[start+t + blockDim.x], start+t + blockDim.x};
    }

    // *** convergent reduction kernel ***

    // in each step, one of the inputs comes from an increasing distance away
    //  much better for SM's warp utilization to stride/=2 rather than stride/=2
    for (unsigned int stride=blockDim.x; stride>0; stride/=2){
        // after each step, half of the threads are no longer needed;
        // each thread is responsible for an  index location in the partial
        // sum vector ("location of responsibility");
        // tests for wether the thread should be active
        if (t < stride){
            partialSum[t] = op_fns[op_idx](partialSum[t], partialSum[t+stride]);
        }
        __syncthreads();
    }

    /*
    Once we finish execution of the for loop for all the threads, we have entire
    section (2*num_threads in this block) of the input vector reduced to one value,
    so at the end of the kernel only need to have one thread to write the sum of the
    thread block into a new vector (to be index with blockIdx.x);
    In this new vector every element is a partial sum produced by one thread block
    */
    if (t==0){
        atomic_fns[op_idx](&out[batch], partialSum[t]);

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
    // todo : use "-INFINITY"
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

    // another approach is round to the nearest power of 2 to support e.g. (B, 10) inputs,
    // need a power of 2 bc in the kernel loop will be repeatedly dividing by 2
    int power = round_to_power_of_2(single_example_len);
    int num_threads_for_single_example = power/2;

    float num_threads = (float)min(NUM_THREADS, num_threads_for_single_example);
    int num_blocks, B, stride_to_next_b;
    tensor* out;
    if (!is_batched){
        B = 1;
        stride_to_next_b = 0;
        out = TensorScalarFill(fill_value);
        num_blocks = ceil(input->size/(num_threads*2));
    } else {
        B = input->shape[0];
        stride_to_next_b = input->stride[0];
        out = Tensor(B, 1);
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

    // fill the out_value (atomic function max checks)

    value out_cpu[B];
    for (int i=0; i<B; i++){
        out_cpu[i].val = fill_value;
    }
    checkCudaErrors(cudaMemcpy(out_device, out_cpu, size, cudaMemcpyHostToDevice));

    // launch

    ReductionKernel<<<dimGrid, dimBlock>>>(op_idx, input->data, out_device, stride_to_next_b, single_example_len, len_padding);

    // separate members of the struct into the two tensors

    value* out_host = (value*)malloc(size);
    checkCudaErrors(cudaMemcpy(out_host, out_device, size, cudaMemcpyDeviceToHost));

    tensor* scratch = COPY_FROM_DEVICE(TensorLike(out));
    out = COPY_FROM_DEVICE(out);

    for (int b=0; b<B; b++){
        out->data[b] = out_host[b].val;
        // convert global input idx (in range 0-B*N) into per b idx (in range 0-N),
        // bc select_set_ in reduce_max_bwd expects each element of idx in range 0-N
        scratch->data[b] = (float)(out_host[b].idx - b*input->stride[0]);
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
