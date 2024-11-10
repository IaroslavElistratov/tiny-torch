
void set_backend_cuda(void);

void copy_to_cuda(tensor* t){
    printf("\ncopy_to_cuda\n");
    if (t->device==CUDA){
        return;
    }
    // not needed for copying data itself,
    // but need it bc this fn can be called
    // from inside a tensor constructor --
    // in which case it will set this member
    t->device = CUDA;

    float* t_device;
    int size = t->size * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&t_device, size);
    // todo: exit from program everywhere in case of error
    if (err != cudaSuccess){
        printf("[cuda malloc] error");
    }
    err = cudaMemcpy(t_device, t->data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("[cuda memcopy] error");
    }
    // todo: free cpu t->data (currently memory leak)
    t->data = t_device;
}

tensor* copy_from_cuda(tensor* t) {
    printf("\ncopy_from_cuda\n");
    t->device = CUDA;

    // todo: can just define a macro for print to call 4 lines below and then call the orignal print2d (no need for cuda_print_2d)
    cudaDeviceSynchronize();
    int size = t->size * sizeof(float);
    float* host_data = (float*)malloc(size);
    cudaError_t err = cudaMemcpy(host_data, t->data, size, cudaMemcpyDeviceToHost);
    // todo: define a macro CUDA_CHECK for unwrapping this
    if (err != cudaSuccess){
        printf("[cuda memcopy] error: %s",  cudaGetErrorString(err));
        return NULL;
    }

    // Need to update strides to contiguous (bc if workflow was: original strided tensor -> to_device -> to_host -- t->data is no longer strided, it's contiguous)
    //      but the strides are outdated and do not reflect this (they contain old values for the strides of the original strided tensor)
    //      can be problematic, bc most of my loops (over the t->data) use strides to access and element -- but now strides are wrong (outdated)
    //  But bc you're not setting t->data to host_data (t->data is the original unchanged data) and it doesn't make sense to modify strides on the original tensor
    //      - it seems can solve by actually setting t->data=host_data (and the contigify the t's strides), but I don't want to do this bc I don't want calls to e.g. print(t)
    //        move data to cpu, which will then necessitate another copy_to_device if I want to run another operation on the "t"
    //      - another solution is to return (from this fn) a new one-off (meaning will be used only by the func which called copy_host e.g. print) tensor with correct strides
    //          - this solves both problems: I avoid overwritting t->data of the original tensor; yet I return a thing with correct strides from this fn -- so that all indexing
    //            (which uses strides) will work correctly. Wt needing to add new hacks to e.g. print to use "at" when t is on CPU, and use other indexing which assumes contiguous
    //            strides when the tensor was just copied from CUDA
    // return host_data;

    // avoid this calling CUDA's TensorLike? I think this recurses infinitely
    //  The problem is that within my library impl sometimes I want TensorLike to be the device tensor (e.g. in AG, in backend/common) other-time I want it to refer to particualr subclass e.g. CPU tensor (like here)
    //  Well if here is the only case when I want to use the the opposite of deivce -- then maybe can just use some hack (like directly calling TensorNoDataNd avoiding the need to call the callback)

    set_backend_cpu();
    tensor* t_copy = TensorLike(t);
    // todo: free t and t_copy->data, currently memory leak
    t_copy->data = host_data;
    set_backend_cuda();
    return t_copy;
}

void set_backend_cuda(void){
    extern void (*COPY_TO_DEVICE)(tensor*);
    extern tensor* (*COPY_FROM_DEVICE)(tensor*);
    COPY_TO_DEVICE = copy_to_cuda;
    COPY_FROM_DEVICE = copy_from_cuda;
}
