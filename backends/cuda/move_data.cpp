
// comment:
// keep in mind that there's an asymmetry between copy_to_device (which
// actually overwrites t->data), and copy_to_host (which returns a new tensor)

void set_backend_cuda(void);

void copy_to_cuda(tensor* t){
    printf("copy_to_cuda\n");
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
        printf("[cuda malloc] error\n");
        exit(1);
    }
    // question-now: should I do contigify before cudaMemcpy?
    err = cudaMemcpy(t_device, t->data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("[cuda memcopy] error\n");
        exit(1);
    }
    // todo: free cpu t->data (currently memory leak)
    t->data = t_device;
}

tensor* copy_from_cuda(tensor* t) {
    printf("copy_from_cuda\n");
    t->device = CUDA;

    // todo: can just define a macro for print to call 4 lines below and then call the orignal print2d (no need for cuda_print_2d)
    cudaDeviceSynchronize();
    int size = t->size * sizeof(float);
    float* host_data = (float*)malloc(size);
    cudaError_t err = cudaMemcpy(host_data, t->data, size, cudaMemcpyDeviceToHost);
    // todo: define a macro CUDA_CHECK for unwrapping this
    if (err != cudaSuccess){
        printf("[cuda memcopy] error: %s\n",  cudaGetErrorString(err));
        exit(1);
    }
    // avoids TensorLike returning a cuda tensor (TensorLike->TensorNd->COPY_TO_DEVICE->copy_to_cuda)
    set_backend_cpu();
    tensor* t_copy = TensorLike(t);
    // todo: free t and t_copy->data, currently memory leak
    t_copy->data = host_data;
    set_backend_cuda();
    return t_copy;
}

void set_backend_cuda(void){
    COPY_TO_DEVICE = copy_to_cuda;
    COPY_FROM_DEVICE = copy_from_cuda;
}
