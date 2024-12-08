
inline void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess){
        printf("[cuda malloc/memcopy] error: %s\n",  cudaGetErrorString(err));
        exit(1);
    }
}

void set_backend_cuda(void);

void copy_to_cuda(tensor* t){
    if (DATA_COPY_DEBUG) printf("copy_to_cuda\n");
    if (t->device==CUDA){
        return;
    }

    assert_contiguous(t);
    t->device = CUDA;

    float* t_device;
    int size = t->size * sizeof(float);
    checkCudaErrors(cudaMalloc((void**)&t_device, size));
    checkCudaErrors(cudaMemcpy(t_device, t->data, size, cudaMemcpyHostToDevice));
    // todo: free cpu t->data
    t->data = t_device;
}

tensor* copy_from_cuda(tensor* t) {
    if (DATA_COPY_DEBUG) printf("copy_from_cuda\n");
    assert_contiguous(t);
    t->device = CUDA;

    cudaDeviceSynchronize();
    int size = t->size * sizeof(float);
    float* host_data = (float*)malloc(size);
    checkCudaErrors(cudaMemcpy(host_data, t->data, size, cudaMemcpyDeviceToHost));

    // avoids TensorLike returning a cuda tensor (TensorLike->TensorNd->COPY_TO_DEVICE->copy_to_cuda)
    set_backend_cpu();
    tensor* t_copy = TensorLike(t);
    // todo: free t and t_copy->data
    t_copy->data = host_data;
    set_backend_cuda();

    return t_copy;
}

void set_backend_cuda(void){
    COPY_TO_DEVICE = copy_to_cuda;
    COPY_FROM_DEVICE = copy_from_cuda;
}
