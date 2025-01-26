
// comment:
// keep in mind that there's an asymmetry between copy_to_device (which
// actually overwrites t->data), and copy_to_host (which returns a new tensor)

// this wrapper macro is needed to pass __FILE__, __LINE__ to the fn,
// because macro (unlike fn) gets expanded at the call sight
#define checkCudaErrors(ans) {_checkCudaErrors((ans), __FILE__, __LINE__);}
inline void _checkCudaErrors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess){
        fprintf(stderr,"[checkCudaErrors] error: %s %s %d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}

// #define checkCudaErrors(msg) \
//     do{\
//         cudaError_t __err = cudaGetLastError(); \
//         if (__err != cudaSuccess) { \
//             fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
//                 msg, cudaGetErrorString(__err), \
//                 __FILE__, __LINE__); \
//             fprintf(stderr, "*** FAILED - ABORTING ***\n"); \
//             exit(1); \
//         } \
//     } while (0)


void set_backend_cuda(void);

void copy_to_cuda(tensor* t){
    if (DATA_COPY_DEBUG) printf("copy_to_cuda\n");
    if (t->device == CUDA){
        return;
    }
    if (t->device != CPU){
        printf("[copy_to_cuda] expected device CPU\n");
        exit(1);
    }

    // question-now: should I do contigify before cudaMemcpy?
    assert_contiguous(t);

    // not needed for copying data itself,
    // but need it bc this fn can be called
    // from inside a tensor constructor --
    // in which case it will set this member

    float* t_device;
    int size = t->size * sizeof(float);
    checkCudaErrors(cudaMalloc((void**)&t_device, size));
    checkCudaErrors(cudaMemcpy(t_device, t->data, size, cudaMemcpyHostToDevice));

    // should free (even assuming that "t" is in the GC list) bc the GC stores pointers
    // to Tensors, not to their underlying data -- so when data got changed the GC
    // does not see this
    free(t->data);
    t->data = t_device;
    t->device = CUDA;
}


tensor* copy_from_cuda(tensor* t) {
    if (DATA_COPY_DEBUG) printf("copy_from_cuda\n");
    if (t->device == CPU){
        return t;
    }
    if (t->device != CUDA){
        printf("[copy_from_cuda] expected device CUDA\n");
        exit(1);
    }
    assert_contiguous(t);

    // todo: can just define a macro for print to call 4 lines below and then call the orignal print2d (no need for cuda_print_2d)
    cudaDeviceSynchronize();
    int size = t->size * sizeof(float);
    float* host_data = (float*)checkMallocErrors(malloc(size));
    checkCudaErrors(cudaMemcpy(host_data, t->data, size, cudaMemcpyDeviceToHost));
    // avoids TensorLike returning a cuda tensor (TensorLike->TensorNd->COPY_TO_DEVICE->copy_to_cuda)

    // no need to set backend to CPU (before constructing the tensor)
    // and then back, because TensorLikeNoData (unlike TensorLike) does
    // NOT invoke COPY_FROM_DEVICE
    tensor* t_copy = TensorLikeNoData(t);
    t_copy->data = host_data;
    t_copy->device = CPU;

    // note: no need to free "t", assuming it was created with tensor constructor then it's already traced by the GC array
    add_to_gc(t_copy);

    return t_copy;
}

void set_backend_cuda(void){
    COPY_TO_DEVICE = copy_to_cuda;
    COPY_FROM_DEVICE = copy_from_cuda;
}
