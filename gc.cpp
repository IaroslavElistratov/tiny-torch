void checkCudaErrors(cudaError_t err);

// todo-high:
//  for cpu backend, the number of tensors created, quickly reaches MAX_GC: e.g. 5,041,911 for BS=128
//  set MAX_GC to 2**23 (8,388,608)?
//
// the ternary operator is not evaluated at pre-processor time
// #define MAX_GC (DEVICE == CUDA) ? 1024 : 2**23
#if DEVICE == CUDA
    #define MAX_GC 1024
#elif DEVICE == CPU
    #define MAX_GC 8388608
#endif

tensor* GC[MAX_GC];
int GC_IDX = -1;

void add_to_gc(tensor* t){
    if (GC_IDX+1 >= MAX_GC){
        printf("Error: max GC len reached\n");
        exit(1);
    }
    GC[++GC_IDX] = t;
    // note: bc add_to_gc runs before set_name, it's not meaningful to print t->name here (since it's a default random name)
    // printf("[add_to_gc] GC_IDX: %i\n", GC_IDX);
}

void free_tensor(tensor* t){
    if (t == NULL){
        // printf("tensor has already been freed: cannot free\n");
        return;
    }
    if (t->device != CPU && t->device != CUDA){
        printf("[free_tensor] unexpected device for tensor %s\n", t->name);
        exit(1);
    }

    // *** free cuda memory ***

    // t->grad was also created with a tensor constructor, so it was separately added
    // to the GC list thus should not free t->grad here (as part of freeing t), bc otherwise
    // when GC reaches t->grad and tries to free it -- will result in a double free
    //
    // similarly, don't free t->inputs -- assume they tensors which were separately created by
    // calling a tensor constructor, which means they are already in the GC list
    //
    // same for t->scratch_space

    // *** free device data ***

    // if (DEBUG){
    //     printf("Freeing %s, GC_IDX: %i\n", t->name, GC_IDX);
    // }
    if (t->data != NULL){
        if (t->device==CUDA){
            checkCudaErrors(cudaFree(t->data));
        } else {
            free(t->data);
        }
        t->data = NULL;
    }

    // *** free cpu memory ***

    if (t->name != NULL){
        free(t->name);
        t->name = NULL;
    }

    // note: these are just pointers on the tensor struct, this will be deallocated with the struct itself
    //  (grad_fn, backward, shape, stride, device, num_dims, num_inputs, num_uses)
    free(t);
    t = NULL;

    // todo-low: free all non_grad_inputs
}

void free_all_tensors(int idx_until){
    // idx_until to avoid free'in weights, and the "dataset" (though can free the "batch")
    printf("GC_IDX: %i\n", GC_IDX);
    for (; GC_IDX>idx_until; GC_IDX--){
        free_tensor(GC[GC_IDX]);
    }
}
