
// noop
void copy_to_cpu(tensor* t){
    printf("\ncopy_to_cpu\n");
    t->device = CPU;
}

// noop
tensor* copy_from_cpu(tensor* t) {
    printf("\ncopy_from_cpu\n");
    t->device = CPU;
    return t;
}

void set_backend_cpu(void){
    extern void (*COPY_TO_DEVICE)(tensor*);
    extern tensor* (*COPY_FROM_DEVICE)(tensor*);
    COPY_TO_DEVICE = copy_to_cpu;
    COPY_FROM_DEVICE = copy_from_cpu;
}
