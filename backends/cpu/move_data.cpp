
// noop
void copy_to_cpu(tensor* t){
    if (DATA_COPY_DEBUG) printf("copy_to_cpu\n");
    if (t->device==CPU){
        return;
    }
    t->device = CPU;
}

// noop
tensor* copy_from_cpu(tensor* t) {
    if (DATA_COPY_DEBUG) printf("copy_from_cpu\n");
    t->device = CPU;
    return t;
}

void set_backend_cpu(void){
    COPY_TO_DEVICE = copy_to_cpu;
    COPY_FROM_DEVICE = copy_from_cpu;
}
