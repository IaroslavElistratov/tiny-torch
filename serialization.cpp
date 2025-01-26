// todo-now:
// think more about how would loading_all_params interact with the GC list?
// simply using save_tensor / load_tensor likely will not correctly work with the GC



void save_tensor(tensor* t, FILE* f){

    // COPY_FROM_DEVICE returns a new tensor, which removes the other fields
    //  (besides t->data) -- but I want to preserve the name of the original param;
    // I hesitate to just copy t->name in COPY_FROM_DEVICE, because this will result
    //    in two different tensors having the same name unless COPY_FROM_DEVICE
    //    immediately frees the cpu tensor without weighting for the GC to do it later;
    //
    // todo-high: longer term change COPY_FROM_DEVICE to modify t->data instead of returning a new tensor?
    char* name = t->name;

    if (t->device == CUDA){
        t = COPY_FROM_DEVICE(t);
    }

    // write / read in a specific order:

    //  (1) fixed sized

    fwrite(&t->num_dims, sizeof(int), 1, f);
    fwrite(&t->size, sizeof(int), 1, f);
    fwrite(&t->is_leaf, sizeof(bool), 1, f);

    // // not meaningful to save/load these fields:
    // fwrite(&t->device, sizeof(int), 1, f);
    // fwrite(&t->num_inputs, sizeof(int), 1, f);
    // fwrite(&t->num_uses, sizeof(int), 1, f);
    // fwrite(&t->_num_uses, sizeof(int), 1, f);

    //  (2) dynamic sized members

    fwrite(t->data, sizeof(float), t->size, f);
    fwrite(t->shape, sizeof(int), MAX_RANK, f);
    fwrite(t->stride, sizeof(int), MAX_RANK, f);
    fwrite(name, sizeof(char), MAX_TENSOR_NAME, f);

    // // not meaningful to save/load these fields -- weight
    // // is a leaf tensor, it does not have these fields:
    // t->op_type;
    // t->grad;
    // t->scratch_space;
    // t->inputs;
    // t->non_grad_inputs;

    // // not meaningful to save/load these -- function
    // // pointers don't make sense when the program restarts
    // t->grad_fn = NULL;
    // t->backward = backward;
}


//  this fn assumes exact order in which struct members were written to the file (inside "save_tensor" fn)
tensor* load_tensor(FILE* f){
    tensor* t = (tensor*)checkMallocErrors(malloc(sizeof(tensor)));

    //  (1) fixed sized:

    fread(&t->num_dims, sizeof(int), 1, f);
    fread(&t->size, sizeof(int), 1, f);
    fread(&t->is_leaf, sizeof(bool), 1, f);
    // fread(&t->device, sizeof(int), 1, f);
    // fread(&t->op_type, sizeof(int), 1, f);
    // fread(&t->num_inputs, sizeof(int), 1, f);
    // fread(&t->num_uses, sizeof(int), 1, f);
    // fread(&t->_num_uses, sizeof(int), 1, f);

    //  (2) dynamic sized members:

    t->data = (float*)checkMallocErrors(malloc(sizeof(float) * t->size));
    fread(t->data, sizeof(float), t->size, f);

    // question-now: I don't think need to allocate space for these -- memory
    //   for the array members is included when "malloc(sizeof(tensor))" above
    fread(t->shape, sizeof(int), MAX_RANK, f);
    fread(t->stride, sizeof(int), MAX_RANK, f);

    t->name = (char*)checkMallocErrors(malloc(sizeof(char) * MAX_TENSOR_NAME));
    fread(t->name, sizeof(char), MAX_TENSOR_NAME, f);

    // (3) set defaults
    t->grad_fn = NULL;
    t->grad = NULL;
    t->backward = backward;
    // COPY_TO_DEVICE expects CPU, sometimes in my constructors
    // I don't set the device -- so it's some random value
    t->device = CPU;
    t->num_inputs = 0;
    t->num_uses = 0;

    // mv to device

    // note: need to do the copying after done reading all
    // the fields -- otherwise reading gets corrupted
    if (DEVICE == CUDA){
        COPY_TO_DEVICE(t);
    }

    if (ferror(f)){
        t = NULL;
    }

    return t;
}


void save_param(param* p, FILE* f){

    // write / read in a specific order:

    //  (1) fixed sized

    fwrite(&p->t, sizeof(int), 1, f);
    fwrite(&p->beta1, sizeof(float), 1, f);
    fwrite(&p->beta2, sizeof(float), 1, f);
    fwrite(&p->epsilon, sizeof(float), 1, f);

    // can't meaningfully save/restore these:
    // p->next;

    //  (2) dynamic sized members

    save_tensor(p->value, f);
    save_tensor(p->velocity, f);
    save_tensor(p->first_moment, f);
    save_tensor(p->second_moment, f);
}


param* load_param(FILE* f){

    param* p = (param*)checkMallocErrors(malloc(sizeof(param)));

    fread(&p->t, sizeof(int), 1, f);
    fread(&p->beta1, sizeof(float), 1, f);
    fread(&p->beta2, sizeof(float), 1, f);
    fread(&p->epsilon, sizeof(float), 1, f);

    p->value = load_tensor(f);
    p->velocity = load_tensor(f);
    p->first_moment = load_tensor(f);
    p->second_moment = load_tensor(f);

    // set defaults
    p->next = NULL;

    if (ferror(f)){
        p = NULL;
    }

    return p;
}


void save_all_params(const char* prefix, int ep_idx){

    char path[50];
    snprintf(path, sizeof(char) * 50, "./generated/checkpoints/%s.dat", prefix);

    // flush buffer
    FILE *f = fopen(static_cast<const char*>(path), "w");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
    }
    fclose(f);

    f = fopen(static_cast<const char*>(path), "a");

    // used in load_all_params to know how many times to iterate
    int num_params = count_params();
    fwrite(&num_params, sizeof(int), 1, f);

    float learning_rate = LR;
    fwrite(&learning_rate, sizeof(float), 1, f);

    fwrite(&ep_idx, sizeof(int), 1, f);

    param* temp = param_head;
    while (temp){
        save_param(temp, f);
        temp = temp->next;
    }
    fclose(f);

    printf("[save_all_params] Parameters saved.\n");
}


int load_all_params(char* prefix){
    if (param_head){
        printf("[load_all_params] loading params when then param list is not empty -- is not supported\n");
        exit(1);
    }

    char path[50];
    snprintf(path, sizeof(char) * 50, "./generated/checkpoints/%s.dat", prefix);

    FILE *f = fopen(static_cast<const char*>(path), "rb");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
    }

    int num_params;
    fread(&num_params, sizeof(int), 1, f);

    // not used at the moment
    float learning_rate;
    fread(&learning_rate, sizeof(float), 1, f);

    // need to return this to continue incrementing ep_idxs after loading
    // (instead of starting again from 0), and avoid overwriting previous checkpoints
    int ep_idx;
    fread(&ep_idx, sizeof(int), 1, f);

    param* loaded;
    for (int i=0; i<num_params; i++){
        loaded = load_param(f);
        printf("[load_all_params] loaded param for %s tensor\n", loaded->value->name);
        loaded->next = param_head;
        param_head = loaded;
    }
    fclose(f);

    printf("[load_all_params] Parameters loaded.\n");
    return ep_idx;
}
