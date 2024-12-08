// include guards
#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#define MAX_INPUTS 2
#define MAX_TENSOR_NAME 20 // unique for each tensor


#define NUM_OPS 23
const char* OP_NAMES[] = {"add", "sub", "mul", "matmul", "pow", "reduce_sum", "relu", "transpose", "batched_matmul", "conv", "batched_conv", "maxpool", "batched_maxpool", "batched_flatten", "select", "log", "exp", "batched_reduce_sum", "repeat", "neg", "div", "max", "batched_max"};
const char* VIS_COLORS[] = {"darkolivegreen1", "lightsalmon1", "skyblue1", "plum1", "mediumpurple1", "aquamarine", "yellow", "seashell", "orchid2", "deeppink1", "deeppink3", "darkseagreen1", "darkseagreen3", "beige", "bisque", "cornsilk", "darkolivegreen1", "tan1", "deepskyblue", "chocolate", "slateblue", "lemonchiffon", "lightgoldenrodyellow"};


// tensor and its fn's are used in both ops.cpp and main.cpp
struct tensor {
    float* data;
    int shape[4];
    int device;

    int stride[4];

    // rank
    int num_dims;

    // to avoid: int size = x->shape[0] * x->shape[1];
    int size;

    // for autograd engine:

    bool is_leaf;

    tensor* grad;
    void (*grad_fn)(tensor* upstream, tensor* out);

    int num_inputs;
    tensor* inputs[MAX_INPUTS];
    int num_uses;

    tensor* scratch_space[1];

    char* name;
    // use char as small int
    int op_type;

    void (*backward)(tensor* t);
};


#define CPU 0
#define CUDA 1

// these two will only be used if DEVICE == CUDA
#define NUM_THREADS 16
#define CUDA_DEBUG false
#define DATA_COPY_DEBUG false

void GetRandomFloat(float*, int);
void set_name(tensor*, const char*);
char* random_chars(int);

tensor* TensorLikeFill(tensor*, float);
tensor* TensorLike(tensor* t);
// tensor* Tensor(...);

void print(tensor*);
void lprint(tensor*);
void cuda_lprint(tensor*);
void sprint(tensor* t);

#endif
