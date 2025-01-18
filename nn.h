// include guards
#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#define MAX_RANK 4
#define MAX_INPUTS 3
#define MAX_TENSOR_NAME 20 // unique for each tensor
#define MAX_SCRATCH_SPACE 1

// todo: avoid needing to manually sync increment op_type, NUM_OPS, VIS_COLORS when adding a new op
//  - use graphviz's pastel19 or set312 color scheme ?
#define NUM_OPS 23
const char* OP_NAMES[] = {"add", "sub", "mul", "matmul", "pow", "reduce_sum", "relu", "transpose", "batched_matmul", "conv", "batched_conv", "maxpool", "batched_maxpool", "batched_flatten", "select", "log", "exp", "batched_reduce_sum", "repeat", "neg", "div", "reduce_max", "batched_reduce_max"};
const char* VIS_COLORS[] = {"darkolivegreen1", "lightsalmon1", "skyblue1", "plum1", "mediumpurple1", "aquamarine", "yellow", "seashell", "orchid2", "deeppink1", "deeppink3", "darkseagreen1", "darkseagreen3", "beige", "bisque", "cornsilk", "darkolivegreen1", "tan1", "deepskyblue", "chocolate", "slateblue", "lemonchiffon", "lightgoldenrodyellow"};


// tensor and its fn's are used in both ops.cpp and main.cpp
struct tensor {
    float* data;
    int shape[MAX_RANK];
    int device;

    // https://arxiv.org/pdf/1102.1523
    // Strides the number of bytes to skip in memory to
    // proceed to the next element. For a (10, 10)
    // array of bytes, for example, the strides may be
    // (10, 1), in other words: proceed one byte to
    // get to the next column and ten bytes to locate
    // the next row.
    int stride[MAX_RANK];

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
    // question-now: malloc memory for this?
    int non_grad_inputs[MAX_INPUTS];
    int num_uses; // for the AG engine: num outputs of the this tensor left to call their out->grad_fn, before calling grad_fn on the current tensor
    int _num_uses;

    // to store data recorded in fwd and used in bwd (wt needing to recompute it in bwd)
    // e.g. in relu, reduce_max, maxpool: idxs recorded during forward _k and used in its corresponding _bwd fn
    tensor* scratch_space[MAX_SCRATCH_SPACE];

    char* name;
    // use char as small int
    int op_type;

    void (*backward)(tensor* t);
};


// enum device {
//     cpu = 1,
//     cuda = 2,
//     // amd = 3,
//     // tpu = 4,
// };

// note: do not use 0 to denote a valid device, it conflicts with NULL pointer
// (e.g. checks like "t->device==CPU", if CPU is 0)
#define CPU 1
#define CUDA 2

// these two will only be used if DEVICE == CUDA
#define NUM_THREADS 16
#define CUDA_DEBUG false
#define DATA_COPY_DEBUG false

void uniform_init(tensor*);
void kaiming_normal_init(tensor*);
void set_name(tensor*, const char*);
char* random_chars(int);

tensor* TensorLikeFill(tensor*, float);
tensor* TensorLike(tensor* t);
void* checkMallocErrors(void* ptr);

void print(tensor*);
void lprint(tensor*);
void cuda_lprint(tensor*);
void sprint(tensor* t);

struct param
{
    tensor* value;
    // sgd:
    tensor* velocity;
    // adam:
    int t;
    float beta1;
    float beta2;
    float epsilon;
    tensor* first_moment;
    tensor* second_moment;

    param* next;
};

struct tuple {
    float item_1;
    float item_2;
};


tuple* get_tuple(float val1, float val2);

#endif
