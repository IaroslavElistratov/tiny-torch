// include guards
#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#define MAX_INPUTS 2
#define MAX_TENSOR_NAME 20 // unique for each tensor

// todo: avoid needing to manually sync increment op_type, NUM_OPS, VIS_COLORS when adding a new op
//  - use graphviz's pastel19 or set312 color scheme ?
#define NUM_OPS 23
const char* OP_NAMES[] = {"add", "sub", "mul", "matmul", "pow", "reduce_sum", "relu", "transpose", "batched_matmul", "conv", "batched_conv", "maxpool", "batched_maxpool", "batched_flatten", "select", "log", "exp", "batched_reduce_sum", "repeat", "neg", "div", "max", "batched_max"};
const char* VIS_COLORS[] = {"darkolivegreen1", "lightsalmon1", "skyblue1", "plum1", "mediumpurple1", "aquamarine", "yellow", "seashell", "orchid2", "deeppink1", "deeppink3", "darkseagreen1", "darkseagreen3", "beige", "bisque", "cornsilk", "darkolivegreen1", "tan1", "deepskyblue", "chocolate", "slateblue", "lemonchiffon", "lightgoldenrodyellow"};


// tensor and its fn's are used in both ops.cpp and main.cpp
struct tensor {
    float* data;
    int shape[4];

    // https://arxiv.org/pdf/1102.1523
    // Strides the number of bytes to skip in memory to
    // proceed to the next element. For a (10, 10)
    // array of bytes, for example, the strides may be
    // (10, 1), in other words: proceed one byte to
    // get to the next column and ten bytes to locate
    // the next row.
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
    int num_uses; // for the AG engine: num outputs of the this tensor left to call their out->grad_fn, before calling grad_fn on the current tensor

    char* name;
    // use char as small int
    int op_type;

    void (*backward)(tensor* t);
};


tensor* EmptyTensor(int s1, int s2);
tensor* EmptyTensorLike(tensor* t);
tensor* Tensor(int s1, int s2);
tensor* TensorLike(tensor* t);
tensor* TensorLikeFill(tensor* t, float value);
tensor* TensorLikeFill3d(tensor* t, float value);
tensor* TensorLikeFill4d(tensor* t, float value);

float* EmptyFloat(int s1, int s2);
float* EmptyFloatLike(tensor* t);
float* FloatLikeFill(tensor* t, int value);

void set_name(tensor*, const char*);


void print_2d(tensor*);
void print_3d(tensor*);
void print_4d(tensor*);

void lprint_2d(tensor* t);
void lprint_3d(tensor* t);
void lprint_4d(tensor* t);

#endif
