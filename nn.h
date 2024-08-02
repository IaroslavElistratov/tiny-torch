// include guards
#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#define MAX_INPUTS 2
#define MAX_TENSOR_NAME 10 // unique for each tensor

// todo: avoid needing to manually sync increment op_type, NUM_OPS, VIS_COLORS when adding a new op
//  - use graphviz's pastel19 or set312 color scheme ?
#define NUM_OPS 8
const char* OP_NAMES[] = {"add", "sub", "mul", "matmul", "pow", "reduce_sum", "relu", "transpose"};
const char* VIS_COLORS[] = {"darkolivegreen1", "lightsalmon1", "skyblue1", "plum1", "mediumpurple1", "aquamarine", "yellow", "seashell"};


// tensor and its fn's are used in both ops.cpp and main.cpp
struct tensor {
    float* data;
    int shape[3];
    int stride[3];

    // to avoid: int size = x->shape[0] * x->shape[1];
    int size;

    // for autograd engine:

    bool is_leaf;

    tensor* grad;
    void (*grad_fn)(tensor* upstream, tensor* out);

    int num_inputs;
    tensor* inputs[MAX_INPUTS];

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

float* EmptyFloat(int s1, int s2);
float* EmptyFloatLike(tensor* t);
float* FloatLikeFill(tensor* t, int value);

#endif