// include guards
#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#define MAX_INPUTS 3

// tensor and its fn's are used in both ops.cpp and main.cpp
struct tensor {
    float* data;
    int shape[2];
    // to avoid
    //  int size = x->shape[0] * x->shape[1];
    int size;

    bool is_leaf;

    // for autograd engine
    void (*grad_fn)(tensor* upstream, tensor* out);
    tensor* grad;
    int num_inputs;
    tensor* inputs[MAX_INPUTS];
    char name;
    char* op_name;

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