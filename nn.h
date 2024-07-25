// include guards
#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

// tensor and its fn's are used in both ops.cpp and main.cpp
struct tensor {
    float* data;
    int shape[2];
    // to avoid
    //  int size = x->shape[0] * x->shape[1];
    int size;
};

tensor* Tensor(int s1, int s2);
tensor* TensorLike(tensor* t);
tensor* TensorLikeFill(tensor* t, float value);

#endif