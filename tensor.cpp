// #include <iostream>
// using namespace std;

#include "nn.h"

float* GetRandomFloat(int num)
{
    float* f_ptr = (float*)malloc(sizeof(float) * num);

    for (int i=0; i<num; i++)
    {
        // https://linux.die.net/man/3/random
        // returns a pseudo-random int between 0 and RAND_MAX
        // normalize to: 0 - 1
        // shift to: -0.5 - 0.5
        // todo: maybe wrongly truncating to 0 due to int division? No bc C promotes args?
        f_ptr[i] = ((float)rand() / RAND_MAX) - 0.5;
    }


    return f_ptr;
}

// todo: add EmptyTensor, EmptyTensorLike, make TensorLikeFill use EmptyTensorLike (instead of TensorLike)

tensor* Tensor(int s1, int s2)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->size = s1*s2;

    t->shape[0] = s1;
    t->shape[1] = s2;

    t->data = GetRandomFloat(s1*s2);

    // true by defult, modified in an op impl if that tensor was produced by an op
    t-> is_leaf = true;
    t->op_name = (char*)malloc(sizeof(char) * 12);
    t->name = '-';

    // for autograd engine
    // cout << t->grad << endl;
    t->grad_fn = NULL;
    t->grad = NULL;
    t->num_inputs = -1;

    return t;
}

/*
for convince to avoid:
    int N = x->shape[0], M = x->shape[1];
    tensor* out = Tensor(N, D);
*/
tensor* TensorLike(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return Tensor(s1, s2);
}

tensor* TensorLikeFill(tensor* t, float value)
{
    // todo: wasteful to init w random value using GetRandomFloat
    //  and then overwrite them anyway
    tensor* t_new = TensorLike(t);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    return t_new;
}


// todo: this ScalarFill seems to specific -- think of smt more general
// todo: add constructor for empty tensors -- kind of like TensorLikeFill(, 0.0)
tensor* TensorScalarFill(float value)
{
    // todo: wasteful to init w random value using GetRandomFloat
    //  and then overwrite them anyway
    tensor* t = Tensor(1, 1);
    // needed bc Tensor initializes to random value
    t->data[0] = value;
    return t;
}

// constructors that take in tensor and return float

float* EmptyFloat(int s1, int s2)
{
    return (float*)malloc(sizeof(float) * s1*s2);
}

// used when only tensor->data is needed, and avoids
// memory leak unlike the below -- bc tensor output
// of TensorLikeFill won't be used anymore (only one
// of its members will)
//  a->grad = TensorLikeFill(a, 1.0)->data;
float* EmptyFloatLike(tensor* t)
{
    return EmptyFloat(t->shape[0], t->shape[1]);
}

float* FloatLikeFill(tensor* t, int value)
{
    float* f_new = EmptyFloatLike(t);
    for (int i=0; i<t->size; i++)
        f_new[i] = value;
    return f_new;
}

// Zeros(1, 1)
// Ones(1, 1)


// x_max is number columns to get to next row
int index(int x, int y, int x_max) {
    return x * x_max + y;
}