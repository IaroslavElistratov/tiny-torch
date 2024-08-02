// #include <iostream>
// using namespace std;

#include "nn.h"
#include "autograd.cpp"

char* random_chars(int num);

void GetRandomFloat(float* dst, int num)
{
    for (int i=0; i<num; i++)
    {
        // https://linux.die.net/man/3/random
        // returns a pseudo-random int between 0 and RAND_MAX
        // normalize to: 0 - 1
        // shift to: -0.5 - 0.5
        // todo: maybe wrongly truncating to 0 due to int division? No bc C promotes args?
        dst[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
}

// todo: add EmptyTensor, EmptyTensorLike, make TensorLikeFill use EmptyTensorLike (instead of TensorLike)

// empty means non-initalized, but with data allocated to it
tensor* _EmptyTensor(int x, int y, int z)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->size = x*y*z;
    // todo-low: one line
    t->shape[0] = x;
    t->shape[1] = y;
    t->shape[2] = z;

    // todo-high: should express later outer strides in terms of inner strides?
    t->stride[0] = y*z;
    t->stride[1] = z;
    t->stride[2] = 1;

    t->data = (float*)malloc(sizeof(float) * t->size);

    // for autograd engine:

    // true by default, modified in an op impl if that tensor was produced by an op
    t->is_leaf = true;
    t->num_inputs = -1;
    t->name = random_chars(3);

    t->grad_fn = NULL;
    t->grad = NULL;
    // note: it makes more sense to set this in ops, because
    // there I set all other autograd attributes on tensors.
    // But it would be repetitive to set the same attr
    // (t->backward) in every op, so set it here
    t->backward = backward;

    return t;
}

tensor* _Tensor(int s1, int s2, int s3)
{
    tensor* t = _EmptyTensor(s1, s2, s3);
    GetRandomFloat(t->data, t->size);
    return t;
}

// to preserve old behavior
// todo-now: must change ops to
//  - index form the inner most (not other-most)
//  - loop over the other-most dim as well
tensor* EmptyTensor(int s1, int s2)
{
    return _EmptyTensor(0, s1, s2);
}

tensor* EmptyTensorLike(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return EmptyTensor(s1, s2);
}

tensor* Tensor(int s1, int s2)
{
    tensor* t = EmptyTensor(s1, s2);
    GetRandomFloat(t->data, t->size);
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

void set_name(tensor* t, const char* name){
    // free the automatically set random name
    // added by the constructor
    free(t->name);

    // todo-low: small inefficiency of always allocating MAX_TENSOR_NAME
    //  even if user provided str is shorter
    t->name = (char*)malloc(sizeof(char) * MAX_TENSOR_NAME);

    int i=0;
    bool is_break = false;
    for (; !is_break && i<MAX_TENSOR_NAME-1; i++) {
        t->name[i] = name[i];
        if (name[i] == '\0')
            is_break = true;
    }

    if (!is_break && name[i+1] != '\0') {
        cout << "[set_name] Warning, specified name larger than MAX_TENSOR_NAME -- truncating" << endl;
        t->name[i+1] = '\0';
    }
}
