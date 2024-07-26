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

tensor* Tensor(int s1, int s2)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->size = s1*s2;

    t->shape[0] = s1;
    t->shape[1] = s2;

    t->data = GetRandomFloat(s1*s2);

    // for autograd engine
    // cout << t->grad << endl;
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
    tensor* t_new = TensorLike(t);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    return t_new;
}

// x_max is number columns to get to next row
int index(int x, int y, int x_max) {
    return x * x_max + y;
}