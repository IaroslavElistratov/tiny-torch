#include "nn.h"
#include "autograd.cpp"

extern void (*COPY_TO_DEVICE)(tensor*);
extern tensor* (*COPY_FROM_DEVICE)(tensor*);



tensor* TensorNoData2d(int y, int z)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->num_dims = 2;
    t->size = y*z;
    // todo-low: one line
    t->shape[0] = y;
    t->shape[1] = z;

    t->stride[0] = z;
    t->stride[1] = 1;

    t->data = NULL;

    // for autograd engine:

    t->is_leaf = true;
    t->num_inputs = -1;
    t->num_uses = 0;
    t->name = random_chars(3);

    t->grad_fn = NULL;
    t->grad = NULL;
    t->backward = backward;

    return t;
}

tensor* TensorNoData3d(int x, int y, int z)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->num_dims = 3;
    t->size = x*y*z;
    t->shape[0] = x;
    t->shape[1] = y;
    t->shape[2] = z;

    t->stride[0] = y*z;
    t->stride[1] = z;
    t->stride[2] = 1;

    t->data = NULL;

    t->is_leaf = true;
    t->num_inputs = -1;
    t->num_uses = 0;
    t->name = random_chars(3);

    t->grad_fn = NULL;
    t->grad = NULL;
    t->backward = backward;

    return t;
}


tensor* TensorNoData4d(int o, int x, int y, int z)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->num_dims = 4;
    t->size = o*x*y*z;
    t->shape[0] = o;
    t->shape[1] = x;
    t->shape[2] = y;
    t->shape[3] = z;

    t->stride[0] = x*y*z;
    t->stride[1] = y*z;
    t->stride[2] = z;
    t->stride[3] = 1;

    t->data = NULL;

    // for autograd engine:
    t->is_leaf = true;
    t->num_inputs = -1;
    t->num_uses = 0;
    t->name = random_chars(3);

    t->grad_fn = NULL;
    t->grad = NULL;
    t->backward = backward;

    return t;
}


// empty means non-initalized, but with data allocated to it
tensor* EmptyTensor2d(int s1, int s2)
{
    tensor* t = TensorNoData2d(s1, s2);
    t->data = (float*)malloc(sizeof(float) * t->size);
    return t;
}

tensor* EmptyTensor3d(int x, int y, int z)
{
    tensor* t = TensorNoData3d(x, y, z);
    t->data = (float*)malloc(sizeof(float) * t->size);
    return t;
}

tensor* EmptyTensor4d(int o, int x, int y, int z)
{
    tensor* t = TensorNoData4d(o, x, y, z);
    t->data = (float*)malloc(sizeof(float) * t->size);
    return t;
}


tensor* Tensor2d(int s1, int s2)
{
    tensor* t = EmptyTensor2d(s1, s2);
    GetRandomFloat(t->data, t->size);
    COPY_TO_DEVICE(t);
    return t;
}

tensor* Tensor3d(int s1, int s2, int s3)
{
    tensor* t = EmptyTensor3d(s1, s2, s3);
    GetRandomFloat(t->data, t->size);
    COPY_TO_DEVICE(t);
    return t;
}

tensor* Tensor4d(int s1, int s2, int s3, int s4)
{
    tensor* t = EmptyTensor4d(s1, s2, s3, s4);
    GetRandomFloat(t->data, t->size);
    COPY_TO_DEVICE(t);
    return t;
}


/*
for convince to avoid:
    int N = x->shape[0], M = x->shape[1];
    tensor* out = Tensor2d(N, D);
*/
tensor* TensorLike2d(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return Tensor2d(s1, s2);
}

tensor* TensorLike3d(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1], s3 = t->shape[2];
    return Tensor3d(s1, s2, s3);
}

tensor* TensorLike4d(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1], s3 = t->shape[2], s4 = t->shape[3];
    return Tensor4d(s1, s2, s3, s4);
}


tensor* TensorLikeFill2d(tensor* t, float value)
{
    tensor* device_t_new = TensorLike2d(t);
    tensor* t_new = COPY_FROM_DEVICE(device_t_new);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    // todo : all TensorLikeFillNd constructors (and TensorScalarFill) invoke COPY_TO_DEVICE
    // twice -- once here, another time in TensorLike2d -> Tensor2d -> COPY_TO_DEVICE
    COPY_TO_DEVICE(t_new);
    return t_new;
}

tensor* TensorLikeFill3d(tensor* t, float value)
{
    tensor* device_t_new = TensorLike3d(t);
    tensor* t_new = COPY_FROM_DEVICE(device_t_new);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    COPY_TO_DEVICE(t_new);
    return t_new;
}

tensor* TensorLikeFill4d(tensor* t, float value)
{
    tensor* device_t_new = TensorLike4d(t);
    tensor* t_new = COPY_FROM_DEVICE(device_t_new);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    COPY_TO_DEVICE(t_new);
    return t_new;
}


tensor* TensorScalarFill(float value)
{
    // todo : wasteful to init w random value using GetRandomFloat
    //  and then overwrite them anyway
    tensor* device_t_new = Tensor2d(1, 1);
    tensor* t_new = COPY_FROM_DEVICE(device_t_new);
    // needed bc Tensor initializes to random value
    t_new->data[0] = value;
    COPY_TO_DEVICE(t_new);
    return t_new;
}




// the below implements function dispatching based on the number of arguments
//  abstracts constructors of each family by dispatching to different impls depending on the number of args

// need this instead of sizeof based impl, otherwise preprocessor fails
#define VA_NARGS_IMPL(_1, _2, _3, _4, _5, N, ...) N
#define VA_NARGS(...) VA_NARGS_IMPL(__VA_ARGS__, 5, 4, 3, 2, 1)

// need inner otherwise preprocessor fails
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

#define MAX_DIM 4

#define TensorNoData(...) CONCAT(TensorNoData, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
#define EmptyTensor(...) CONCAT(EmptyTensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
#define Tensor(...) CONCAT(Tensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)

tensor* TensorLike(tensor* t){
    if (t->num_dims==2)
        return TensorLike2d(t);
    else if (t->num_dims==3)
        return TensorLike3d(t);
    else if (t->num_dims==4)
        return TensorLike4d(t);
    else
        exit(1);
}

tensor* TensorLikeFill(tensor* t, float val){
    if (t->num_dims==2)
        return TensorLikeFill2d(t, val);
    else if (t->num_dims==3)
        return TensorLikeFill3d(t, val);
    else if (t->num_dims==4)
        return TensorLikeFill4d(t, val);
    else
        exit(1);
}
