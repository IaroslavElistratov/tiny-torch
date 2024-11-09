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

        // not truncating to 0 due to int division, bc C promotes args
        dst[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
}


tensor* TensorNoData(int y, int z)
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

    // true by default, modified in an op impl if that tensor was produced by an op
    t->is_leaf = true;
    t->num_inputs = -1;
    // unlike num_inputs (which I can't set here in the constructor,
    // bc it's unknown at this moment, bc depends on whatever ops is
    // being called on that tensor), num_uses is 1nown from the start
    // -- it's 0 for all ops
    t->num_uses = 0;
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

// todo: add EmptyTensor, EmptyTensorLike, make TensorLikeFill use EmptyTensorLike (instead of TensorLike)
tensor* TensorNoData3d(int x, int y, int z)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->num_dims = 3;
    t->size = x*y*z;
    t->shape[0] = x;
    t->shape[1] = y;
    t->shape[2] = z;

    // todo-high: should express later outer strides in terms of inner strides?
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

// todo: minimize duplication (this vs other constructors) this was copy-pasted from TensorNoData3d
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
tensor* EmptyTensor(int s1, int s2)
{
    tensor* t = TensorNoData(s1, s2);
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


// todo: rename to Tensor2d, EmptyTensor2d, TensorNoData2d
tensor* Tensor(int s1, int s2)
{
    tensor* t = EmptyTensor(s1, s2);
    GetRandomFloat(t->data, t->size);
    return t;
}

tensor* CudaTensor(int s1, int s2)
{
    // todo-low: directly initialize random floats on gpu (avoid initializing on cpu, and then moving)
    tensor* t = Tensor(s1, s2);
    t->device = CUDA; // device.cuda;

    float* t_device;
    int size = t->size * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&t_device, size);
    // todo: exit from program everywhere in case of error
    if (err != cudaSuccess){
        printf("[cuda malloc] error");
    }
    err = cudaMemcpy(t_device, t->data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("[cuda memcopy] error");
    }
    // todo: free cpu t->data (currently memory leak)
    t->data = t_device;

    return t;
}

tensor* Tensor3d(int s1, int s2, int s3)
{
    tensor* t = EmptyTensor3d(s1, s2, s3);
    GetRandomFloat(t->data, t->size);
    return t;
}

tensor* Tensor4d(int s1, int s2, int s3, int s4)
{
    tensor* t = EmptyTensor4d(s1, s2, s3, s4);
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

tensor* CudaTensorLike(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return CudaTensor(s1, s2);
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


// todo: in each TensorLikeFill wasteful to init w random value
// using GetRandomFloat and then overwrite them anyway
tensor* TensorLikeFill(tensor* t, float value)
{
    tensor* t_new = TensorLike(t);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    return t_new;
}

tensor* TensorLikeFill3d(tensor* t, float value)
{
    tensor* t_new = TensorLike3d(t);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    return t_new;
}

tensor* TensorLikeFill4d(tensor* t, float value)
{
    tensor* t_new = TensorLike4d(t);
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

tensor* EmptyTensorLike(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return EmptyTensor(s1, s2);
}


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
        printf("[set_name] Warning, specified name larger than MAX_TENSOR_NAME -- truncating\n");
        t->name[i+1] = '\0';
    }
}


// // constructors that take in tensor and return float

// float* EmptyFloat(int s1, int s2)
// {
//     return (float*)malloc(sizeof(float) * s1*s2);
// }

// // used when only tensor->data is needed, and avoids
// // memory leak unlike the below -- bc tensor output
// // of TensorLikeFill won't be used anymore (only one
// // of its members will)
// //  a->grad = TensorLikeFill(a, 1.0)->data;
// float* EmptyFloatLike(tensor* t)
// {
//     return EmptyFloat(t->shape[0], t->shape[1]);
// }

// float* FloatLikeFill(tensor* t, int value)
// {
//     float* f_new = EmptyFloatLike(t);
//     for (int i=0; i<t->size; i++)
//         f_new[i] = value;
//     return f_new;
// }
