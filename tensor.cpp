#include "nn.h"
#include "autograd.cpp"


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


// todo: rename to Tensor2d, EmptyTensor2d, TensorNoData2d
tensor* Tensor2d(int s1, int s2)
{
    tensor* t = EmptyTensor2d(s1, s2);
    GetRandomFloat(t->data, t->size);
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


// todo: in each TensorLikeFill wasteful to init w random value
// using GetRandomFloat and then overwrite them anyway
tensor* TensorLikeFill2d(tensor* t, float value)
{
    tensor* t_new = TensorLike2d(t);
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
    tensor* t = Tensor2d(1, 1);
    // needed bc Tensor initializes to random value
    t->data[0] = value;
    return t;
}

tensor* EmptyTensorLike2d(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return EmptyTensor2d(s1, s2);
}



void _copy_data_to_cuda(tensor* t)
{
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
}

float* _copy_data_to_cpu(tensor* t) {
    // todo: can just define a macro for print to call 4 lines below and then call the orignal print2d (no need for cuda_print_2d)
    cudaDeviceSynchronize();
    int size = t->size * sizeof(float);
    float* host_data = (float*)malloc(size);
    cudaError_t err = cudaMemcpy(host_data, t->data, size, cudaMemcpyDeviceToHost);
    // todo: define a macro CUDA_CHECK for unwrapping this
    if (err != cudaSuccess){
        printf("[cuda memcopy] error: %s",  cudaGetErrorString(err));
    }
    return host_data;
}

tensor* CudaTensor2d(int s1, int s2)
{
    // todo-low: directly initialize random floats on gpu (avoid initializing on cpu, and then moving)
    tensor* t = Tensor2d(s1, s2);
    t->device = CUDA; // device.cuda;
    _copy_data_to_cuda(t);
    return t;
}

tensor* CudaTensorLike2d(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return CudaTensor2d(s1, s2);
}

tensor* CudaTensorLikeFill2d(tensor* t, float value)
{
    tensor* t_new = TensorLikeFill2d(t, value);
    t_new->device = CUDA; // device.cuda;
    _copy_data_to_cuda(t_new);
    return t_new;
}



// implement function dispatching based on the number of arguments 
//  abstracting constructors of single family requires dispatching to different fn impls depending on the number of args

// need this instead of sizeof based impl, otherwise preprocessor fails
#define VA_NARGS_IMPL(_1, _2, _3, _4, _5, N, ...) N
#define VA_NARGS(...) VA_NARGS_IMPL(__VA_ARGS__, 5, 4, 3, 2, 1)

// need inner otherwise preprocessor fails
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

#define MAX_DIM 4

// todo: static_assert(VA_NARGS(__VA_ARGS__) <= MAX_DIM, "[constructor] error")
#define TensorNoData(...) CONCAT(TensorNoData, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
#define EmptyTensor(...) CONCAT(EmptyTensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
#define Tensor(...) CONCAT(Tensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
#define CudaTensor(...) CONCAT(CudaTensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)


// question-now: use funcs of this form instead of the macros above?

// #include <stdarg.h>

// #define COUNT_ARGS(...) \
//     (sizeof((int[]){__VA_ARGS__})/sizeof(int))

// #define CudaTensor(...) CudaTensorImpl(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

// tensor* CudaTensorImpl(int num_args, ...){
//     va_list args;
//     va_start(args, arg_count);

//     int s0 = va_arg(args, int);
//     int s1 = va_arg(args, int);

//     if (num_args == 2){
//         return CudaTensor2d(s0, s1);
//     } else if (num_args == 3){
//         int s2 = va_arg(args, int);
//         return CudaTensor3d(s0, s1, s2);
//     } else if (num_args == 4){
//         int s2 = va_arg(args, int);
//         int s3 = va_arg(args, int);
//         return CudaTensor3d(s0, s1, s2, s3);
//     }
// }


// comment:
//  use functions (instead of macros) for TensorLike, TensorLikeFill, CudaTensorLike, CudaTensorLikeFill
//  bc preprocessor can't evaluate runtime value from a struct member at pre-processing time, so handle it in a fn

tensor* TensorLike(tensor* t){
    if (t->num_dims==2)
        return TensorLike2d(t);
    else if (t->num_dims==3)
        return TensorLike3d(t);
    else if (t->num_dims==4)
        return TensorLike4d(t);
    else
        return NULL;
}

tensor* TensorLikeFill(tensor* t, float val){
    if (t->num_dims==2)
        return TensorLikeFill2d(t, val);
    else if (t->num_dims==3)
        return TensorLikeFill3d(t, val);
    else if (t->num_dims==4)
        return TensorLikeFill4d(t, val);
    else
        return NULL;
}

tensor* CudaTensorLike(tensor* t){
    if (t->num_dims==2)
        return CudaTensorLike2d(t);
    // else if (t->num_dims==3)
    //     return CudaTensorLike3d(t);
    // else if (t->num_dims==4)
    //     return CudaTensorLike4d(t);
    else
        return NULL;
}

tensor* CudaTensorLikeFill(tensor* t, float val){
    if (t->num_dims==2)
        return CudaTensorLikeFill2d(t, val);
    // else if (t->num_dims==3)
    //     return CudaTensorLikeFill3d(t, val);
    // else if (t->num_dims==4)
    //     return CudaTensorLikeFill4d(t, val);
    else
        return NULL;
}
