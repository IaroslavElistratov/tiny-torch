#include "nn.h"
#include "autograd.cpp"


// todo-now: assert callback func pointers are not NULL, before calling them

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

    // todo: put the fields below into an autograd_op struct,
    // and store a single pointer to that struct on the tensor
    // -- this way more organized

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

// todo: EmptyTensor constructors at the moment do not call COPY_TO_DEVICE

// empty means non-initalized, but with data allocated to it
tensor* EmptyTensor2d(int s1, int s2)
{
    tensor* t = TensorNoData2d(s1, s2);
    t->data = (float*)malloc(sizeof(float) * t->size);
    t->device = CPU;
    return t;
}

tensor* EmptyTensor3d(int x, int y, int z)
{
    tensor* t = TensorNoData3d(x, y, z);
    t->data = (float*)malloc(sizeof(float) * t->size);
    t->device = CPU;
    return t;
}

tensor* EmptyTensor4d(int o, int x, int y, int z)
{
    tensor* t = TensorNoData4d(o, x, y, z);
    t->data = (float*)malloc(sizeof(float) * t->size);
    t->device = CPU;
    return t;
}


tensor* Tensor2d(int s1, int s2)
{
    tensor* t = EmptyTensor2d(s1, s2);
    GetRandomFloat(t->data, t->size);
    // todo-low: directly initialize random floats on gpu (avoid initializing on cpu, and then moving)
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


// todo: in each TensorLikeFill wasteful to init w random value
// using GetRandomFloat and then overwrite them anyway
tensor* TensorLikeFill2d(tensor* t, float value)
{
    tensor* device_t_new = TensorLike2d(t);
    tensor* t_new = COPY_FROM_DEVICE(device_t_new);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    // todo-high: all TensorLikeFillNd constructors (and TensorScalarFill) invoke COPY_TO_DEVICE twice -- once here, another time in TensorLike2d -> Tensor2d -> COPY_TO_DEVICE
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
    // todo: wasteful to init w random value using GetRandomFloat
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

// todo: static_assert(VA_NARGS(__VA_ARGS__) <= MAX_DIM, "[constructor] error")
#define Tensor(...) CONCAT(Tensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)

// this preserves device, needed because often in cpu backend, one off tensors are created using TensorNoData,
// which are then passed to other kernels_, when these kernels run input checks, they expect to see that tensor
// passed to them has t->device=CPU -- so I modified the TensorNoData constructor to preserve the t->device flag
#define _TensorNoData(...) CONCAT(TensorNoData, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
#define TensorNoData(...) ({tensor* out = _TensorNoData(__VA_ARGS__); out->device=DEVICE; out;})

#define EmptyTensor(...) CONCAT(EmptyTensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
// conv_k_ -> out = EmptyTensor(...) -> (COPY_TO_DEVICE NOT called, so backend isn't set)
//  when out of conv_k_ fed to some next kernel (e.g. relu), relu does input checks, expects to see out->device=CPU
// NOTE: unlike TensorNoData (which does NOT allocate t->data on any device), for EmptyTensor it's incorrect to set "out->device=DEVICE" -- because EmptyTensor only allocates on cpu at the moment
// #define _EmptyTensor(...) CONCAT(EmptyTensor, CONCAT(VA_NARGS(__VA_ARGS__), d))(__VA_ARGS__)
// #define EmptyTensor(...) ({tensor* out = _EmptyTensor(__VA_ARGS__); out->device=CPU; out;})

/*
// question-now: 
//  use funcs of this form instead of the macros above?
//  This will also allow to call e.g. COPY_TO_DEVICE only once in e.g. Tensor fn, instead of needing to call it multiple times from each impl (e.g. Tensor2d)

#include <stdarg.h>

#define COUNT_ARGS(...) \
    (sizeof((int[]){__VA_ARGS__})/sizeof(int))

#define Tensor(...) TensorImpl(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

tensor* TensorImpl(int num_args, ...){
    va_list args;
    va_start(args, num_args);

    int s0 = va_arg(args, int);
    int s1 = va_arg(args, int);

    if (num_args == 2){
        return Tensor2d(s0, s1);
    } else if (num_args == 3){
        int s2 = va_arg(args, int);
        return Tensor3d(s0, s1, s2);
    } else if (num_args == 4){
        int s2 = va_arg(args, int);
        int s3 = va_arg(args, int);
        return Tensor3d(s0, s1, s2, s3);
    }
    va_end(args);
}
*/

// comment:
//  use functions (instead of macros) for TensorLike, TensorLikeFill, bc preprocessor can't
//  evaluate runtime value from a struct member at pre-processing time, so handle it in a fn

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
