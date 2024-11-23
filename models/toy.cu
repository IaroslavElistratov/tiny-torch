#include <iostream> // todo: use C only
using namespace std;

#define DEVICE CUDA

#include "../nn.h"
#include "../tensor.cpp"
#include "../ops.cpp"
#include "../utils.cpp" // graphviz
#include "../print.cpp"

// todo-now:
// for cuda, use same operation abstractions (as for cpu), but make these abstractions call host stubs for cuda kernels (instead of my kernles for cpu) -- this will preserve my graph building functionality
//   - re-use this from ops.cpp: use same names for kernels, just make them refer to different impls (cpu, cuda) depending on wether device is CUDA or not -- this will reduce code duplication needed to copy paste ops

// todo-now:
// and "matmul_bwd", "batched_matmul_bwd", "div_bwd", "pow_bwd", "reduce_sum_bwd" can also be re-used!
// Seems possible to overwrite all tensor constructors (from CPU to CUDA) when DEVICE macro is set to CUDA, and re-use _bwd[s] ?
//  No need to overwrite constructors, just move selection of what subclass to call (cpu or cuda) to the constructor fn itself

// todo-now:
// Need to abstract away loop over individual elements and then can re-use most of the fwd kernels


// // test_reduce
// int main() {
//     srand(123);
//     set_backend_device();

//     int B = 2, H = 16;

//     // todo-high: currently don't support num_elements lower than NUM_THREADS*2 -- add guards to the kernel
//     tensor* x = Tensor(B, H);
//     set_name(x, "x"); print(x);

//     // tensor* out = reduce_sum(x);
//     tensor* out = reduce_max(x);
//     set_name(out, "out"); print(out);

//     out->backward(out);
//     return 0;
// }



// test_batched_reduce
int main() {
    srand(123);
    set_backend_device();

    int B = 2, H = 32;

    // todo-high: currently don't support 2nd dim lower than NUM_THREADS*2 -- add guards to the kernel
    tensor* x = Tensor(B, H);
    set_name(x, "x"); print(x);

    // tensor* out = batched_reduce_sum(x);
    tensor* out = batched_reduce_max(x);
    set_name(out, "out"); print(out);

    // out->backward(out);
    return 0;
}

// int test_backends(){
//     srand(123);

//     int N = 16, M = 8;

//     set_backend_device();
//     tensor* cuda_t = Tensor(N, M);
//     set_name(cuda_t, "cuda_t"); print(cuda_t);

//     set_backend_cpu();
//     tensor* cpu_tens = Tensor(N, M);
//     set_name(cpu_tens, "cpu_tens"); print(cpu_tens);
//     return 0;
// }


// int test_bmm() {
//     srand(123);

//     int B=3, N = 2, M = 8, D = 4;

//     set_backend_device();

//     tensor* x = Tensor(B, N, M);
//     set_name(x, "x"); print(x);

//     tensor* w1 = Tensor(B, M, D);
//     set_name(w1, "w1"); print(w1);

//     tensor* out = batched_matmul(x, w1);
//     set_name(out, "out"); print(out);

//     out->backward(out);
//     return 0;
// }



// int test_conv() {
//     srand(123);

//     int H = 4, W = 4, C = 3, F = 5, K = 2;

//     set_backend_device();

//     tensor* x = Tensor(C, H, W);
//     set_name(x, "x"); print(x);

//     tensor* kernels = Tensor(F, C, K, K);
//     set_name(kernels, "kernels"); print(kernels);

//     tensor* out = conv(x, kernels);
//     set_name(out, "out"); print(out);

//     out->backward(out);
//     return 0;
// }


// int test_batched_conv() {
//     srand(123);

//     // int B = 3000, H = 128, W = 128, C = 3, F = 5, K = 2;
//     int B = 2, H = 4, W = 4, C = 3, F = 5, K = 2;

//     set_backend_device();

//     tensor* x = Tensor(B, C, H, W);
//     set_name(x, "x"); print(x);

//     tensor* kernels = Tensor(F, C, K, K);
//     set_name(kernels, "kernels"); print(kernels);

//     tensor* out = batched_conv(x, kernels);
//     set_name(out, "out"); print(out);

//     out->backward(out);
//     return 0;
// }


// int test_pool() {
//     srand(123);

//     int H = 4, W = 4, C = 3, K = 2;

//     set_backend_device();

//     tensor* x = Tensor(C, H, W);
//     set_name(x, "x"); print(x);

//     tensor* out = maxpool(x);
//     set_name(out, "out"); print(out);

//     out->backward(out);
//     return 0;
// }


// int test_batched_pool() {
//     srand(123);

//     int B = 2, H = 4, W = 4, C = 3, K = 2;

//     set_backend_device();

//     tensor* x = Tensor(B, C, H, W);
//     set_name(x, "x"); print(x);

//     tensor* out = batched_maxpool(x);
//     set_name(out, "out"); print(out);

//     out->backward(out);
//     return 0;
// }



int test_simple_ops() {
    srand(123);

    int N = 2, M = 8, D = 4;

    // by default set to DEVICE backend
    set_backend_device();

    tensor* x = Tensor(N, M);
    set_name(x, "x"); print(x);

    tensor* w1 = Tensor(M, D);
    set_name(w1, "w1"); print(w1);

    tensor* mm = matmul(x, w1);     // (N, D)
    print(mm);

    tensor* tr = transpose(mm);     // (D, N)
    print(tr);

    tensor* w2 = TensorLike(tr);    // (D, N)
    set_name(w2, "w2"); print(w2);
    tensor* ad = add(tr, w2);       // (D, N)
    print(ad);

    tensor* w3 = TensorLike(ad);    // (D, N)
    set_name(w3, "w3"); print(w3);
    // todo: hangs when replacing "ad" w "tr" below
    tensor* su = sub(ad, w3);       // (D, N)
    print(su);

    tensor* w4 = TensorLike(su);    // (D, N)
    set_name(w4, "w4"); print(w4);
    tensor* di = div(su, w4);       // (D, N)
    print(di);

    tensor* pw = pow(di, 2);
    tensor* ng = neg(pw);
    tensor* ex = exp(ng);
    tensor* out = log(ex);
    set_name(out, "out"); print(out);

    out->backward(out);
    graphviz(out);

    return 0;
}
