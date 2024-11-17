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

void test_backends(void){
    srand(123);

    int N = 16, M = 8;

    set_backend_device();
    tensor* cuda_t = Tensor(N, M);
    set_name(cuda_t, "cuda_t"); print(cuda_t);

    set_backend_cpu();
    tensor* cpu_tens = Tensor(N, M);
    set_name(cpu_tens, "cpu_tens"); print(cpu_tens);
}



int main() {
    srand(123);

    int N = 16;
    int M = 8;
    int D = 4;

    // by default set to DEVICE backend
    set_backend_device();

    tensor* x = Tensor(N, M);
    set_name(x, "x"); print(x);

    tensor* w1 = Tensor(M, D);
    set_name(w1, "w1"); print(w1);

    tensor* mm = matmul(x, w1);     // (N=16, D=4)
    set_name(mm, "mm"); print(mm);

    tensor* tr = transpose(mm);     // (D=4, N=16)
    set_name(tr, "tr"); print(tr);

    tensor* w2 = TensorLike(tr);     // (D=4, N=16)
    set_name(w2, "w2"); print(w2);
    tensor* ad = add(tr, w2);       // (D, N)
    set_name(ad, "ad"); print(ad);

    ad->backward(ad);
    graphviz(ad);

    return 0;
}
