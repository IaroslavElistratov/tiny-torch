#include <iostream> // todo: use C only
using namespace std;

#define DEVICE CUDA
#define print(a) ((DEVICE == CUDA) ? cuda_print_2d(a) : print_2d(a))

#include "../tensor.cpp"
#include "../ops.cpp"
#include "../utils.cpp"
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

int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int N = 16;
    int M = 8;
    int D = 4;

    tensor* x = CudaTensor(N, M);
    set_name(x, "x"); print(x);

    tensor* w1 = CudaTensor(M, D);
    set_name(w1, "w1"); print(w1);

    tensor* out = matmul(x, w1); // (N, D)
    set_name(out, "out_mm"); print(out);

    tensor* out_tr = transpose(out);       // (D, N)
    set_name(out_tr, "out_tr"); print(out_tr);

    out_tr->backward(out_tr);
    graphviz(out_tr);

    return 0;
}
