#include <iostream> // todo: use C only


#include <cstring>


// for autoguard
#include <deque> // deque from standard template library (STL)

// Mersenne Twister
#include <random>

#define DEVICE CUDA
#define N_SAMPLES 10000

#define PYTHON_API true


extern "C" {
#include "../nn.h"
}

void free_all_tensors(int a){
};

void add_to_gc(tensor* a){
};

int GC_IDX = 0;

extern "C" {
#include "../tensor.cpp"
#include "../ops.cpp"
#include "../cifar10.cpp"
#include "../print.cpp"
#include "../optim.cpp"
#include "../codegen.cpp"
#include "../serialization.cpp"

// these funcs are only required for my python API
#include "./python_bindings.cpp"

}


// int main(void) {
//     srand(123);
//     set_backend_device();
//     fclose(fopen("./generated/log.txt", "w"));

//     int gc_until = GC_IDX;

//     // each epoch
//     free_all_tensors(gc_until);

//     // cudaDeviceReset();
//     return 0;
// }
