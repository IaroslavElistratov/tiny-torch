#include <iostream> // todo: use C only
// #include <stdlib.h>
// #include <iomanip> // for  input-output manipulation
using namespace std;

// #include "nn.h"
#include "tensor.cpp"
#include "ops.cpp"
#include "utils.cpp"


#define NUM_EP 2
#define LR 0.02
#define DEBUG  1

#ifdef DEBUG
#define print(f_p) _print(f_p)
#else
#define print(f_p)
#endif


// const char* dims = ":, 4:11, 0";
// arrays of 3, bc 3 is max nums of dims at the moment
// int starts[3] = {-1, -1, -1};
// int ends[3] = {-1, -1, -1};
// int max_str_len = sizeof("*:*, *:*, *:*") / sizeof(char);
// for (int i=0; dims[i] != '\0' && i<max_str_len; i++){
// }

// todo: relax the constrains later
//  support ":"
//  support ":n" and "n:"
//  support omitting at the both ends


int index_2d(tensor* t, int y, int z){
    return t->stride[0]*y + t->stride[1]*z;
}

int index_3d(tensor* t, int x, int y, int z){
    return t->stride[0]*x + t->stride[1]*y + t->stride[2]*z;
}


/*
//index_s funcs are not convenient to use when ids are variables
s = "%i, %i".format(yi+starts[0], zi+starts[1])
int inp_idx = index_2d(t, s);

// better to use when they are constants
int inp_idx = index_3d(t, "10, 51, 9");
*/

// int index_s2d(tensor* t, const char* dims){
//     int idxs[2] = {dims[0]-'0', dims[3]-'0'};
//     return t->stride[0]*idxs[0] + t->stride[1]*idxs[1];
// }

// int index_s3d(tensor* t, const char* dims){
//     int idxs[3] = {dims[0]-'0', dims[3]-'0', dims[6]-'0'};
//     return t->stride[0]*idxs[0] + t->stride[1]*idxs[1] + t->stride[2]*idxs[2];
// }



tensor* slice_2d(tensor* t, const char* dims){

    // also converts char to int
    int starts[2] = {dims[0]-'0', dims[5]-'0'};
    int ends[2] = {dims[2]-'0', dims[7]-'0'};

    // lowercase to denote sizes of the slice, not of t
    int y = ends[0] - starts[0];
    int z = ends[1] - starts[1];

    tensor* out = EmptyTensor(y, z);

    // lower-case to denote dims of the slice

    for (int yi=0; yi<y; yi++){
        for (int zi=0; zi<z; zi++){
            int out_idx = index_2d(out, yi, zi);
            int inp_idx = index_2d(t, yi+starts[0], zi+starts[1]);
            out->data[out_idx] = t->data[inp_idx];
        }
    }
    return out;
}

// todo: can I make this re-use slice_2d?
tensor* slice_3d(tensor* t, const char* dims){

    // also converts char to int
    int starts[3] = {dims[0]-'0', dims[5]-'0', dims[10]-'0'};
    int ends[3] = {dims[2]-'0', dims[7]-'0', dims[12]-'0'};

    // lowercase to denote sizes of the slice, not of t
    int x = ends[0] - starts[0];
    int y = ends[1] - starts[1];
    int z = ends[2] - starts[2];

    // todo: TensorNoData
    tensor* out = _EmptyTensor(x, y, z);

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            for (int zi=0; zi<z; zi++){
                int out_idx = index_3d(out, xi, yi, zi);
                int inp_idx = index_3d(t, xi+starts[0], yi+starts[1], zi+starts[2]);
                out->data[out_idx] = t->data[inp_idx];
            }
        }
    }
    return out;
}

void print_3d(tensor* t)
{
    printf("\n%s: ", t->name);

    int x = t->shape[0];
    int y = t->shape[1];
    int z = t->shape[2];

    int strides_out[3] = {y*z, z, 1};

    printf("\n");

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            for (int zi=0; zi<z; zi++){
                int idx = index_3d(t, xi, yi, zi);
                printf("%8.4f, ", t->data[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}


int main() {
    srand(123);

    tensor* x = Tensor(3, 7);
    set_name(x, "x"); print(x);

    tensor* x_slice = slice_2d(x, "1:3, 3:6");
    print(x_slice);

    tensor* y = _Tensor(4, 3, 7);
    set_name(y, "y");
    print_3d(y);

    tensor* y_slice = slice_3d(y, "2:4, 1:3, 3:6");
    set_name(y_slice, "y_slice");
    print_3d(y_slice);

    // const char* dims = "0:1, 4:11";
    return 0;
}
