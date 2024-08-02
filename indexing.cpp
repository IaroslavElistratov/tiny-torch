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

tensor* slice_2d(tensor* t, const char* dims){

    // also converts char to int
    int starts[2] = {dims[0]-'0', dims[5]-'0'};
    int ends[2] = {dims[2]-'0', dims[7]-'0'};

    cout << "starts[0]: " << starts[0] << endl;

    // lowercase to denote sizes of the slice, not of t
    int y = ends[0] - starts[0];
    int z = ends[1] - starts[1];

    tensor* out = EmptyTensor(y, z);

    // strides_out
    int Z = t->shape[1];
    // upper-case to denote dims of original input
    int strides_in[2] = {Z, 1};
    // lower-case to denote dims of the slice
    int strides_out[2] = {z, 1};

    for (int yi=0; yi<y; yi++){
        for (int zi=0; zi<z; zi++){
            int out_idx = strides_out[0]*yi + strides_out[1]*zi;
            int inp_idx = strides_in[0]*(yi+starts[0]) + strides_in[1]*(zi+starts[1]);
            out->data[out_idx] = t->data[inp_idx];
            // cout << "out_idx: " << out_idx << endl;
            // cout << "inp_idx: " << inp_idx << endl << endl;
        }
    }
    return out;
}

    const char* dims = "0:1, 4:11, 0:2";

// todo: can I make this re-use slice_2d?
tensor* slice_3d(tensor* t, const char* dims){

    // also converts char to int
    int starts[3] = {dims[0]-'0', dims[5]-'0', dims[10]-'0'};
    int ends[3] = {dims[2]-'0', dims[7]-'0', dims[12]-'0'};

    // lowercase to denote sizes of the slice, not of t
    int x = ends[0] - starts[0];
    int y = ends[1] - starts[1];
    int z = ends[2] - starts[2];

    // todo-now: 3d tensor
    tensor* out = _EmptyTensor(x, y, z);

    int Y = t->shape[1];
    int Z = t->shape[2];

    int strides_in[3] = {Y*Z, Z, 1};
    int strides_out[3] = {y*z, z, 1};

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            for (int zi=0; zi<z; zi++){
                int out_idx = strides_out[0]*xi + strides_out[1]*yi + strides_out[2]*zi;
                int inp_idx = strides_in[0]*(xi+starts[0]) + strides_in[1]*(yi+starts[1]) + strides_in[2]*(zi+starts[2]);
                out->data[out_idx] = t->data[inp_idx];
                // cout << "out_idx: " << out_idx << endl;
                // cout << "inp_idx: " << inp_idx << endl << endl;
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
                int idx = strides_out[0]*xi + strides_out[1]*yi + strides_out[2]*zi;
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

    tensor* x_slice = slice_2d(x, "1:3, 4:7");
    print(x_slice);

    tensor* y = _Tensor(4, 3, 7);
    set_name(y, "y");
    print_3d(y);

    tensor* y_slice = slice_3d(y, "2:4, 1:3, 4:7");
    set_name(y_slice, "y_slice");
    print_3d(y_slice);

    // const char* dims = "0:1, 4:11";
    return 0;
}
