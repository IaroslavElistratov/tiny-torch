#include <iostream> // todo: use C only
#include <stdlib.h> // iot

using namespace std;

#include "parse.cpp"


// todo-low: use C++ to implement these as methods on the struct, so that don't need to explicitly pass first arg
// todo-high: implement "index", "slice", "view" funcs -- recursively? Removing need to manually impl for each new n_dim

int index_2d(tensor* t, int y, int z){
    return t->stride[0]*y + t->stride[1]*z;
}

int index_3d(tensor* t, int x, int y, int z){
    return t->stride[0]*x + t->stride[1]*y + t->stride[2]*z;
}

int index_4d(tensor* t, int o, int x, int y, int z){
    return t->stride[0]*o + t->stride[1]*x + t->stride[2]*y + t->stride[3]*z;
}


// owning views:
//   todo: name it "contigify"


tensor* slice_2d(tensor* t, const char* dims){

    // also converts char to int
    int* parsed_dims = parse_idxs(dims, 2);
    int* starts = parsed_dims;
    int* ends = parsed_dims + 2;

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

// todo: can make this re-use slice_2d?
tensor* slice_3d(tensor* t, const char* dims){


    // also converts char to int
    int* parsed_dims = parse_idxs(dims, 3);
    // int* starts = parsed_dims[0];
    // int* ends = parsed_dims[1];
    int* starts = parsed_dims;
    int* ends = parsed_dims + 3;

    // lowercase to denote sizes of the slice, not of t
    int x = ends[0] - starts[0];
    int y = ends[1] - starts[1];
    int z = ends[2] - starts[2];

    tensor* out = EmptyTensor3d(x, y, z);

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


// non-owning views:


tensor* view_2d(tensor* t, const char* dims){

    // also converts char to int
    int* parsed_dims = parse_idxs(dims, 2);
    int* starts = parsed_dims;
    int* ends = parsed_dims + 2;

    // lowercase to denote sizes of the slice, not of t
    int y = ends[0] - starts[0];
    int z = ends[1] - starts[1];

    tensor* out = TensorNoData(y, z);
    // the default constructor sets strides based on the shapes provided to the constructor.
    // This is correct in general, however here heed to change
    out->stride[0] = t->stride[0];
    out->stride[1] = t->stride[1]; // this is more general than setting to 1

    // data should point to the first element (of the view) in the original tensor
    out->data = &t->data[index_2d(t, starts[0], starts[1])];

    // comment:  
    // no need to loop since in this fn no need to copy or even access the elements
    // as oppose to (slice_2d, slice_3d)

    return out;
}

tensor* view_3d(tensor* t, const char* dims){
    // also converts char to int
    int* parsed_dims = parse_idxs(dims, 3);
    int* starts = parsed_dims;
    int* ends = parsed_dims + 3;

    // lowercase to denote sizes of the slice, not of t
    int x = ends[0] - starts[0];
    int y = ends[1] - starts[1];
    int z = ends[2] - starts[2];

    tensor* out = TensorNoData3d(x, y, z);

    out->stride[0] = t->stride[0];
    out->stride[1] = t->stride[1];
    out->stride[2] = t->stride[2];

    out->data = &t->data[index_3d(t, starts[0], starts[1], starts[2])];
    return out;
}


/*
Used in elementwise ops, which were previously implemented (see below), and this is not valid when input is not contiguous
    > for (int i=0; i<out->size; i++)
    >   out->data[i] = a->data[i] + b->data[i];
*/


int at_2d(tensor* t, int idx)
{
    int z = t->shape[1];
    // todo: y instead of stride -- bc want n in shapes here
    int y_idx = idx / z;
    int z_idx = idx % z;
    return index_2d(t, y_idx, z_idx);
}

int at_3d(tensor* t, int idx)
{
    int x = t->shape[0];
    int y = t->shape[1];
    int z = t->shape[2];

    // num elements in x: y*z
    int x_idx = idx / (y*z);
    // remaining idx
    idx -= x_idx * (y*z);

    // num elements in y: z
    int y_idx = idx / z;
    idx -= (y_idx * z);

    // remaining z
    int z_idx = idx % z;

    return index_3d(t, x_idx, y_idx, z_idx);
}
