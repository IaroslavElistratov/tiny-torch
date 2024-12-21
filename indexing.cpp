#include <stdarg.h>
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

// most of the logic here unpacks varying number of args so that idx_Nd can be called
int index(tensor* t, ...){
    va_list args;
    va_start(args, t);

    int idx;
    int s0 = va_arg(args, int);
    int s1 = va_arg(args, int);

    if (t->num_dims==2){
        idx = index_2d(t, s0, s1);
    } else if (t->num_dims==3){
        int s2 = va_arg(args, int);
        idx = index_3d(t, s0, s1, s2);
    } else if (t->num_dims==4){
        int s2 = va_arg(args, int);
        int s3 = va_arg(args, int);
        idx = index_4d(t, s0, s1, s2, s3);
    } else {
        printf("[index] unexpected t->num_dims: %i\n", t->num_dims);
        sprint(t);
        exit(1);
    }
    va_end(args);
    return idx;
}


// todo: name it "contigify"
// owning views:


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

    tensor* out = EmptyTensor(x, y, z);

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

tensor* slice(tensor* t, const char* dims){
    if (t->num_dims==2) return slice_2d(t, dims);
    else if (t->num_dims==3) return slice_3d(t, dims);
    else {
        printf("[slice] unexpected t->num_dims\n");
        exit(1);
    };
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

    tensor* out = TensorNoData(x, y, z);

    out->stride[0] = t->stride[0];
    out->stride[1] = t->stride[1];
    out->stride[2] = t->stride[2];

    out->data = &t->data[index_3d(t, starts[0], starts[1], starts[2])];
    return out;
}

tensor* view(tensor* t, const char* dims){
    if (t->num_dims==2) return view_2d(t, dims);
    else if (t->num_dims==3) return view_3d(t, dims);
    else {
        printf("[view] unexpected t->num_dims\n");
        exit(1);
    };
}


/*
Used in elementwise ops, which were previously implemented (see below), and this is not valid when input is not contiguous
    > for (int i=0; i<out->size; i++)
    >   out->data[i] = a->data[i] + b->data[i];
*/


int at_2d(tensor* t, int idx){
    int z = t->shape[1];
    // todo: y instead of stride -- bc want n in shapes here
    int y_idx = idx / z;
    int z_idx = idx % z;
    return index_2d(t, y_idx, z_idx);
}

int at_3d(tensor* t, int idx){
    // int x = t->shape[0];
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

int at(tensor* t, int idx){
    if (t->num_dims==2) return at_2d(t, idx);
    else if (t->num_dims==3) return at_3d(t, idx);
    else {
        printf("[at] unexpected t->num_dims (expected 2 or 3)\n");
        exit(1);
        // return -1;
    };
}
