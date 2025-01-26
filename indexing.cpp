#include <stdarg.h>


// todo: support omitting at the both ends
//   auto-filling missing dims -- allows "kernel[0]" instead of below
//   curr_filter = slice_4d(kernel, "f, 0:C, 0:HH, 0:WW"); // (F, C, HH, WW) -> (C, HH, WW)


// todo:
//  support ":"
//  support ":n" and "n:"
//  support omitting at the both ends
//      auto-filling missing dims -- allows "kernel[0]" instead of below
//      curr_filter = slice_4d(kernel, "f, 0:C, 0:HH, 0:WW"); // (F, C, HH, WW) -> (C, HH, WW)

struct ax {
    int start;
    int end;
};

ax axis(int start, int end){
    ax out = {start, end};
    return out;
}




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


// comment: unlike my regular VA_NARGS -- this does -1
// #define VA_NARGS_IMPL_MINUS1(_0, _1, _2, _3, _4, N, ...) N
// #define VA_NARGS_MINUS1(...) VA_NARGS_IMPL_MINUS1(__VA_ARGS__, 4, 3, 2, 1, 0)
// #define index(t, ...) CONCAT(index_, CONCAT(VA_NARGS_MINUS1(__VA_ARGS__), d))(t, __VA_ARGS__)
#define index(t, ...) CONCAT(index_, CONCAT(VA_NARGS(__VA_ARGS__), d))(t, __VA_ARGS__)


// todo: name it "contigify"
// owning views:


tensor* slice_2d(tensor* t, ax axis0, ax axis1){

    // lowercase to denote sizes of the slice, not of t
    int y = axis0.end - axis0.start;
    int z = axis1.end - axis1.start;

    tensor* out = EmptyTensor(y, z);

    // lower-case to denote dims of the slice
    for (int yi=0; yi<y; yi++){
        for (int zi=0; zi<z; zi++){
            int out_idx = index_2d(out, yi, zi);
            int inp_idx = index_2d(t, yi+axis0.start, zi+axis1.start);
            out->data[out_idx] = t->data[inp_idx];
        }
    }
    return out;
}

// todo: can make this re-use slice_2d?
tensor* slice_3d(tensor* t, ax axis0, ax axis1, ax axis2){

    // lowercase to denote sizes of the slice, not of t
    int x = axis0.end - axis0.start;
    int y = axis1.end - axis1.start;
    int z = axis2.end - axis2.start;

    tensor* out = EmptyTensor(x, y, z);

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            for (int zi=0; zi<z; zi++){
                int out_idx = index_3d(out, xi, yi, zi);
                int inp_idx = index_3d(t, xi+axis0.start, yi+axis1.start, zi+axis2.start);
                out->data[out_idx] = t->data[inp_idx];
            }
        }
    }
    return out;
}


#define slice(t, ...) CONCAT(slice_, CONCAT(VA_NARGS(__VA_ARGS__), d))(t, __VA_ARGS__)



// non-owning views:

tensor* view_2d(tensor* t, ax axis0, ax axis1){

    // lowercase to denote sizes of the slice, not of t
    int y = axis0.end - axis0.start;
    int z = axis1.end - axis1.start;

    tensor* out = TensorNoData(y, z);
    // the default constructor sets strides based on the shapes provided to the constructor.
    // This is correct in general, however here heed to change
    out->stride[0] = t->stride[0];
    out->stride[1] = t->stride[1]; // this is more general than setting to 1

    // data should point to the first element (of the view) in the original tensor
    out->data = &t->data[index_2d(t, axis0.start, axis1.start)];

    // comment:  
    // no need to loop since in this fn no need to copy or even access the elements
    // as oppose to (slice_2d, slice_3d)

    return out;
}

tensor* view_3d(tensor* t, ax axis0, ax axis1, ax axis2){
    // lowercase to denote sizes of the slice, not of t
    int x = axis0.end - axis0.start;
    int y = axis1.end - axis1.start;
    int z = axis2.end - axis2.start;

    tensor* out = TensorNoData(x, y, z);

    out->stride[0] = t->stride[0];
    out->stride[1] = t->stride[1];
    out->stride[2] = t->stride[2];

    out->data = &t->data[index_3d(t, axis0.start, axis1.start, axis2.start)];
    return out;
}

#define view(t, ...) CONCAT(view_, CONCAT(VA_NARGS(__VA_ARGS__), d))(t, __VA_ARGS__)



/*
Used in elementwise ops, which were previously implemented (see below), and this is not valid when input is not contiguous
    > for (int i=0; i<out->size; i++)
    >   out->data[i] = a->data[i] + b->data[i];
*/


int at_2d(tensor* t, int idx){
    int z = t->shape[1];
    // z instead of stride -- bc want num elements in shapes here
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

int at_4d(tensor* t, int idx){
    // int x = t->shape[0];
    int y = t->shape[1];
    int z = t->shape[2];
    int o = t->shape[3];

    // num elements in x: y*z*o
    int x_idx = idx / (y*z*o);
    // remaining idx
    idx -= x_idx * (y*z*o);

    // num elements in y: z*o
    int y_idx = idx / (z*o);
    idx -= y_idx * (z*o);

    // num elements in z: o
    int z_idx = idx / o;
    idx -= (z_idx * o);

    // remaining o
    int o_idx = idx % o;

    return index_4d(t, x_idx, y_idx, z_idx, o_idx);
}

int at(tensor* t, int idx){
    if (idx > t->size){
        printf("[at] index cannot be greater than t->size\n");
        exit(1);
    }
    if (t->num_dims==2) return at_2d(t, idx);
    else if (t->num_dims==3) return at_3d(t, idx);
    else if (t->num_dims==4) return at_4d(t, idx);
    else {
        printf("[at] unexpected t->num_dims (2, 3, or 4)\n");
        exit(1);
    };
}
