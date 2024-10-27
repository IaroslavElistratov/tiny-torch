// #include "indexing.cpp" // imported though main -> ops.cpp -> indexing.cpp
#include <stdio.h> // sprintf

#define IS_DEBUG false
#define IS_DEBUG_MP false


/*
x: Input data of shape (C, H, W)
w: Filter weights of shape (F, C, HH, WW)

todo: implement pytorch's C optional as way to support tunable args
  - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
  - 'pad': The number of pixels that will be used to zero-pad the input. 
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides) along the height and width axes of the input.

Returns a tuple of:
- out: Output data, of shape (F, H', W')
*/
tensor* conv_k_(tensor* input, tensor* kernel, tensor* out) {

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    int stride = 2, pad = 0;
    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    if (IS_DEBUG){
        printf("[conv_k_] h_out: %i\n", h_out);
        printf("[conv_k_] w_out: %i\n", w_out);
    }

    for (int f=0; f<F; f++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                // 0. select current filter

                // todo: slice
                //   - need 4d slice? not necessarily, can just do pointer arithmetic to skip to the right kernel -- but problem is that it unwraps tensor to float, but kernels expect tensor
                //   - have the unified mechansm for all slices -- replace "f*grad_kernels->stride[0];" with slice 4d

                if (IS_DEBUG)
                    printf("[conv_k_] f*C*HH*WW: %i\n", f*C*HH*WW);

                // simple pointer arithmetic to skip from "f" kernels
                float* curr_kernel = kernel->data + f*kernel->stride[0];

                // workaround to put the data into tensor type
                // todo-high:: constructor does NOT take care of setting correct strides
                tensor* curr_filter = TensorNoData3d(C, HH, WW);
                curr_filter->data = curr_kernel;

                if (IS_DEBUG){
                    set_name(curr_filter, "curr_filter"); print_3d(curr_filter);
                    printf("[conv_k_] f*C*HH*WW: %i\n", f*kernel->stride[0]);
                }

                // 1. select the chunk of input that this location in the ouput (f, h, w) is looking at
                int vert_start = hight * stride;
                int vert_end = vert_start + HH;
                int horiz_start = width * stride;
                int horiz_end = horiz_start + WW;

                // e.g. "0:3, 30:32, 30:32" is 18 digits
                char buffer[20];
                // sprintf -- create strings with specified formats, similar to printf(), but instead of printing to the standard output, it stores the resulting string in a character array provided by the user
                sprintf(buffer, "0:%i, %i:%i, %i:%i", C, vert_start, vert_end, horiz_start, horiz_end);
                // todo: use view instead of slice
                //   - mul_k (used below) -- needs to use .at when looping over a->size
                tensor* x_slice = slice_3d(input, buffer);

                if (IS_DEBUG){
                    printf("[conv_k_] buffer x_slice %s\n", buffer);
                    printf("[conv_k_] x_slice->shape: %i, %i, %i\n", x_slice->shape[0], x_slice->shape[1], x_slice->shape[2]);
                    set_name(x_slice, "x_slice"); print_3d(x_slice);
                }

                // 2. element-wise multiply and sum
                tensor* curr_out = mul_k(x_slice, curr_filter);
                if (IS_DEBUG){
                    printf("[conv_k_] curr_out->shape: %i, %i, %i\n", curr_out->shape[0], curr_out->shape[1], curr_out->shape[2]);
                    set_name(curr_out, "curr_out"); print(curr_out);
                }

                // todo: add flag to tensors, is_contiguous and make each op check that flag and err otherwise
                curr_out = reduce_sum_k(curr_out);

                // 3. write to the current location at the output
                if (IS_DEBUG)
                    printf("[conv_k_] curr_out->data[0]: %f\n", curr_out->data[0]);
                out->data[index_3d(out, f, hight, width)] = curr_out->data[0];
            }
        }
    }
    return out;
}

// input (C, H, W)
// kernel (F, C, HH, WW)
tensor* conv_k(tensor* input, tensor* kernel) {
    int H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    // todo: de-duplicate same calculations of h_out, w_out for all the kernels in this file
    int stride = 2, pad = 0;
    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    if (IS_DEBUG){
        printf("[conv_k] h_out: %i\n", h_out);
        printf("[conv_k] w_out: %i\n", w_out);
    }

    tensor* out = EmptyTensor3d(F, h_out, w_out);
    return  conv_k_(input, kernel, out);
}

// conv output, upstream: (F, h_out, w_out)
void bwd_conv_k(tensor* upstream, tensor* out) {

    tensor* input = out->inputs[0];
    tensor* kernel = out->inputs[1];

    if (IS_DEBUG)
        printf("[bwd_conv_k] input->shape: %i, %i, %i\n", input->shape[0], input->shape[1], input->shape[2]);

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    int stride = 2, pad = 0;
    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    // make sure it's not initialised with garbage
    tensor* grad_kernels = TensorLikeFill4d(kernel, 0.0);
    tensor* grad_x = TensorLikeFill3d(input, 0.0);

    for (int f=0; f<F; f++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                // select the chunk of input that this location in the ouput (f, h, w) is looking at
                int vert_start = hight * stride;
                int vert_end = vert_start + HH;
                int horiz_start = width * stride;
                int horiz_end = horiz_start + WW;

                // python: x_slice = input[:, vert_start:vert_end, horiz_start:horiz_end]
                char buffer[20];
                sprintf(buffer, "0:%i, %i:%i, %i:%i", C, vert_start, vert_end, horiz_start, horiz_end);
                tensor* x_slice = slice_3d(input, buffer); // corresponding slice (was used in forward)
                if (IS_DEBUG){
                    printf("[bwd_conv_k] buffer x_slice %s\n", buffer);
                    printf("[bwd_conv_k] x_slice->shape: %i, %i, %i\n", x_slice->shape[0], x_slice->shape[1], x_slice->shape[2]);
                    set_name(x_slice, "x_slice"); print_3d(x_slice);
                }

                // python: curr_filter = index(kernel, f); // (F, C, HH, WW) -> (C, HH, WW)
                tensor* curr_filter = TensorNoData3d(C, HH, WW); // filter (was used in forward)
                curr_filter->data = kernel->data + f*kernel->stride[0];
                if (IS_DEBUG){
                    printf("[bwd_conv_k] f*C*HH*WW: %i\n", f*C*HH*WW);
                    set_name(curr_filter, "curr_filter"); print_3d(curr_filter);
                }


                // python:
                //    current_dout = dout[i, f, hight, width]                           # current up-stream grad scalar
                //    current_x = x[i, :, vert_start:vert_end, horiz_start:horiz_end]   # corresponding slice (was used in forward)
                //    dw[f] += current_x * current_dout
                // 1. local grad of a kernel applied to an x_slice is x_slice
                // 2. then element-wise multiply that local grad with upstream
                // 3. then because there's multiple locations in x (x patches) we slid the kernel through -- elementwise sum the grad

                // python: curr_upstream = upstream[f,h,w]
                float curr_upstream_float = upstream->data[index_3d(upstream, f, hight, width)]; // scalar
                tensor* curr_upstream = TensorLikeFill3d(x_slice, curr_upstream_float); // broadcast scalar grad to the shape of the slice
                if (IS_DEBUG){
                    printf("[bwd_conv_k] curr_upstream_float: %f", curr_upstream_float);
                    printf("\n[bwd_conv_k] curr_upstream.shape: %i, %i, %i", curr_upstream->shape[0], curr_upstream->shape[1], curr_upstream->shape[2]);
                    set_name(curr_upstream, "curr_upstream"); print_3d(curr_upstream);
                }

                // kernel->grad = add(kernel->grad, mul_k(x_slice, curr_upstream))
                tensor* curr_downstream = mul_k(x_slice, curr_upstream);

                // record downstream grad of the current slice, into the larger tensor (for the downstream grad)

                // workaround for not having non owning slice 4d
                //    todo-high: constructor does not set correct strides in this case
                tensor* curr_downstream_slice_in_larger_tensor = TensorNoData3d(C, HH, WW);
                curr_downstream_slice_in_larger_tensor->data = grad_kernels->data + f*grad_kernels->stride[0];

                add_k_(curr_downstream_slice_in_larger_tensor, curr_downstream, curr_downstream_slice_in_larger_tensor);


                // python:
                //    dx[i, :, vert_start:vert_end, horiz_start:horiz_end] += current_w * current_dout
                // 1. local grad of a x_patch applied to an kernel is kernel
                // 2. then element-wise multiply that local grad with upstream
                // 3. then because there's multiple locations in x (x patches) we slid the kernel through -- elementwise sum the grad

                curr_downstream = mul_k(curr_filter, curr_upstream);

                // record downstream grad of the current slice, into the larger tensor (for the downstream grad)
                curr_downstream_slice_in_larger_tensor = view_3d(grad_x, buffer);

                add_k_(curr_downstream_slice_in_larger_tensor, curr_downstream, curr_downstream_slice_in_larger_tensor);
            }
        }
    }

    kernel->grad = grad_kernels;
    input->grad = grad_x;
}

tensor* conv(tensor* input, tensor* kernel) {
    input->num_uses++;
    kernel->num_uses++;
    tensor* t = conv_k(input, kernel);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = input;
    t->inputs[1] = kernel;
    t->op_type = 9;
    t->grad_fn = bwd_conv_k;
    return t;
}



// x (B, C, H, W)
// w (F, C, HH, WW)
tensor* batched_conv_k(tensor* input, tensor* kernel){

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    int stride = 2, pad = 0;
    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    if (IS_DEBUG){
        printf("[batched_conv_k] h_out: %i\n", h_out);
        printf("[batched_conv_k] w_out: %i\n", w_out);
    }

    tensor* out = EmptyTensor4d(B, F, h_out, w_out);

    for (int i=0; i<B; i++){
        // comment: same semantics as in batched_matmul_k

        tensor* curr_out = TensorNoData3d(F, h_out, w_out);
        curr_out->data = out->data + (i * out->stride[0]);

        tensor* curr_x = TensorNoData3d(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

        // comment: add support for 5d tensors? -- NO, w stays 4d
        conv_k_(curr_x, kernel, curr_out);
    }
    return out;
}

// x (B, C, H, W)
// w (F, C, HH, WW)
// conv output; upstream: (B, F, h_out, w_out)
void bwd_batched_conv_k(tensor* upstream, tensor* out) {

    tensor* input = out->inputs[0];
    tensor* kernel = out->inputs[1];

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    int stride = 2, pad = 0;
    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;
    if (IS_DEBUG){
        printf("[bwd_batched_conv_k] h_out: %i\n", h_out);
        printf("[bwd_batched_conv_k] w_out: %i\n", w_out);
    }

    // make sure it's not initialised with garbage
    tensor* grad_x = TensorLikeFill4d(input, 0.0);
    tensor* grad_kernels = TensorLikeFill4d(kernel, 0.0);

    for (int i=0; i<B; i++){
        // comment: same semantics as in batched_matmul_k

        tensor* curr_x = TensorNoData3d(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

        tensor* curr_upstream = TensorNoData3d(F, h_out, w_out);
        curr_upstream->data = upstream->data + (i * upstream->stride[0]);

        tensor* curr_out = TensorNoData3d(F, h_out, w_out);
        curr_out->data = out->data + (i * out->stride[0]);

        // bwd_conv_k unpacks this
        curr_out->inputs[0] = curr_x;
        curr_out->inputs[1] = kernel;

        bwd_conv_k(curr_upstream, curr_out);
        tensor* curr_grad_x = curr_x->grad; // set by bwd_conv_k
        tensor* curr_grad_filter = kernel->grad; // set by bwd_conv_k

        for (int ii=0; ii<curr_grad_x->size; ii++){
            int offset_batch = i * grad_x->stride[0];
            grad_x->data[offset_batch + ii] = grad_x->data[offset_batch + ii] + curr_grad_x->data[ii];
        }

        // can iterate over the grad_kernels bc regardless that we're feeding to bwd_conv_k inputs
        // only for the current b, we're feeding entire kernel (bc there's no batch dim in kernels)
        for (int ii=0; ii<grad_kernels->size; ii++){
            grad_kernels->data[ii] = grad_kernels->data[ii] + curr_grad_filter->data[ii];
        }
        // todo-low:
        // add_k_(curr_grad_filter, grad_kernels, grad_kernels);
    }

    input->grad = grad_x;
    kernel->grad = grad_kernels;
}

tensor* batched_conv(tensor* input, tensor* kernel) {
    input->num_uses++;
    kernel->num_uses++;
    tensor* t = batched_conv_k(input, kernel);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = input;
    t->inputs[1] = kernel;
    t->op_type = 10;
    t->grad_fn = bwd_batched_conv_k;
    return t;
}



// x (C, H, W)
tensor* maxpool_k_(tensor* input, tensor* out) {

    // todo: up until and including "int horiz_end" line, was copied from conv. Reduce duplication.

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];

    // hyperparameters
    int HH = 2, WW = 2;
    int stride = 2, pad = 0;

    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    if (IS_DEBUG_MP){
        printf("[maxpool_k_] h_out: %i\n", h_out);
        printf("[maxpool_k_] w_out: %i\n", w_out);
    }

    for (int c=0; c<C; c++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                if (IS_DEBUG_MP)
                    printf("[maxpool_k_] c*C*HH*WW: %i\n", c*C*HH*WW);
                // 1. select the chunk of input that this location in the ouput (f, h, w) is looking at
                int vert_start = hight * stride;
                int vert_end = vert_start + HH;
                int horiz_start = width * stride;
                int horiz_end = horiz_start + WW;

                // select only 1 channel here
                int c_next = c + 1;
                char buffer[20];
                sprintf(buffer, "%i:%i, %i:%i, %i:%i", c, c_next, vert_start, vert_end, horiz_start, horiz_end);
                tensor* x_slice = view_3d(input, buffer);
                if (IS_DEBUG_MP){
                    printf("[maxpool_k_] buffer x_slice %s\n", buffer);
                    printf("[maxpool_k_] x_slice->shape: %i, %i, %i\n", x_slice->shape[0], x_slice->shape[1], x_slice->shape[2]);
                    set_name(x_slice, "x_slice"); print_3d(x_slice);
                }

                // select maximum element
                // need to recompute this during backward (so that you put local grad 1 into the lications where there was maximum element)
                float max = x_slice->data[0];
                for (int i=0; i<x_slice->size; i++){
                    if (x_slice->data[at_3d(x_slice, i)] > max) {
                        // comment: 
                        //  crucial to use at_3d here and not simple "x_slice->data[i]" bc x_slice is a view thus it's NON contiguous!
                        //  alternatively, can create x_slice with "slice" instead of "view" -- the former will make a contiguous copy, in which case fine to index with "x_slice->data[i]
                        max = x_slice->data[at_3d(x_slice, i)];
                    }
                }
                if (IS_DEBUG_MP)
                    printf("[maxpool_k_] max: %f\n", max);
                out->data[index_3d(out, c, hight, width)] = max;
            }
        }
    }
    return out;
}

// x (C, H, W)
// w (F, C, HH, WW)
// todo: was copy/pate from conv_k -- reduce duplication
tensor* maxpool_k(tensor* input) {
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];

    // hyperparameters
    int HH = 2, WW = 2;
    int stride = 2, pad = 0;

    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    if (IS_DEBUG_MP){
        printf("[maxpool_k] h_out: %i\n", h_out);
        printf("[maxpool_k] w_out: %i\n", w_out);
    }

    tensor* out = EmptyTensor3d(C, h_out, w_out);
    return  maxpool_k_(input, out);
}

// todo-high:
//  - most of below is copy-pasted from the bwd_conv_k
//  - it's wasteful to have maxpool logic (65 lines) duplicated exactly in its bwd (which needed for re-computing local grad) -- instead just add another filed on the tensor called local_grad
void bwd_maxpool_k(tensor* upstream, tensor* out) {

    tensor* input = out->inputs[0];

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];

    // hyperparams
    int HH = 2, WW = 2;
    int stride = 2, pad = 0;

    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    tensor* downstream = TensorLikeFill3d(input, 0.0);

    for (int c=0; c<C; c++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                if (IS_DEBUG_MP)
                    printf("[bwd_maxpool_k] c*C*HH*WW: %i\n", c*C*HH*WW);

                // 1. select the chunk of input that this location in the ouput (f, h, w) is looking at
                int vert_start = hight * stride;
                int vert_end = vert_start + HH;
                int horiz_start = width * stride;
                int horiz_end = horiz_start + WW;

                char buffer[20];
                int c_next = c + 1;
                sprintf(buffer, "%i:%i, %i:%i, %i:%i", c, c_next, vert_start, vert_end, horiz_start, horiz_end);
                tensor* x_slice = view_3d(input, buffer);

                // local
                tensor* local = TensorLikeFill3d(x_slice, 0.0);
                int idx_max = 0;
                float max = x_slice->data[0];
                for (int i=0; i<x_slice->size; i++){
                    if (x_slice->data[at_3d(x_slice, i)] > max) {
                        max = x_slice->data[at_3d(x_slice, i)];
                        // bc local is contiguous, record contiguous idx (not the x_slice's idx)
                        idx_max = i;
                    }
                }
                local->data[idx_max] = 1.0;
                if (IS_DEBUG_MP)
                    set_name(local, "local"), print_3d(local);

                // upstream
                float curr_upstream_float = upstream->data[index_3d(upstream, c, hight, width)]; // scalar
                tensor* curr_upstream = TensorLikeFill3d(x_slice, curr_upstream_float); // broadcast scalar grad to the shape of the slice

                // downstream
                tensor* curr_downstream = mul_k(local, curr_upstream);
                if (IS_DEBUG_MP)
                    set_name(curr_downstream, "curr_downstream"), print_3d(curr_downstream);

                // record downstream grad of the current slice, into the larger tensor (corresponding to the downstream grad)
                tensor* downstream_slice = view_3d(downstream, buffer);
                // todo-low: use _copy_arr instead of the below; modify that fn to use at instead of t->data[i]
                for (int i=0; i<downstream_slice->size; i++){
                    // note you use at_3d on the downstream_slice bc it's a slice and therefore it's not contiguous, on the
                    // other hand, curr_downstream is contiguous, so simple curr_downstream->data[i] suffices
                    downstream_slice->data[at_3d(downstream_slice, i)] = curr_downstream->data[i];
                }

            }
        }
    }
    input->grad = downstream;
}

tensor* maxpool(tensor* input) {
    input->num_uses++;
    tensor* t = maxpool_k(input);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = input;
    t->op_type = 11;
    t->grad_fn = bwd_maxpool_k;
    return t;
}



// x (B, C, H, W)
// w (F, C, HH, WW)
// todo: copy/paste from batched_conv_k -- reduce duplication
tensor* batched_maxpool_k(tensor* input){

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];

    int HH = 2, WW = 2;
    int stride = 2, pad = 0;

    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;

    if (IS_DEBUG_MP){
        printf("[batched_maxpool_k] h_out: %i\n", h_out);
        printf("[batched_maxpool_k] w_out: %i\n", w_out);
    }

    tensor* out = EmptyTensor4d(B, C, h_out, w_out);

    for (int i=0; i<B; i++){
        tensor* curr_out = TensorNoData3d(C, h_out, w_out);
        curr_out->data = out->data + (i * out->stride[0]);

        tensor* curr_x = TensorNoData3d(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

        maxpool_k_(curr_x, curr_out);
    }
    return out;
}

// todo: copy from bwd_batched_conv_k -- reduce duplication
void bwd_batched_maxpool_k(tensor* upstream, tensor* out) {

    tensor* input = out->inputs[0];

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];

    int HH = 2, WW = 2;
    int stride = 2, pad = 0;

    int h_out = 1 + (H + 2 * pad - HH) / stride;
    int w_out = 1 + (W + 2 * pad - WW) / stride;
    if (IS_DEBUG_MP){
        printf("[bwd_batched_maxpool_k] h_out: %i\n", h_out);
        printf("[bwd_batched_maxpool_k] w_out: %i\n", w_out);
    }

    // make sure it's not initialised with garbage
    tensor* downstream = TensorLikeFill4d(input, 0.0);

    for (int i=0; i<B; i++){
        // comment: same semantics as in batched_matmul_k

        tensor* curr_x = TensorNoData3d(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

        tensor* curr_upstream = TensorNoData3d(C, h_out, w_out);
        curr_upstream->data = upstream->data + (i * upstream->stride[0]);

        tensor* curr_out = TensorNoData3d(C, h_out, w_out);
        curr_out->data = out->data + (i * out->stride[0]);

        // bwd_maxpool_k unpacks this
        curr_out->inputs[0] = curr_x;

        bwd_maxpool_k(curr_upstream, curr_out);
        tensor* curr_downstream = curr_x->grad; // set by bwd_maxpool_k

        for (int ii=0; ii<curr_downstream->size; ii++){
            int offset_batch = i*downstream->stride[0];
            downstream->data[offset_batch + ii] = curr_downstream->data[ii];
        }
        // todo-low:
        // add_k_(curr_downstream, downstream, downstream);
    }
    input->grad = downstream;
}

tensor* batched_maxpool(tensor* input) {
    input->num_uses++;
    tensor* t = batched_maxpool_k(input);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = input;
    t->op_type = 12;
    t->grad_fn = bwd_batched_maxpool_k;
    return t;
}
