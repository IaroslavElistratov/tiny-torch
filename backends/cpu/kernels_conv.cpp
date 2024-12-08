// #include "indexing.cpp" // imported though main -> ops.cpp -> indexing.cpp
#include <stdio.h> // sprintf

#define STRIDE 2


tensor* conv_k_(tensor* input, tensor* kernel, tensor* out) {

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    for (int f=0; f<F; f++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                // 0. select current filter

                // simple pointer arithmetic to skip from "f" kernels
                float* curr_kernel = kernel->data + f*kernel->stride[0];
                tensor* curr_filter = TensorNoData(C, HH, WW);
                curr_filter->data = curr_kernel;

                // 1. select the chunk of input
                int vert_start = hight * STRIDE;
                int vert_end = vert_start + HH;
                int horiz_start = width * STRIDE;
                int horiz_end = horiz_start + WW;

                char buffer[20];
                sprintf(buffer, "0:%i, %i:%i, %i:%i", C, vert_start, vert_end, horiz_start, horiz_end);
                tensor* x_slice = slice(input, buffer);

                // 2. element-wise multiply and sum
                tensor* curr_out = mul_k(x_slice, curr_filter);
                curr_out = reduce_sum_k(curr_out);

                out->data[index(out, f, hight, width)] = curr_out->data[0];
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

    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    tensor* out = EmptyTensor(F, h_out, w_out);
    return  conv_k_(input, kernel, out);
}

// conv output, upstream: (F, h_out, w_out)
void bwd_conv_k(tensor* upstream, tensor* out) {

    tensor* input = out->inputs[0];
    tensor* kernel = out->inputs[1];

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    // make sure it's not initialised with garbage
    tensor* grad_kernels = TensorLikeFill(kernel, 0.0);
    tensor* grad_x = TensorLikeFill(input, 0.0);

    for (int f=0; f<F; f++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                // select the chunk of input that this location in the ouput (f, h, w) is looking at
                int vert_start = hight * STRIDE;
                int vert_end = vert_start + HH;
                int horiz_start = width * STRIDE;
                int horiz_end = horiz_start + WW;

                // python: x_slice = input[:, vert_start:vert_end, horiz_start:horiz_end]
                char buffer[20];
                sprintf(buffer, "0:%i, %i:%i, %i:%i", C, vert_start, vert_end, horiz_start, horiz_end);
                tensor* x_slice = slice(input, buffer); // corresponding slice (was used in forward)

                // python: curr_filter = index(kernel, f); // (F, C, HH, WW) -> (C, HH, WW)
                tensor* curr_filter = TensorNoData(C, HH, WW); // filter (was used in forward)
                curr_filter->data = kernel->data + f*kernel->stride[0];

                // python: curr_upstream = upstream[f,h,w]
                float curr_upstream_float = upstream->data[index(upstream, f, hight, width)]; // scalar
                tensor* curr_upstream = TensorLikeFill(x_slice, curr_upstream_float); // broadcast scalar grad to the shape of the slice

                tensor* curr_downstream = mul_k(x_slice, curr_upstream);

                // record downstream grad of the current slice, into the larger tensor (for the downstream grad)

                // workaround for not having non owning slice 4d
                tensor* curr_downstream_slice_in_larger_tensor = TensorNoData(C, HH, WW);
                curr_downstream_slice_in_larger_tensor->data = grad_kernels->data + f*grad_kernels->stride[0];

                add_k_(curr_downstream_slice_in_larger_tensor, curr_downstream, curr_downstream_slice_in_larger_tensor);

                curr_downstream = mul_k(curr_filter, curr_upstream);

                // record downstream grad of the current slice, into the larger tensor (for the downstream grad)
                curr_downstream_slice_in_larger_tensor = view(grad_x, buffer);

                add_k_(curr_downstream_slice_in_larger_tensor, curr_downstream, curr_downstream_slice_in_larger_tensor);
            }
        }
    }

    kernel->grad = grad_kernels;
    input->grad = grad_x;
}



// x (B, C, H, W)
// w (F, C, HH, WW)
tensor* batched_conv_k(tensor* input, tensor* kernel){

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    tensor* out = EmptyTensor(B, F, h_out, w_out);

    for (int i=0; i<B; i++){
        tensor* curr_out = TensorNoData(F, h_out, w_out);
        curr_out->data = out->data + (i * out->stride[0]);

        tensor* curr_x = TensorNoData(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

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

    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    // make sure it's not initialised with garbage
    tensor* grad_x = TensorLikeFill(input, 0.0);
    tensor* grad_kernels = TensorLikeFill(kernel, 0.0);

    for (int i=0; i<B; i++){
        tensor* curr_x = TensorNoData(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

        tensor* curr_upstream = TensorNoData(F, h_out, w_out);
        curr_upstream->data = upstream->data + (i * upstream->stride[0]);

        tensor* curr_out = TensorNoData(F, h_out, w_out);
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

        for (int ii=0; ii<grad_kernels->size; ii++){
            grad_kernels->data[ii] = grad_kernels->data[ii] + curr_grad_filter->data[ii];
        }
    }

    input->grad = grad_x;
    kernel->grad = grad_kernels;
}



// x (C, H, W)
tensor* maxpool_k_(tensor* input, tensor* out) {

    // todo: up until and including "int horiz_end" line, was copied from conv. Reduce duplication.
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int HH = 2, WW = 2;

    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    for (int c=0; c<C; c++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                // 1. select the chunk of input that this location in the ouput (f, h, w) is looking at
                int vert_start = hight * STRIDE;
                int vert_end = vert_start + HH;
                int horiz_start = width * STRIDE;
                int horiz_end = horiz_start + WW;

                // select only 1 channel here
                int c_next = c + 1;
                char buffer[20];
                sprintf(buffer, "%i:%i, %i:%i, %i:%i", c, c_next, vert_start, vert_end, horiz_start, horiz_end);
                tensor* x_slice = view(input, buffer);

                // select maximum element
                // recompute during backward
                float max = x_slice->data[0];
                for (int i=0; i<x_slice->size; i++){
                    if (x_slice->data[at(x_slice, i)] > max) {
                        max = x_slice->data[at(x_slice, i)];
                    }
                }
                out->data[index(out, c, hight, width)] = max;
            }
        }
    }
    return out;
}


tensor* maxpool_k(tensor* input) {
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];

    int HH = 2, WW = 2;
    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    tensor* out = EmptyTensor(C, h_out, w_out);
    return  maxpool_k_(input, out);
}


void bwd_maxpool_k(tensor* upstream, tensor* out) {

    tensor* input = out->inputs[0];
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];

    int HH = 2, WW = 2;
    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    tensor* downstream = TensorLikeFill(input, 0.0);

    for (int c=0; c<C; c++){
        for (int hight=0; hight<h_out; hight++){
            for (int width=0; width<w_out; width++){

                int vert_start = hight * STRIDE;
                int vert_end = vert_start + HH;
                int horiz_start = width * STRIDE;
                int horiz_end = horiz_start + WW;

                char buffer[20];
                int c_next = c + 1;
                sprintf(buffer, "%i:%i, %i:%i, %i:%i", c, c_next, vert_start, vert_end, horiz_start, horiz_end);
                tensor* x_slice = view(input, buffer);

                // local
                tensor* local = TensorLikeFill(x_slice, 0.0);
                int idx_max = 0;
                float max = x_slice->data[0];
                for (int i=0; i<x_slice->size; i++){
                    if (x_slice->data[at(x_slice, i)] > max) {
                        max = x_slice->data[at(x_slice, i)];
                        // bc local is contiguous, record contiguous idx (not the x_slice's idx)
                        idx_max = i;
                    }
                }
                local->data[idx_max] = 1.0;

                // upstream
                float curr_upstream_float = upstream->data[index(upstream, c, hight, width)]; // scalar
                tensor* curr_upstream = TensorLikeFill(x_slice, curr_upstream_float); // broadcast scalar grad to the shape of the slice

                // downstream
                tensor* curr_downstream = mul_k(local, curr_upstream);

                // record downstream grad of the current slice, into the larger tensor (corresponding to the downstream grad)
                tensor* downstream_slice = view(downstream, buffer);
                for (int i=0; i<downstream_slice->size; i++){
                    downstream_slice->data[at(downstream_slice, i)] = curr_downstream->data[i];
                }

            }
        }
    }
    input->grad = downstream;
}



// x (B, C, H, W)
// w (F, C, HH, WW)
tensor* batched_maxpool_k(tensor* input){

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];

    int HH = 2, WW = 2;
    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    tensor* out = EmptyTensor(B, C, h_out, w_out);

    for (int i=0; i<B; i++){
        tensor* curr_out = TensorNoData(C, h_out, w_out);
        curr_out->data = out->data + (i * out->stride[0]);

        tensor* curr_x = TensorNoData(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

        maxpool_k_(curr_x, curr_out);
    }
    return out;
}


void bwd_batched_maxpool_k(tensor* upstream, tensor* out) {

    tensor* input = out->inputs[0];

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int HH = 2, WW = 2;

    int h_out = 1 + (H - HH) / STRIDE;
    int w_out = 1 + (W - WW) / STRIDE;

    tensor* downstream = TensorLikeFill(input, 0.0);

    for (int i=0; i<B; i++){
        tensor* curr_x = TensorNoData(C, H, W);
        curr_x->data = input->data + (i * input->stride[0]);

        tensor* curr_upstream = TensorNoData(C, h_out, w_out);
        curr_upstream->data = upstream->data + (i * upstream->stride[0]);

        tensor* curr_out = TensorNoData(C, h_out, w_out);
        curr_out->data = out->data + (i * out->stride[0]);

        // bwd_maxpool_k unpacks this
        curr_out->inputs[0] = curr_x;

        bwd_maxpool_k(curr_upstream, curr_out);
        tensor* curr_downstream = curr_x->grad;

        for (int ii=0; ii<curr_downstream->size; ii++){
            int offset_batch = i*downstream->stride[0];
            downstream->data[offset_batch + ii] = curr_downstream->data[ii];
        }
    }
    input->grad = downstream;
}
