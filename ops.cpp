#include "nn.h"
#include "indexing.cpp"
#include "backends/select_backend.cpp"


// **** operations ****
//   - shared between CUDA and CPU kernels
//   - user-facing, unlike other abstractions
//   - call primitives, in addition, record data for the autograd:
//   - allocating grad buffers
//   - write local derivatives, for each input


// todo-now: temporarily commented out ops which don't yet have CUDA impl, so that compilation doesn't fail bc linker can't find _k expected by these ops

tensor* add(tensor* a, tensor* b) {
    a->num_uses++;
    b->num_uses++;
    tensor* t = add_k(a, b);
    t->is_leaf = false;
    // todo-low: check bool tensor->requires_grad before doing steps below, including allocating buffers

    // todo: this can be further abstracted -- creating a binary_op function
    // todo: and even further abstracted -- creating a binary_elementwise_op function

    // fill the additional info on out tensor
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->op_type = 0;
    t->grad_fn = add_bwd;
    return t;
}

tensor* sub(tensor* a, tensor* b) {
    a->num_uses++;
    b->num_uses++;
    tensor* t = sub_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->op_type = 1;
    t->grad_fn = sub_bwd;
    return t;
}

tensor* mul(tensor* a, tensor* b) {
    a->num_uses++;
    b->num_uses++;
    tensor* t = mul_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->op_type = 2;
    t->grad_fn = mul_bwd;
    return t;
}

tensor* matmul(tensor* a, tensor* b) {
    a->num_uses++;
    b->num_uses++;
    // e.g.: a(N, M) @ b(M, D) = t(N, D)
    tensor* t = matmul_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->op_type = 3;
    t->grad_fn = matmul_bwd;
    return t;
}

tensor* div(tensor* a, tensor* b) {
    a->num_uses++;
    b->num_uses++;
    tensor* t = div_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->op_type = 20;
    t->grad_fn = div_bwd;
    return t;
}

// tensor* repeat(tensor* a, int num_repeats) {
//     a->num_uses++;
//     tensor* t = repeat_k(a, num_repeats);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = a;
//     t->op_type = 18;
//     t->grad_fn = repeat_bwd;
//     return t;
// }

// tensor* select(tensor* a, tensor* idx) {
//     a->num_uses++;
//     idx->num_uses++;
//     tensor* t = select_k(a, idx);
//     t->is_leaf = false;
//     t->num_inputs = 2;
//     t->inputs[0] = a;
//     t->inputs[1] = idx;
//     t->op_type = 14;
//     t->grad_fn = select_bwd;
//     return t;
// }

tensor* pow(tensor* a, int exponent) {
    a->num_uses++;
    tensor* t = pow_k(a, exponent);
    t->is_leaf = false;
    //  comment: note by "inputs" I mean tensor inputs (INPUTS which I'll use compute grads wrt to)
    //  so here even if this op has two inputs, it really has one, for the purpose of the autograd
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 4;
    t->grad_fn = pow_bwd;
    return t;
}

// tensor* reduce_sum(tensor* a) {
//     a->num_uses++;
//     tensor* t = reduce_sum_k(a);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = a;
//     t->op_type = 5;
//     t->grad_fn = reduce_sum_bwd;
//     return t;
// }

// tensor* relu(tensor* a) {
//     a->num_uses++;
//     tensor* t = relu_k(a);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = a;
//     t->op_type = 6;
//     t->grad_fn = relu_bwd;
//     return t;
// }

tensor* transpose(tensor* a) {
    a->num_uses++;
    tensor* t = transpose_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 7;
    t->grad_fn = transpose_bwd;
    return t;
}

// tensor* max(tensor* a) {
//     a->num_uses++;
//     tensor* t = max_k(a);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = a;
//     t->op_type = 21;
//     t->grad_fn = max_bwd;
//     return t;
// }

tensor* neg(tensor* a) {
    a->num_uses++;
    tensor* t = neg_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 19;
    t->grad_fn = neg_bwd;
    return t;
}

// todo: does it conflict with C's math.exp ?
tensor* exp(tensor* a) {
    a->num_uses++;
    tensor* t = exp_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 16;
    t->grad_fn = exp_bwd;
    return t;
}

tensor* log(tensor* a) {
    a->num_uses++;
    tensor* t = log_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 15;
    t->grad_fn = log_bwd;
    return t;
}

tensor* batched_matmul(tensor* a, tensor* b) {
    a->num_uses++;
    b->num_uses++;
    // e.g.: a(B, N, M) @ b(B, M, D) = t(B, N, D)
    tensor* t = batched_matmul_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->op_type = 8;
    t->grad_fn = batched_matmul_bwd;
    return t;
}

// tensor* batched_flatten(tensor* a) {
//     a->num_uses++;
//     tensor* t = batched_flatten_k(a);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = a;
//     t->op_type = 13;
//     t->grad_fn = batched_flatten_bwd;
//     return t;
// }

// tensor* batched_reduce_sum(tensor* a) {
//     a->num_uses++;
//     tensor* t = batched_reduce_sum_k(a);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = a;
//     t->op_type = 17;
//     t->grad_fn = batched_reduce_sum_bwd;
//     return t;
// }

// tensor* batched_max(tensor* a) {
//     a->num_uses++;
//     tensor* t = batched_max_k(a);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = a;
//     t->op_type = 22;
//     t->grad_fn = batched_max_bwd;
//     return t;
// }

// tensor* conv(tensor* input, tensor* kernel) {
//     input->num_uses++;
//     kernel->num_uses++;
//     tensor* t = conv_k(input, kernel);
//     t->is_leaf = false;
//     t->num_inputs = 2;
//     t->inputs[0] = input;
//     t->inputs[1] = kernel;
//     t->op_type = 9;
//     t->grad_fn = bwd_conv_k;
//     return t;
// }

// tensor* batched_conv(tensor* input, tensor* kernel) {
//     input->num_uses++;
//     kernel->num_uses++;
//     tensor* t = batched_conv_k(input, kernel);
//     t->is_leaf = false;
//     t->num_inputs = 2;
//     t->inputs[0] = input;
//     t->inputs[1] = kernel;
//     t->op_type = 10;
//     t->grad_fn = bwd_batched_conv_k;
//     return t;
// }

// tensor* maxpool(tensor* input) {
//     input->num_uses++;
//     tensor* t = maxpool_k(input);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = input;
//     t->op_type = 11;
//     t->grad_fn = bwd_maxpool_k;
//     return t;
// }

// tensor* batched_maxpool(tensor* input) {
//     input->num_uses++;
//     tensor* t = batched_maxpool_k(input);
//     t->is_leaf = false;
//     t->num_inputs = 1;
//     t->inputs[0] = input;
//     t->op_type = 12;
//     t->grad_fn = bwd_batched_maxpool_k;
//     return t;
// }
