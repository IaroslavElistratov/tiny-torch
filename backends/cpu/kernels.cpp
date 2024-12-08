#include <math.h> // log, pow


tensor* neg_k(tensor*);
tensor* pow_k(tensor*, int);
tensor* transpose_k(tensor*);
tensor* mul_k(tensor*, tensor*);
tensor* batched_transpose_k(tensor*);
tensor* batched_reduce_sum_k(tensor*);


// **** kernels *****
//   - operate on tensors
//   - allocate ouput buff
//      - return new tensor


//    binary elementwise

tensor* unsafe_add_k_(tensor* a, tensor* b, tensor* out) {
    if (a->size != b->size){
        printf("[unsafe_add_k_] expected size to match");
        exit(1);
    }
    for (int i=0; i<out->size; i++){
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

tensor* add_k_(tensor* a, tensor* b, tensor* out) {
    for (int i=0; i<out->size; i++)
        out->data[at(out, i)] = a->data[at(a, i)] + b->data[at(b, i)];
    return out;
}

tensor* add_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    return add_k_(a, b, out);
}


tensor* sub_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] - b->data[i];
    return out;
}


tensor* mul_k_(tensor* a, tensor* b, tensor* out) {
    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] * b->data[i];
    return out;
}

tensor* mul_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    return mul_k_(a, b, out);
}


//    binary


void matmul_k_(tensor* a, tensor* b, tensor* out)
{
    int N = a->shape[0], M = a->shape[1];
    int D = b->shape[1];

    // (N, M) @ (M, D) = (N, D)
    for (int n=0; n<N; n++){
        for (int d=0; d<D; d++){
            float sum = 0;
            for (int m=0; m<M; m++){
                sum += a->data[index_2d(a, n, m)] * b->data[index_2d(b, m, d)];
            }
            out->data[index_2d(out, n, d)] = sum;
        }
    }
}

tensor* matmul_k(tensor* a, tensor* b)
{
    int N = a->shape[0], D = b->shape[1];
    tensor* out = Tensor(N, D);
    matmul_k_(a, b, out);
    return out;
}


tensor* div_k(tensor* a, tensor* b) {

    if (a->num_dims!=2 || b->num_dims!=2){
        printf("[div_k] Error");
        exit(1);
    }

    tensor* out = TensorLike(a);

    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] / b->data[i];
    return out;
}


tensor* repeat_k(tensor* a, int num_repeats) {

    if (a->num_dims!=2 || a->shape[1]!=1){
        printf("[repeat] Error");
        exit(1);
    }

    int B = a->shape[0];
    tensor* out = Tensor(B, num_repeats);

    for (int b=0; b<B; b++){
        float* curr_a = a->data + b;
        float* curr_out = out->data + (b * out->stride[0]);
        for (int i=0; i<num_repeats; i++){
            *(curr_out+i) = *(curr_a);
        }
    }
    return out;
}


tensor* select_k(tensor* a, tensor* idx) {

    if (a->num_dims!=2 || idx->num_dims!=2 || idx->shape[1]!=1 || idx->shape[0]!=a->shape[0]) {
        printf("[select] Error shape");
        exit(1);
    }

    int B = a->shape[0];
    tensor* out = Tensor(B, 1);

    for (int b=0; b<B; b++){
        float* curr_a = a->data + (a->stride[0]*b);
        out->data[b] = *(curr_a + (int)idx->data[b]);
    }
    return out;
}


void select_set_(tensor* a, tensor* idx, int value) {
    if (a->num_dims!=2 || idx->num_dims!=2 || idx->shape[1]!=1 || idx->shape[0]!=a->shape[0]) {
        printf("[select_set_] Error shape");
        exit(1);
    }
    int B = a->shape[0];
    for (int b=0; b<B; b++){
        float* curr_a = a->data + (a->stride[0]*b);
        *(curr_a + (int)idx->data[b]) = value;
    }
}


//    unary


tensor* pow_k(tensor* a, int exponent) {
    tensor* out = TensorLikeFill(a, 0.0);
    for (int i=0; i<out->size; i++)
        // pow here refers to C's math function, not tiny-torch's op
        out->data[i] = (float)pow(a->data[i], exponent);
    return out;
}


tensor* reduce_sum_k(tensor* a) {
    // reduce to scalar
    tensor* out = TensorScalarFill(0.0);
    for (int i=0; i<a->size; i++) {
        out->data[0] += a->data[i];
    }
    return out;
}


tensor* relu_k(tensor* a) {
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++){
        if (a->data[i] < 0.0){
            out->data[i] = 0.0;
            // todo:
            // scratch_space->data[]
        } else {
            out->data[i] = a->data[i];
        }
    }
    // out->scratch_space[0] = scratch_space;
    return out;
}

void relu_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* local = TensorLike(a);
    for (int i=0; i<local->size; i++) {
        local->data[i] = a->data[i] > 0.0 ? 1.0 : 0.0;
    }

    a->grad = mul_k(local, upstream);
}


tensor* transpose_k(tensor* x) {
    // todo:
    // int shape_0 = x->shape[0];
    // x->shape[0] = x->shape[1];
    // x->shape[1] = shape_0;

    // int stride_0 = x->stride[0];
    // x->stride[0] = x->stride[1];
    // x->stride[1] = stride_0;
    // return x;


    int s1 = x->shape[0], s2 = x->shape[1];
    int stride_next_row = s2, stride_next_col = 1;
    tensor* out = Tensor(s2, s1);

    int idx_orig = 0;
    for (int s1_transposed=0; s1_transposed < s2; s1_transposed++){
        for (int s2_transposed=0; s2_transposed < s1; s2_transposed++){

            int idx_trans = (s1_transposed * stride_next_col) + (s2_transposed * stride_next_row);
            out->data[idx_orig++] = x->data[idx_trans];
        }
    }
    return out;
}



tensor* reduce_max_k(tensor* a) {
    tensor* out = TensorScalarFill(a->data[0]);
    out->scratch_space[0] = TensorLikeFill(out, 0.);
    for (int i=0; i<a->size; i++) {
        if (a->data[i] > out->data[0]){
            out->data[0] = a->data[i];
            out->scratch_space[0]->data[0] = i;
        }
    }
    return out;
}


tensor* neg_k(tensor* a) {
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++)
        out->data[i] = - a->data[i];
    return out;
}


tensor* exp_k(tensor* a) {
    if (a->num_dims!=2) {
        printf("[exp_k] Error shape");
        exit(1);
    }
    tensor* out = TensorLikeFill(a, 0.0);
    for (int i=0; i<out->size; i++){
        out->data[i] = expf(a->data[i]);
    }
    return out;
}


tensor* log_k(tensor* a) {
    if (a->num_dims!=2) {
        printf("[log_k] Error shape");
        exit(1);
    }
    tensor* out = TensorLikeFill(a, 0.0);
    for (int i=0; i<out->size; i++){
        out->data[i] = logf(a->data[i]);
    }
    return out;
}


//    batched


// a.shape (B, N, M)
// b.shape (B, M, D)
tensor* batched_matmul_k(tensor* a, tensor* b)
{
    int B = a->shape[0], N = a->shape[1], M = a->shape[2];
    int D = b->shape[2];

    tensor* out = Tensor(B, N, D);

    for (int i=0; i<B; i++){
        tensor* curr_out = TensorNoData(N, D);
        curr_out->data = out->data + (i * out->stride[0]);

        tensor* curr_a = TensorNoData(N, M);
        curr_a->data = a->data + (i * a->stride[0]);

        tensor* curr_b = TensorNoData(M, D);
        curr_b->data = b->data + (i * b->stride[0]);

        matmul_k_(curr_a, curr_b, curr_out);
    }
    return out;
}

tensor* batched_reduce_sum_k(tensor* a) {

    if (a->num_dims!=2){
        printf("[batched_reduce_sum] Error");
        exit(1);
    }

    int B = a->shape[0], N = a->shape[1];
    tensor* out = Tensor(B, 1);

    for (int b=0; b<B; b++){
        tensor* curr_a = TensorNoData(1, N);
        curr_a->data = a->data + (b * a->stride[0]);

        tensor* curr_out = reduce_sum_k(curr_a);
        out->data[b] = curr_out->data[0];
    }
    return out;
}


tensor* batched_reduce_max_k(tensor* a) {

    if (a->num_dims!=2){
        printf("[batched_max] Error");
        exit(1);
    }

    int B = a->shape[0], N = a->shape[1];
    tensor* out = Tensor(B, 1);
    out->scratch_space[0] = TensorLikeFill(out, 0.);

    for (int b=0; b<B; b++){
        tensor* curr_a = TensorNoData(1, N);
        curr_a->data = a->data + (b * a->stride[0]);

        tensor* curr_out = reduce_max_k(curr_a);
        out->data[b] = curr_out->data[0];
        out->scratch_space[0]->data[b] = curr_out->scratch_space[0]->data[0];
    }
    return out;
}

tensor* batched_transpose_k(tensor* x){
    int shape_1 = x->shape[1];
    x->shape[1] = x->shape[2];
    x->shape[2] = shape_1;

    int stride_1 = x->stride[1];
    x->stride[1] = x->stride[2];
    x->stride[2] = stride_1;
    return x;
}
