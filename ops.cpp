#include "nn.h"
#include "indexing.cpp"

#include <math.h> // log, pow


tensor* neg_k(tensor*);
tensor* pow_k(tensor*, int);
tensor* transpose_k(tensor*);
tensor* mul_k(tensor*, tensor*);
tensor* batched_transpose_k(tensor*);
tensor* batched_reduce_sum_k(tensor*);


void _strcp(char* src, char* dst, int size){
    for (int i=0; i<size; i++) {
        dst[i] = src[i];
        cout << src[i] << endl;
    }
}

void _copy_arr(float* src, float* dst, int size) {
    for (int i=0; i<size; i++)
        dst[i] = src[i];
}

// **** kernels *****
//   - operate on tensors
//   - allocate ouput buff
//      - return new tensor

// **** operations ****
//   - user-facing, unlike other abstractions
//   - call primitives, in addition, record data for the autograd:
//   - allocating grad buffers
//   - write local derivatives, for each input


//    binary elementwise


tensor* add_k_(tensor* a, tensor* b, tensor* out) {
    int (*at_fn_ptr)(tensor*, int) = NULL;
    if (a->num_dims==2) {
        at_fn_ptr = at_2d;
    } else if (a->num_dims==3) {
        at_fn_ptr = at_3d;
    } else {
        printf("[add_k_] Error");
        return NULL;
    }

    for (int i=0; i<out->size; i++)
        out->data[at_fn_ptr(out, i)] = a->data[at_fn_ptr(a, i)] + b->data[at_fn_ptr(b, i)];
    return out;
}

tensor* add_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    return add_k_(a, b, out);
}

void add_bwd(tensor* upstream, tensor* out) {
    // out is an ouput of the op, it's used to
    // retrieve pointers to inputs tensors
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local grad (note also allocates buff)
    tensor* a_local = TensorLikeFill(a, 1.0);
    tensor* b_local = TensorLikeFill(b, 1.0);

    // downstream
    a->grad = mul_k(a_local, upstream);
    b->grad = mul_k(b_local, upstream);

    // free(a_local), free(b_local);
}

tensor* add(tensor* a, tensor* b) {
    a->num_uses++;
    b->num_uses++;
    tensor* t = add_k(a, b);
    t->is_leaf = false;
    // fill the additional info on out tensor
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->op_type = 0;
    t->grad_fn = add_bwd;
    return t;
}


tensor* sub_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] - b->data[i];
    return out;
}

void sub_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    tensor* a_local = TensorLikeFill(a, 1.0);
    tensor* b_local = TensorLikeFill(b, -1.0);

    // downstream = local * upstream
    a->grad = mul_k(a_local, upstream);
    b->grad = mul_k(b_local, upstream);

    // free(a_local), free(b_local);
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


tensor* mul_k(tensor* a, tensor* b) {

    tensor* out = NULL;

    // todo-high: move this into TensorLike fn itself; rename current TensorLike to TensorLike2d
    if (a->num_dims==2)
        out = TensorLike(a);
    else if (a->num_dims==3)
        out = TensorLike3d(a);
    else if (a->num_dims==4)
        out = TensorLike4d(a);
    else {
        printf("[mul_k] Error");
        return NULL;
    }

    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] * b->data[i];
    return out;
}

void mul_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    tensor* a_local = b;
    tensor* b_local = a;

    // downstream
    a->grad = mul_k(a_local, upstream);
    b->grad = mul_k(b_local, upstream);
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


//    binary not-elementwise


// a.shape (N, M)
// b.shape (M, D)
// out.shape (N, D)
void matmul_k_(tensor* a, tensor* b, tensor* out)
{
    // todo: add assert that num dims in inputs == 2

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

// a.shape (N, M)
// b.shape (M, D)
// out.shape (N, D)
tensor* matmul_k(tensor* a, tensor* b)
{
    int N = a->shape[0], D = b->shape[1];
    tensor* out = Tensor(N, D);
    matmul_k_(a, b, out);
    return out;
}

void matmul_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // 1. local

    // upstream(N, D)   // same as t.shape
    //   a - ?
    //      a(N, M), so a_grad(N, M)
    //      upstream(N, D) @ b.t(D, M) = a_grad(N, M)
    //   b - ?
    //      b(M, D), so b_grad(M, D)
    //      a.t(M, N) @ upstream(N, D) = b_grad(M, D)

    tensor* local_a = transpose_k(b); // (M, D) -> (D, M)
    tensor* local_b = transpose_k(a); // (N, M) -> (M, N)

    // 2. wire local with upstream
    // upstream(N, D) @ b.t(D, M) = a_grad(N, M)
    a->grad = matmul_k(upstream, local_a);
    // a.t(M, N) @ upstream(N, D) = b_grad(M, D)
    b->grad = matmul_k(local_b, upstream);

    // free(local_a), free(local_b);
}

tensor* matmul(tensor* a, tensor* b){
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



tensor* div_k(tensor* a, tensor* b) {

    if (a->num_dims!=2 || b->num_dims!=2){
        printf("[div_k] Error");
        return NULL;
    }

    tensor* out = TensorLike(a);

    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] / b->data[i];
    return out;
}

void div_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    tensor* a_local = div_k(TensorLikeFill(a, 1.0), b);
    tensor* b_local = neg_k(div_k(a, pow_k(b, 2)));

    // downstream
    if (!a->grad)
        a->grad = TensorLikeFill(a, 0.0);

    if (!b->grad)
        b->grad = TensorLikeFill(b, 0.0);

    tensor* a_grad = mul_k(a_local, upstream);
    tensor* b_grad = mul_k(b_local, upstream);
    // does "+="
    add_k_(a->grad, a_grad, a->grad);
    add_k_(b->grad, b_grad, b->grad);

    // free(a_local), free(b_local);
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



tensor* repeat_k(tensor* a, int num_repeats) {

    if (a->num_dims!=2 || a->shape[1]!=1){
        printf("[repeat] Error");
        return NULL;
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

void repeat_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];

    if (a->num_dims!=2)
        printf("[repeat] Error");

    // sum together each row of upstream
    a->grad = batched_reduce_sum_k(upstream);

    // free(local);
}

tensor* repeat(tensor* a, int num_repeats) {
    a->num_uses++;
    tensor* t = repeat_k(a, num_repeats);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 18;
    t->grad_fn = repeat_bwd;
    return t;
}



tensor* select_k(tensor* a, tensor* idx) {
    if (a->num_dims!=2 || idx->num_dims!=2 || idx->shape[1]!=1 || idx->shape[0]!=a->shape[0]) {
        printf("[select] Error shape");
        return NULL;
    }

    int B = a->shape[0];
    tensor* out = Tensor(B, 1);

    for (int b=0; b<B; b++){
        float* curr_a = a->data + (a->stride[0]*b);
        out->data[b] = *(curr_a + (int)idx->data[b]);
    }
    return out;
}

// a:   (s1, s2)
// idx: (s1, 1)
// out: (s1, 1)
void select_bwd(tensor* upstream, tensor* out) {

    tensor* a=out->inputs[0];
    tensor* idx=out->inputs[1];
    int B = a->shape[0];

    idx->grad = TensorLikeFill(idx, 1.0);  // (s1, 1)
    for (int i=0; i<idx->grad->size; i++)
        idx->grad->data[i] = idx->grad->data[i] * upstream->data[i];

    a->grad = TensorLikeFill(a, 0.0);      // (s1, s2)
    for (int b=0; b<B; b++) {
        int offset_batch = b * a->grad->stride[0];
        float* curr_a_grad = a->grad->data + offset_batch;
        // 1.0 is local
        *(curr_a_grad + (int)idx->data[b]) = 1.0 * upstream->data[b];
    }
}

tensor* select(tensor* a, tensor* idx) {
    a->num_uses++;
    idx->num_uses++;
    tensor* t = select_k(a, idx);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = idx;
    t->op_type = 14;
    t->grad_fn = select_bwd;
    return t;
}



//    unary


tensor* pow_k(tensor* a, int exponent) {
    tensor* out = TensorLikeFill(a, 0.0);
    for (int i=0; i<out->size; i++)
        // pow here refers to C's math function, not tiny-torch's op
        out->data[i] = (float)pow(a->data[i], exponent);
    return out;
}

void pow_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    tensor* local = mul_k(TensorLikeFill(a, 2.0), a);
    // 2. wire local with upstream
    a->grad = mul_k(local, upstream);
    // free(local);
}

tensor* pow(tensor* a, int exponent) {
    a->num_uses++;
    tensor* t = pow_k(a, exponent);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 4;
    t->grad_fn = pow_bwd;
    return t;
}


tensor* reduce_sum_k(tensor* a) {
    // reduce to scalar
    tensor* out = TensorScalarFill(0.0);
    for (int i=0; i<a->size; i++) {
        out->data[0] += a->data[i];
    }
    return out;
}

void reduce_sum_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    tensor* local = TensorLikeFill(a, 1.0);
    // 2. wire local with upstream
    // make upstream and local to be same shape (currently upstream is a scalar, while local is a 2d tensor)
    tensor* broadcasted_upstream = TensorLikeFill(a, upstream->data[0]);
    a->grad = mul_k(local, broadcasted_upstream);
    // free(local);
}

tensor* reduce_sum(tensor* a) {
    a->num_uses++;
    tensor* t = reduce_sum_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 5;
    t->grad_fn = reduce_sum_bwd;
    return t;
}



tensor* relu_k(tensor* a) {

    tensor* out = NULL;

    if (a->num_dims==2)
        out = TensorLike(a);
    else if (a->num_dims==3)
        out = TensorLike3d(a);
    else if (a->num_dims==4)
        out = TensorLike4d(a);
    else {
        printf("[relu] Error");
        return NULL;
    }

    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
    return out;
}

void relu_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];

    tensor* local = NULL;

    if (a->num_dims==2)
        local = TensorLike(a);
    else if (a->num_dims==3)
        local = TensorLike3d(a);
    else if (a->num_dims==4)
        local = TensorLike4d(a);
    else
        printf("[relu] Error");

    for (int i=0; i<local->size; i++) {
        local->data[i] = a->data[i] > 0.0 ? 1.0 : 0.0;
    }

    a->grad = mul_k(local, upstream);
    // free(local);
}

tensor* relu(tensor* a) {
    a->num_uses++;
    tensor* t = relu_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 6;
    t->grad_fn = relu_bwd;
    return t;
}



// todo-high:
// By modifying strides, for example, an array can be transposed
// or reshaped at zero cost (no memory needs to be copied).
tensor* transpose_k(tensor* x) {
    // int shape_0 = x->shape[0];
    // x->shape[0] = x->shape[1];
    // x->shape[1] = shape_0;

    // int stride_0 = x->stride[0];
    // x->stride[0] = x->stride[1];
    // x->stride[1] = stride_0;
    // return x;


    int s1 = x->shape[0], s2 = x->shape[1];
    // [S1, S2] -> [S2, S1]
    //   - to go to next row, need to skip S2 elements
    //   - to go to next column, need to skip 1
    int stride_next_row = s2, stride_next_col = 1;

    tensor* out = Tensor(s2, s1);

    // basically need indexing pattern to access elements of array
    // next idx is different inside-a-row vs when switching to next row
    //    when within a row -- next idx is "curr_idx + stride_col"
    //    when switching to new row -- "first idx (of curr row) + stride_row"

    int idx_orig = 0;

    // these two loops are only needed to continently provide -- ranges over s2, s1
    for (int s1_transposed=0; s1_transposed < s2; s1_transposed++){
        for (int s2_transposed=0; s2_transposed < s1; s2_transposed++){

            int idx_trans = (s1_transposed * stride_next_col) + (s2_transposed * stride_next_row);
            // cout << "idx_orig: " << idx_orig
            out->data[idx_orig++] = x->data[idx_trans];
            // cout << "; idx_trans: " << idx_trans << endl;
        }
    }
    return out;
}

// todo: bwd formula
void transpose_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    a->grad = transpose_k(upstream);
}

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


tensor* max_k(tensor* a) {
    // set inital minimum to the first element
    tensor* out = TensorScalarFill(a->data[0]);
    for (int i=0; i<a->size; i++) {
        out->data[0] = (a->data[i] > out->data[0]) ? a->data[i] : out->data[0];
    }
    return out;
}

void max_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    tensor* local = TensorLikeFill(a, 0.0);
    int idx_max = 0;
    float max = a->data[idx_max];
    for (int i=0; i<a->size; i++) {
        if (a->data[i] > max){
            max = a->data[i];
            idx_max = i;
        }
    }
    local->data[idx_max] = 1.0;
    // 2. wire local with upstream
    tensor* broadcasted_upstream = TensorLikeFill(a, upstream->data[0]);
    a->grad = mul_k(local, broadcasted_upstream);
    // free(local);
}

tensor* max(tensor* a) {
    a->num_uses++;
    tensor* t = max_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 21;
    t->grad_fn = max_bwd;
    return t;
}



tensor* neg_k(tensor* a) {

    tensor* out = NULL;

    if (a->num_dims==2)
        out = TensorLike(a);
    else if (a->num_dims==3)
        out = TensorLike3d(a);
    else if (a->num_dims==4)
        out = TensorLike4d(a);
    else {
        printf("[neg] Error");
        return NULL;
    }

    for (int i=0; i<out->size; i++)
        out->data[i] = - a->data[i];
    return out;
}

void neg_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* local = NULL;

    if (a->num_dims==2)
        local = TensorLikeFill(a, -1.0);
    else if (a->num_dims==3)
        local = TensorLikeFill3d(a, -1.0);
    else if (a->num_dims==4)
        local = TensorLikeFill4d(a, -1.0);
    else
        printf("[neg] Error");

    a->grad = mul_k(local, upstream);
}

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



tensor* exp_k(tensor* a) {
    if (a->num_dims!=2) {
        printf("[exp_k] Error shape");
        return NULL;
    }
    tensor* out = TensorLikeFill(a, 0.0);
    for (int i=0; i<out->size; i++){
        out->data[i] = expf(a->data[i]);
    }
    return out;
}

void exp_bwd(tensor* upstream, tensor* out) {
    tensor* a=out->inputs[0];
    if (!a->grad)
        a->grad = TensorLikeFill(a, 0.0);

    for (int i=0; i<a->grad->size; i++) {
        float local = expf(a->data[i]);
        a->grad->data[i] = local * upstream->data[i];
    }
}


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



tensor* log_k(tensor* a) {
    if (a->num_dims!=2) {
        printf("[log_k] Error shape");
        return NULL;
    }
    tensor* out = TensorLikeFill(a, 0.0);
    for (int i=0; i<out->size; i++){
        out->data[i] = logf(a->data[i]);
    }
    return out;
}

void log_bwd(tensor* upstream, tensor* out) {
    tensor* a=out->inputs[0];
    a->grad = TensorLikeFill(a, 0.0);

    float log_e = logf(M_E);

    for (int i=0; i<a->grad->size; i++) {
        float x = a->data[i];
        float local = 1/x * log_e;
        a->grad->data[i] = local * upstream->data[i];
    }
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



//    batched



// a.shape (B, N, M)
// b.shape (B, M, D)
tensor* batched_matmul_k(tensor* a, tensor* b)
{
    int B = a->shape[0], N = a->shape[1], M = a->shape[2];
    int D = b->shape[2];

    tensor* out = Tensor3d(B, N, D);

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

void batched_matmul_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // upstream(B, N, D)   // same as t.shape
    //   a - ?
    //      a(B, N, M), so a_grad(B, N, M)
    //      upstream(B, N, D) @ b.t(B, D, M) = a_grad(B, N, M)
    //   b - ?
    //      b(B, M, D), so b_grad(B, M, D)
    //      a.t(B, M, N) @ upstream(B, N, D) = b_grad(B, M, D)

    tensor* local_a = batched_transpose_k(b); // (B, M, D) -> (B, D, M)
    tensor* local_b = batched_transpose_k(a); // (B, N, M) -> (B, M, N)

    // 2. wire local with upstream
    // upstream(B, N, D) @ b.t(B, D, M) = a_grad(B, N, M)
    a->grad = batched_matmul_k(upstream, local_a);
    // a.t(B, M, N) @ upstream(B, N, D) = b_grad(B, M, D)
    b->grad = batched_matmul_k(local_b, upstream);

    // free(local_a), free(local_b);
}

tensor* batched_matmul(tensor* a, tensor* b)
{
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



tensor* batched_flatten_k(tensor* a) {
    int B = a->shape[0], out_dim = -1;

    if (a->num_dims==3)
        out_dim = a->shape[1] * a->shape[2];
    else if (a->num_dims==4)
        out_dim = a->shape[1] * a->shape[2] * a->shape[3];
    else {
        printf("[batched_flatten] Error");
        return NULL;
    }

    tensor* out = Tensor(B, out_dim);

    for (int i=0; i<a->size; i++)
        out->data[i] = a->data[i];
    return out;
}

void batched_flatten_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];

    if (a->num_dims==3)
        a->grad = TensorLike3d(a);
    else if (a->num_dims==4)
        a->grad = TensorLike4d(a);
    else
        printf("[batched_flatten] Error");

    // reshape upstream into the shape of a
    for (int i=0; i<upstream->size; i++)
        a->grad->data[i] = upstream->data[i];

    // free(local);
}

tensor* batched_flatten(tensor* a) {
    a->num_uses++;
    tensor* t = batched_flatten_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 13;
    t->grad_fn = batched_flatten_bwd;
    return t;
}



tensor* batched_reduce_sum_k(tensor* a) {

    if (a->num_dims!=2){
        printf("[batched_reduce_sum] Error");
        return NULL;
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

void batched_reduce_sum_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    int B = a->shape[0], N = a->shape[1];

    if (a->num_dims!=2)
        printf("[batched_reduce_sum] Error");

    if (!a->grad)
        // important to fill with 0's if we gonna "+=" to it below.
        // If we instead simply overwrite it, then wouldn't matter,
        // but bc we do "+=" it does matter (if there's any garbage
        // data, the grad will be += to it)
        a->grad = TensorLikeFill(a, 0.0);

    for (int b=0; b<B; b++){
        tensor* curr_a = TensorNoData(1, N);
        curr_a->data = a->data + (b * a->stride[0]);    // grad will be set by reduce_sum_bwd

        tensor* curr_out = TensorNoData(1, 1);
        curr_out->data = out->data + (b * out->stride[0]);

        curr_out->inputs[0] = curr_a;

        tensor* curr_upstream = TensorNoData(1, 1);
        curr_upstream->data = upstream->data + (b * upstream->stride[0]);

        reduce_sum_bwd(curr_upstream, curr_out);

        for (int i=0; i<curr_a->grad->size; i++){
            int offset_batch = b * a->grad->stride[0];
            a->grad->data[offset_batch + i] = a->grad->data[offset_batch + i] + curr_a->grad->data[i];
        }
    }

    // free(local);
}

tensor* batched_reduce_sum(tensor* a) {
    a->num_uses++;
    tensor* t = batched_reduce_sum_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 17;
    t->grad_fn = batched_reduce_sum_bwd;
    return t;
}



tensor* batched_max_k(tensor* a) {

    if (a->num_dims!=2){
        printf("[batched_max] Error");
        return NULL;
    }

    int B = a->shape[0], N = a->shape[1];
    tensor* out = Tensor(B, 1);

    for (int b=0; b<B; b++){
        tensor* curr_a = TensorNoData(1, N);
        curr_a->data = a->data + (b * a->stride[0]);

        tensor* curr_out = max_k(curr_a);
        out->data[b] = curr_out->data[0];
    }
    return out;
}

void batched_max_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    int B = a->shape[0], N = a->shape[1];

    if (a->num_dims!=2)
        printf("[batched_max] Error");

    if (!a->grad)
        a->grad = TensorLikeFill(a, 0.0);

    for (int b=0; b<B; b++){
        tensor* curr_a = TensorNoData(1, N);
        curr_a->data = a->data + (b * a->stride[0]);    // grad will be set by max_bwd

        tensor* curr_out = TensorNoData(1, 1);
        curr_out->data = out->data + (b * out->stride[0]);

        curr_out->inputs[0] = curr_a;

        tensor* curr_upstream = TensorNoData(1, 1);
        curr_upstream->data = upstream->data + (b * upstream->stride[0]);

        max_bwd(curr_upstream, curr_out);

        for (int i=0; i<curr_a->grad->size; i++){
            int offset_batch = b * a->grad->stride[0];
            a->grad->data[offset_batch + i] = a->grad->data[offset_batch + i] + curr_a->grad->data[i];
        }
    }

    // free(local);
}

tensor* batched_max(tensor* a) {
    a->num_uses++;
    tensor* t = batched_max_k(a);
    t->is_leaf = false;
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->op_type = 22;
    t->grad_fn = batched_max_bwd;
    return t;
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
