#include "nn.h"

//@@@@@@ testing backdrop engine @@@@@@



// **** kernels *****
//   - operate on floats
//   - have output buffer provided as input
//      - return void

// note: naming convention: assume "_elementwise", by default -- and ask to specify otherwise (if not elementwise) -- e.g. reduce_, matrix_, etc

void strcp(char* src, char* dst, int size){
    for (int i=0; i<size; i++) {
        dst[i] = src[i];
        cout << src[i] << endl;
    }
}

void copy_arr(float* src, float* dst, int size) {
    for (int i=0; i<size; i++)
        dst[i] = src[i];
}

float _pow(float x, int exp) {
    for (int i=0; i<exp; i++)
        x *= x;
    return x;
}

//    binary elementwise

void add_k(float* a, float* b, float* out, int size){
    for (int i=0; i<size; i++)
        out[i] = a[i] + b[i];
}

void sub_k(float* a, float* b, float* out, int size){
    for (int i=0; i<size; i++)
        out[i] = a[i] - b[i];
}

void mul_k(float* a, float* b, float* out, int size){
    for (int i=0; i<size; i++)
        out[i] = a[i] * b[i];
}

// binary

void matmul_k(float* a, float* b, float* out, int N, int M, int D){
    // (N, M) @ (M, D) = (N, D)
    for (int n=0; n<N; n++){
        for (int d=0; d<D; d++){
            float sum = 0;
            for (int m=0; m<M; m++){
                // n*M, because for each "n" you skip "M" values
                sum += a[n*M + m] * b[m*D + d];
            }
            out[n*D + d] = sum;
        }
    }
}

//    unary elementwise

void pow_k(float* a, int b, float* out, int size){
    for (int i=0; i<size; i++)
        out[i] = _pow(a[i], b);
}

void reduce_sum_k(float* a, float* out, int size) {
    for (int i=0; i<size; i++)
        *out += a[i];
}



// **** primitives *****
//   - operate on tensors (call into kernels)
//   - allocate ouput buff
//      - return new tensor

//    binary elementwise

tensor* add_p(tensor* a, tensor* b) {
    tensor* t = TensorLike(a);
    add_k(a->data, b->data, t->data, a->size);
    return t;
}

tensor* mul_p(tensor* a, tensor* b) {
    tensor* t = TensorLike(a);
    mul_k(a->data, b->data, t->data, a->size);
    return t;
}

tensor* sub_p(tensor* a, tensor* b) {
    tensor* t = TensorLike(a);
    sub_k(a->data, b->data, t->data, a->size);
    return t;
}

//    binary not-elementwise

tensor* matmul_p(tensor* a, tensor* b)
{
    int N = a->shape[0], M = a->shape[1];
    int D = b->shape[1];
    tensor* out = Tensor(N, D);
    matmul_k(a->data, b->data, out->data, N, M, D);
    return out;
}

//    unary

tensor* pow_p(tensor* a, int exponent) {
    tensor* t = TensorLikeFill(a, 0.0);
    pow_k(a->data, exponent, t->data, t->size);
    return t;
}

tensor* reduce_sum_p(tensor* a) {
    // reduce to scalar
    tensor* t = TensorScalarFill(0.0);
    reduce_sum_k(a->data, t->data, a->size);
    return t;
}



// **** operations ****
//   - user-facing, unlike other abstractions
//   - call primitives, in addition, record data for the autograd:
//   - allocating grad buffers
//   - write local derivatives, for each input

//    binary elementwise

void add_bwd(float* upstream, tensor* out) {
    // out is an ouput of the op, it's used to
    // retrieve pointers to inputs tensors
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // store local grad in the grad field
    // (note also allocates buff)
    a->grad = FloatLikeFill(a, 1.0);
    b->grad = FloatLikeFill(b, 1.0);

    // downstream = local * upstream

    // note also, already does +=
    //   inp->grad = (inp->grad) * (upstream);
    mul_k(a->grad, upstream, a->grad, a->size);
    mul_k(b->grad, upstream, b->grad, b->size);

}

tensor* add(tensor* a, tensor* b) {
    tensor* t = add_p(a, b);
    t->is_leaf = false;
    // todo-low: check bool tensor->requires_grad before doing steps below, including allocating buffers

    // todo: this can be further abstracted -- creating a binary_op function
    // todo: and even further abstracted -- creating a binary_elementwise_op function

    // fill the additional info on out tensor
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;

    char op_name[] = "add";
    strcp(t->op_name, op_name, strlen(op_name));

    t->grad_fn = add_bwd;
    return t;
}


void sub_bwd(float* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    a->grad = FloatLikeFill(a, 1.0);
    b->grad = FloatLikeFill(b, -1.0);

    // downstream = local * upstream
    mul_k(a->grad, upstream, a->grad, a->size);
    mul_k(b->grad, upstream, b->grad, b->size);
}

tensor* sub(tensor* a, tensor* b) {
    tensor* t = sub_p(a, b);
    t->is_leaf = false;

    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;

    t->grad_fn = sub_bwd;
    return t;
}


void mul_bwd(float* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // todo: don't need malloc here? Bc a->data, b->data
    //  already malloc'ed -- can just save a pointer to them?

    // local
    //   1. alloc memory
    a->grad = EmptyFloatLike(a);
    b->grad = EmptyFloatLike(b);
    //   2. copy over
    copy_arr(b->data, a->grad, a->size);
    copy_arr(a->data, b->grad, b->size);

    // downstream = local * upstream
    mul_k(a->grad, upstream, a->grad, a->size);
    mul_k(b->grad, upstream, b->grad, b->size);
}

tensor* mul(tensor* a, tensor* b) {
    tensor* t = mul_p(a, b);
    t->is_leaf = false;

    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;

    t->grad_fn = mul_bwd;
    return t;
}


//    binary not-elementwise

void transpose_k(float* x, float* out, int s1, int s2);

void matmul_bwd(float* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];
    int N = a->shape[0], M = a->shape[1];
    int D = b->shape[1];

    // 1. local

    // Even though you gonna store e.g. a transpose in a->grad (and not a itself)
    //  it does not make sense to allocate like below because the dims are logical but
    //  mem is contiguous anyway -- so for both of the constructors below obviously same memory will be allocated
    //    EmptyFloat(a->shape[1], a->shape[0]);   // reversed shapes to store Transpose
    //    EmptyFloatLike(a);

    // note: should allocate a_buff not with a.size but w b.size
    // note: using this intermidiate variable instead of directly
    //  allocating "a->grad = EmptyFloatLike(b)", because the final
    //  a->grad is of shape a not of shape b. It would be incorrect
    //  to "a->grad = EmptyFloatLike(b)". So I keep these temporary
    //  variables (local_a, local_b) and de-allocate them after
    //  computing actual a->grad
    float* local_a = EmptyFloatLike(b);
    float* local_b = EmptyFloatLike(a);

    // upstream(M, D)   // same as t.shape
    //   a - ?
    //      a(M, N), so a_grad(M, N)
    //      upstream(M, D) @ b.t(D, N) = a_grad(M, N)
    //   b - ?
    //      b(N, D), so b_grad(N, D)
    //      a.t(N, M) @ upstream(M, D) = b_grad(N, D)

    // signature: transpose_k(float* x, float* out, int s1, int s2)
    transpose_k(b->data, local_a, N, D); // (N, D) -> (D, N)
    transpose_k(a->data, local_b, M, N); // (M, N) -> (N, M)

    // 2. wire local with upstream
    a->grad = EmptyFloatLike(a);
    b->grad = EmptyFloatLike(b);
    // upstream(M, D) @ b.t(D, N) = a_grad(M, N)
    matmul_k(upstream, local_a, a->grad, M, D, N);
    // a.t(N, M) @ upstream(M, D) = b_grad(N, D)
    matmul_k(local_b, upstream, b->grad, N, M, D);

    // note:
    free(local_a), free(local_b);
}

tensor* matmul(tensor* a, tensor* b){
    // e.g.: a(M, N) @ b(N, D) = t(M, D)
    tensor* t = matmul_p(a, b);
    t->is_leaf = false;

    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;

    t->grad_fn = matmul_bwd;
    return t;
}


//  unary

void pow_bwd(float* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    // store local in grad in the grad field
    a->grad = FloatLikeFill(a, 2.0);
    // todo-now: maybe to solve problems of needing kernles for floats, just make grad be a tensor itself (not float)?
    //    bc "ElementwiseMul(TensorLikeFill(a, 2.0), a);" is much better than below
    mul_k(a->grad, a->data, a->grad, a->size);
    // 2. wire local with upstream
    mul_k(a->grad, upstream, a->grad, a->size);
}

tensor* pow(tensor* a, int exponent) {
    tensor* t = pow_p(a, exponent);
    t->is_leaf = false;

    //  comment: note by "inputs" I mean tensor inputs (INPUTS which I'll use compute grads wrt to)
    //  so here even if this op has two inputs, it really has one, for the purpose of the autograd
    t->num_inputs = 1;
    t->inputs[0] = a;

    t->grad_fn = pow_bwd;
    return t;
}


void reduce_sum_bwd(float* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    a->grad = FloatLikeFill(a, 1.0);
    // 2. wire local with upstream
    mul_k(a->grad, upstream, a->grad, a->size);
}

tensor* reduce_sum(tensor* a) {
    tensor* t = reduce_sum_p(a);
    t->is_leaf = false;
    // fill the additional info on out tensor
    t->num_inputs = 1;
    t->inputs[0] = a;
    char op_name[] = "reduce_sum";  // {a,b,c,\0}
    // use strlen given arr is \0 terminated
    strcp(t->op_name, op_name, strlen(op_name));

    t->grad_fn = reduce_sum_bwd;
    return t;
}






//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

tensor* ElementwiseMul(tensor* a, tensor* b)
{
    // todo: here and in all ops, assert same size
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return out;
}


// todo: add tests -- https://github.com/tensorflow/tensorflow/commit/6f4a0e96d853d1d8fe05a8dd8f7ba0cd0fb0e79b#diff-65511a88d2951377144d77a2de94c0f597c4664189d3d5ac730e653560b64f31R259-R342
tensor* Matmul(tensor* x, tensor* w)
{
    int N = x->shape[0], M = x->shape[1];
    int D = w->shape[1];
    // (N, M) @ (M, D) = (N, D)
    // todo: add a way to create empty tensor -- bc currently my Tensor constructor also fills it tens w random values
    tensor* out = Tensor(N, D);

    for (int n=0; n<N; n++) {
        for (int d=0; d<D; d++) {
            float sum = 0;
            for (int m=0; m<M; m++){
                // cout << "x_idx: " << n*M+m << endl;
                // cout << "y_idx: " << m*D+d << endl;
                sum += x->data[n*M+m] * w->data[m*D+d];
                // cout << endl;
            }
            // cout << "sum: " << sum << endl;
            out->data[n*D+d] = sum;
        // cout << endl;
        }
    }

    // for (int n=0; n<N; n++) {
    //     for (int d=0; d<D; d++) {
    //         float sum = 0;
    //         for (int m=0; m<M; m++){
    //             // cout << "x_idx: " << index(n, m, n) << endl;
    //             // cout << "y_idx: " << index(m, d, m) << endl;
    //             sum += x[index(n, m, M)] * w[index(m, d, D)];
    //         }
    //         out[index(n, d, D)] = sum;
    //     }
    // }

    return out;
}

// todo: add tests
/*
    out2[1] = 0.123;
    print(ReluBackward(out2, N*D), N, D);
*/
tensor* Relu(tensor* x)
{
    tensor* out = TensorLike(x);
    for (int i=0; i<out->size; i++) {
        out->data[i] = x->data[i] > 0.0 ? x->data[i] : 0.0;
        // mask[i] = x[i] > 0.0 ? 1.0 : 0.0;
    }
    return out;
}

tensor* GetReluMask(tensor* x)
{
    tensor* out = TensorLike(x);
    for (int i=0; i<out->size; i++) {
        out->data[i] = x->data[i] > 0.0 ? 1.0 : 0.0;
    }
    return out;
}


/* pow tests
    cout << "pow(2, 2): " << pow(2, 2) << endl;
    cout << "pow(2, 2): " << pow(4, 8) << endl;
*/
float pow(float x, int exp)
{
    for (int i=0; i<exp; i++)
        x *= x;
    return x;
}

// note: use capital letters for functions that allocate heap mem, use lowercase otherwise
float mse(tensor* x, tensor* y)
{
    float diff, out = 0;
    for (int i=0; i<x->size; i++) {
        diff = y->data[i] - x->data[i];
        out += pow(diff, 2);
    }
    return out;
}


/* Transpose tests
    float arr[] = {0., 1., 2., 3.};
    cout << "\narr: ";
    Print(arr, 2, 2);
    float* arr_T = Transpose(arr, 2, 2);
    cout << "\ntransposed: ";
    Print(arr_T, 2, 2);

    float arr[] = {0., 1., 2.,
                    3., 4., 5.};
    cout << "\narr: ";
    Print(arr, 2, 3);
    float* arr_T = Transpose(arr, 2, 3);
    cout << "\ntransposed: ";
    Print(arr_T, 3, 2);
*/

// todo-low: use "const int" for s1, s2?
// [S1, S2] -> [S2, S1]
tensor* Transpose(tensor* x)
{
    int s1 = x->shape[0], s2 = x->shape[1];
    // [S1, S2]
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

// note: s1, s2 -- are before Transpose
// todo: re-implement form scratch (this and its Op, and its Primine)
void transpose_k(float* x, float* out, int s1, int s2)
{
    int stride_next_row = s2, stride_next_col = 1;
    int idx_orig = 0;

    for (int s1_transposed=0; s1_transposed < s2; s1_transposed++){
        for (int s2_transposed=0; s2_transposed < s1; s2_transposed++){

            int idx_trans = (s1_transposed * stride_next_col) + (s2_transposed * stride_next_row);
            out[idx_orig++] = x[idx_trans];
        }
    }
}

