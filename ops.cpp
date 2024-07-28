#include "nn.h"

//@@@@@@ testing backdrop engine @@@@@@


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

float _pow(float x, int exp) {
    for (int i=0; i<exp; i++)
        x *= x;
    return x;
}

// note: naming convention: assume "_elementwise", by default -- and ask to specify otherwise (if not elementwise) -- e.g. reduce_, matrix_, etc

// **** kernels *****
//   - operate on tensors
//   - allocate ouput buff
//      - return new tensor

//    binary elementwise

// todo: use "cost tensor*" instead of "tensor"
tensor* add_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] + b->data[i];
    return out;
}

tensor* sub_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] - b->data[i];
    return out;
}

tensor* mul_k(tensor* a, tensor* b) {
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] * b->data[i];
    return out;
}



//    binary not-elementwise

tensor* matmul_k(tensor* a, tensor* b)
{
    int N = a->shape[0], M = a->shape[1];
    int D = b->shape[1];
    tensor* out = Tensor(N, D);

    // (N, M) @ (M, D) = (N, D)
    for (int n=0; n<N; n++){
        for (int d=0; d<D; d++){
            float sum = 0;
            for (int m=0; m<M; m++){
                // n*M, because for each "n" you skip "M" values
                sum += a->data[n*M + m] * b->data[m*D + d];
            }
            out->data[n*D + d] = sum;
        }
    }
    return out;
}

//    unary

tensor* pow_k(tensor* a, int exponent) {
    tensor* out = TensorLikeFill(a, 0.0);
    for (int i=0; i<out->size; i++)
        out->data[i] = _pow(a->data[i], exponent);
    return out;
}

tensor* reduce_sum_k(tensor* a) {
    // reduce to scalar
    tensor* out = TensorScalarFill(0.0);
    for (int i=0; i<a->size; i++)
        out->data[0] += a->data[i];
    return out;
}



// **** operations ****
//   - user-facing, unlike other abstractions
//   - call primitives, in addition, record data for the autograd:
//   - allocating grad buffers
//   - write local derivatives, for each input

//    binary elementwise

void add_bwd(tensor* upstream, tensor* out) {
    // out is an ouput of the op, it's used to
    // retrieve pointers to inputs tensors
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local grad (note also allocates buff)
    tensor* a_local = TensorLikeFill(a, 1.0);
    tensor* b_local = TensorLikeFill(b, 1.0);

    // downstream = local * upstream

    // note also, already does +=
    //   inp->grad = (inp->grad) * (upstream);
    a->grad = mul_k(a_local, upstream);
    b->grad = mul_k(b_local, upstream);

    // todo-now: is it a correct way to de-alloc a struct?
    //  - and all other ops below
    // free(a_local), free(b_local);
}

tensor* add(tensor* a, tensor* b) {
    tensor* t = add_k(a, b);
    t->is_leaf = false;
    // todo-low: check bool tensor->requires_grad before doing steps below, including allocating buffers

    // todo: this can be further abstracted -- creating a binary_op function
    // todo: and even further abstracted -- creating a binary_elementwise_op function

    // fill the additional info on out tensor
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;

    char op_name[] = "add";
    _strcp(t->op_name, op_name, strlen(op_name));

    t->grad_fn = add_bwd;
    return t;
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
    tensor* t = sub_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->grad_fn = sub_bwd;
    return t;
}


void mul_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // todo: don't need malloc here? Bc a->data, b->data
    //  already malloc'ed -- can just save a pointer to them?

    // note: no need to alloc buff for intermidiates, bc mul_k
    //  does not mutate its inputs

    // local
    tensor* a_local = b;
    tensor* b_local = a;

    // downstream = local * upstream
    a->grad = mul_k(a_local, upstream);
    b->grad = mul_k(b_local, upstream);
}

tensor* mul(tensor* a, tensor* b) {
    tensor* t = mul_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->grad_fn = mul_bwd;
    return t;
}


//    binary not-elementwise

tensor* Transpose(tensor* x);

void matmul_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

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

    // upstream(M, D)   // same as t.shape
    //   a - ?
    //      a(M, N), so a_grad(M, N)
    //      upstream(M, D) @ b.t(D, N) = a_grad(M, N)
    //   b - ?
    //      b(N, D), so b_grad(N, D)
    //      a.t(N, M) @ upstream(M, D) = b_grad(N, D)

    // todo-now: re-impl transpose as kernel and op, and replace below w transpose_k
    tensor* local_a = Transpose(b); // (N, D) -> (D, N)
    tensor* local_b = Transpose(a); // (M, N) -> (N, M)

    // 2. wire local with upstream
    // upstream(M, D) @ b.t(D, N) = a_grad(M, N)
    a->grad = matmul_k(upstream, local_a);
    // a.t(N, M) @ upstream(M, D) = b_grad(N, D)
    b->grad = matmul_k(local_b, upstream);

    // note:
    // free(local_a), free(local_b);
}

tensor* matmul(tensor* a, tensor* b){
    // e.g.: a(M, N) @ b(N, D) = t(M, D)
    tensor* t = matmul_k(a, b);
    t->is_leaf = false;
    t->num_inputs = 2;
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->grad_fn = matmul_bwd;
    return t;
}


//  unary

void pow_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    // store local in grad in the grad field
    // todo-low: mem leak
    tensor* local = mul_k(TensorLikeFill(a, 2.0), a);
    // 2. wire local with upstream
    a->grad = mul_k(local, upstream);
    // free(local);
}

tensor* pow(tensor* a, int exponent) {
    tensor* t = pow_k(a, exponent);
    t->is_leaf = false;
    //  comment: note by "inputs" I mean tensor inputs (INPUTS which I'll use compute grads wrt to)
    //  so here even if this op has two inputs, it really has one, for the purpose of the autograd
    t->num_inputs = 1;
    t->inputs[0] = a;
    t->grad_fn = pow_bwd;
    return t;
}


void reduce_sum_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    tensor* local = TensorLikeFill(a, 1.0);
    // 2. wire local with upstream
    a->grad = mul_k(local, upstream);
    // free(local);
}

tensor* reduce_sum(tensor* a) {
    tensor* t = reduce_sum_k(a);
    t->is_leaf = false;

    t->num_inputs = 1;
    t->inputs[0] = a;

    char op_name[] = "reduce_sum";  // {a,b,c,\0}
    // use strlen given arr is \0 terminated
    _strcp(t->op_name, op_name, strlen(op_name));

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
