/*
implementations of these are exactly the same for CPU and CUDA -- reuse

Temporarily commented out ops which don't yet have CUDA impl, so that compilation
doesn't fail bc linker can't find _k expected by these ops
*/


void add_bwd(tensor* upstream, tensor* out) {
    // out is an ouput of the op, it's used to
    // retrieve pointers to inputs tensors
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local grad (note also allocates buff)
    tensor* a_local = TensorLikeFill(a, 1.0);
    tensor* b_local = TensorLikeFill(b, 1.0);

    // downstream = local * upstream

    // todo-now: do += instead of replacing existing grad
    a->grad = mul_k(a_local, upstream);
    b->grad = mul_k(b_local, upstream);

    // todo-now: is it a correct way to de-alloc a struct?
    //  - and all other ops below
    // free(a_local), free(b_local);
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

    // note:
    // free(local_a), free(local_b);
}

void div_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    tensor* a_local = div_k(TensorLikeFill(a, 1.0), b);
    tensor* b_local = neg_k(div_k(a, pow_k(b, 2)));

    // downstream

    // comment: 
    // "a->grad = mul_k(a_local, upstream)" overwrite's input grad, the below does "+=" to it
    if (!a->grad)
        a->grad = TensorLikeFill(a, 0.0);
    else {
        printf("[div_bwd] a->grad exists!\n");
    }

    if (!b->grad)
        b->grad = TensorLikeFill(b, 0.0);
    else {
        printf("[div_bwd] b->grad exists!\n");
    }

    tensor* a_grad = mul_k(a_local, upstream);
    tensor* b_grad = mul_k(b_local, upstream);
    // does "+="
    add_k_(a->grad, a_grad, a->grad);
    add_k_(b->grad, b_grad, b->grad);

    // free(a_local), free(b_local);
}

// void repeat_bwd(tensor* upstream, tensor* out) {
//     tensor* a = out->inputs[0];

//     if (a->num_dims!=2){
//         printf("[repeat] Error");
//         exit(1);
//     }

//     // sum together each row of upstream
//     a->grad = batched_reduce_sum_k(upstream);

//     // free(local);
// }

void pow_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    // store local in grad in the grad field
    // todo-low: mem leak
    // todo: below assumes exponent=2 ?
    tensor* local = mul_k(TensorLikeFill(a, 2.0), a);
    // 2. wire local with upstream
    a->grad = mul_k(local, upstream);
    // free(local);
}

void exp_bwd(tensor* upstream, tensor* out) {
    tensor* a=out->inputs[0];
    if (!a->grad)
        a->grad = TensorLikeFill(a, 0.0);
    else
        printf("[exp_bwd] a->grad exists!\n");

    tensor* local = out;
    mul_k_(local, upstream, a->grad);
}

void log_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    a->grad = TensorLikeFill(a, 0.0);

    // approximately 2.718282 C Math exp() Function e is the base of the natural system of logarithms (approximately 2.718282)
    // Some implementations of the <math. h> library include a constant M_E
    float log_e = logf(M_E);
    // non-vectorized form: "(1/a) * log_e;"
    tensor* local = mul_k(div_k(TensorLikeFill(a, 1), a), TensorLikeFill(a, log_e));
    mul_k_(local, upstream, a->grad);
}

// void reduce_sum_bwd(tensor* upstream, tensor* out) {
//     tensor* a = out->inputs[0];
//     // 1. local
//     tensor* local = TensorLikeFill(a, 1.0);
//     // 2. wire local with upstream
//     // make upstream and local to be same shape (currently upstream is a scalar, while local is a 2d tensor)
//     tensor* broadcasted_upstream = TensorLikeFill(a, upstream->data[0]);
//     a->grad = mul_k(local, broadcasted_upstream);
//     // free(local);
// }

// todo-high: it doesn't make sense to have a transpose_bwd bc it just calls transpose -- at the moment this fn is used bc calling convention (args signature) for _bwd funcs is different from the fwd funcs
// todo: bwd formula
void transpose_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    a->grad = transpose_k(upstream);
}

void neg_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* local = NULL;

    local = TensorLikeFill(a, -1.0);
    a->grad = mul_k(local, upstream);
}

// // comment: shapes are same as matmul_bwd, but with additional (B,) dim first
// void batched_matmul_bwd(tensor* upstream, tensor* out) {
//     tensor* a = out->inputs[0];
//     tensor* b = out->inputs[1];

//     // upstream(B, N, D)   // same as t.shape
//     //   a - ?
//     //      a(B, N, M), so a_grad(B, N, M)
//     //      upstream(B, N, D) @ b.t(B, D, M) = a_grad(B, N, M)
//     //   b - ?
//     //      b(B, M, D), so b_grad(B, M, D)
//     //      a.t(B, M, N) @ upstream(B, N, D) = b_grad(B, M, D)

//     tensor* local_a = batched_transpose_k(b); // (B, M, D) -> (B, D, M)
//     tensor* local_b = batched_transpose_k(a); // (B, N, M) -> (B, M, N)

//     // 2. wire local with upstream
//     // upstream(B, N, D) @ b.t(B, D, M) = a_grad(B, N, M)
//     a->grad = batched_matmul_k(upstream, local_a);
//     // a.t(B, M, N) @ upstream(B, N, D) = b_grad(B, M, D)
//     b->grad = batched_matmul_k(local_b, upstream);

//     // note:
//     // free(local_a), free(local_b);
// }
