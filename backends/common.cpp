void add_bwd(tensor* upstream, tensor* out) {
    // out is an ouput of the op, it's used to
    // retrieve pointers to inputs tensors
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local grad (note also allocates buff)
    tensor* a_local = TensorLikeFill(a, 1.0);
    tensor* b_local = TensorLikeFill(b, 1.0);

    // downstream = local * upstream

    a->grad = mul_k(a_local, upstream);
    b->grad = mul_k(b_local, upstream);
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

}

void div_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    tensor* a_local = div_k(TensorLikeFill(a, 1.0), b);
    tensor* b_local = neg_k(div_k(a, pow_k(b, 2)));

    // downstream
    maybe_init_grad(a);
    maybe_init_grad(b);

    tensor* a_grad = mul_k(a_local, upstream);
    tensor* b_grad = mul_k(b_local, upstream);
    // does "+="
    add_k_(a->grad, a_grad, a->grad);
    add_k_(b->grad, b_grad, b->grad);

    // free(a_local), free(b_local);
}

void repeat_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];

    if (a->num_dims!=2){
        printf("[repeat] Error");
        exit(1);
    }

    // sum together each row of upstream
    a->grad = batched_reduce_sum_k(upstream);

    // free(local);
}

// reduce_sum: (B, N) -> (B, 1)
void batched_reduce_sum_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0]; // (B, N)
    if (a->num_dims!=2){
        printf("[batched_reduce_sum] Error\n");
        exit(1);
    }

    maybe_init_grad(a);

    int N = a->shape[1];
    tensor* local = TensorLikeFill(a, 1.0); // (B, 1)
    tensor* upstream_broadcasted = repeat_k(upstream, N); // (B, 1) -> (B,N)
    tensor* a_grad = mul_k(local, upstream_broadcasted);

    // add to existing a grad
    add_k_(a->grad, a_grad, a->grad);
    // free(local);
}

void pow_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    // store local in grad in the grad field
    tensor* local = mul_k(TensorLikeFill(a, 2.0), a);
    // 2. wire local with upstream
    a->grad = mul_k(local, upstream);
    // free(local);
}

void exp_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    maybe_init_grad(a);
    tensor* local = out;
    mul_k_(local, upstream, a->grad);
}

void log_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    a->grad = TensorLikeFill(a, 0.0);
    float log_e = logf(M_E);
    // non-vectorized form: "(1/a) * log_e;"
    tensor* local = mul_k(div_k(TensorLikeFill(a, 1), a), TensorLikeFill(a, log_e));
    mul_k_(local, upstream, a->grad);
}

void reduce_sum_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // 1. local
    tensor* local = TensorLikeFill(a, 1.0);
    // 2. wire local with upstream
    tensor* upstream_host = COPY_FROM_DEVICE(upstream);
    tensor* broadcasted_upstream = TensorLikeFill(a, upstream_host->data[0]);
    a->grad = mul_k(local, broadcasted_upstream);
    // free(local);
}

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

void batched_matmul_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    tensor* local_a = batched_transpose_k(b); // (B, M, D) -> (B, D, M)
    tensor* local_b = batched_transpose_k(a); // (B, N, M) -> (B, M, N)

    // 2. wire local with upstream
    // upstream(B, N, D) @ b.t(B, D, M) = a_grad(B, N, M)
    a->grad = batched_matmul_k(upstream, local_a);
    // a.t(B, M, N) @ upstream(B, N, D) = b_grad(B, M, D)
    b->grad = batched_matmul_k(local_b, upstream);

    // free(local_a), free(local_b);
}

// a(B, N), idx(B, 1) = out(B, 1)
void select_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    tensor* idx = out->inputs[1];
    int N = a->shape[1];

    // local
    tensor* local = TensorLikeFill(a, 0.0); // (B, N)
    select_set_(local, idx, 1.);

    // downstream
    tensor* upstream_broadcasted = repeat_k(upstream, N); // (B, 1) -> (B, N);
    tensor* a_grad = mul_k(local, upstream_broadcasted);

    // add to existing grad
    maybe_init_grad(a);
    add_k_(a->grad, a_grad, a->grad);

    // grad wrt idx
    maybe_init_grad(idx);
    local = TensorLikeFill(idx, 1.0);
    tensor* idx_grad = mul_k(local, upstream);
    add_k_(idx->grad, idx_grad, idx->grad);
}


void reduce_max_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // todo: use select_set_
    int idx = (int)out->scratch_space[0]->data[0];

    // 1. local
    tensor* local = TensorLikeFill(a, 0.0);
    tensor* local_host = COPY_FROM_DEVICE(local);
    local_host->data[idx] = 1.0;
    COPY_TO_DEVICE(local_host);

    // 2. wire local with upstream
    tensor* upstream_host = COPY_FROM_DEVICE(upstream);
    tensor* broadcasted_upstream = TensorLikeFill(a, upstream_host->data[0]);

    a->grad = mul_k(local_host, broadcasted_upstream);
    // free(local);
}

// select: a(B, N), idx(B, 1) = out(B, 1)
void batched_reduce_max_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    if (a->num_dims!=2){
        printf("[batched_max] Unexpected num_dims\n");
        exit(1);
    }

    int N = a->shape[1];
    tensor* idx = out->scratch_space[0]; // (B, 1)

    // local
    tensor* local = TensorLikeFill(a, 0.0); // (B, N)
    select_set_(local, idx, 1.);

    // downstream
    tensor* upstream_broadcasted = repeat_k(upstream, N); // (B, 1) -> (B, N);
    tensor* a_grad = mul_k(local, upstream_broadcasted);

    // add to existing grad
    maybe_init_grad(a);
    add_k_(a->grad, a_grad, a->grad);

    // free(local);
    // free(a_grad);
    // free(upstream_broadcasted);
}


void batched_flatten_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    maybe_init_grad(a);
    unsafe_add_k_(a->grad, upstream, a->grad);
    // free(local);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fwd kernels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


tensor* batched_flatten_k(tensor* a) {
    int B = a->shape[0];

    if (!a->num_dims==3 && !a->num_dims==4){
        printf("[batched_flatten] Shape error\n");
        exit(1);
    }

    int out_dim = a->size / a->shape[0];
    tensor* out = TensorLikeFill(Tensor(B, out_dim), 0.);

    // use add_k_ on zero initialized out, to avoid needing to
    // impl for-loop copy (_copy_arr) as separate primitive in both backends;
    unsafe_add_k_(out, a, out);
    return out;
}
