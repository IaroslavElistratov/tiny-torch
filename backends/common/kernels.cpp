/*
implementations of these are exactly the same for CPU and CUDA -- reuse

Temporarily commented out ops which don't yet have CUDA impl, so that compilation
doesn't fail bc linker can't find _k expected by these ops
*/


void add_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);

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
}

void sub_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    tensor* a_local = TensorLikeFill(a, 1.0);
    tensor* b_local = TensorLikeFill(b, -1.0);

    // downstream = local * upstream
    tensor* curr_a_grad = mul_k(a_local, upstream);
    tensor* curr_b_grad = mul_k(b_local, upstream);

    maybe_init_grad(a);
    add_k_(a->grad, curr_a_grad, a->grad);
    maybe_init_grad(b);
    add_k_(b->grad, curr_b_grad, b->grad);
}

void mul_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
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
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // 1. local

    // Even though you gonna store e.g. a transpose in a->grad (and not a itself)
    //  it does not make sense to allocate like below because the dims are logical but
    //  mem is contiguous anyway -- so for both of the constructors below obviously same memory will be allocated
    //    EmptyFloat(a->shape[1], a->shape[0]);   // reversed shapes to store Transpose
    //    EmptyFloatLike(a);

    // note: should allocate a_buff not with a.size but w b.size
    // note: using this intermediate variable instead of directly
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
}

void div_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    tensor* b = out->inputs[1];

    // local
    tensor* a_local = div_k(TensorLikeFill(a, 1.0), b);
    tensor* b_local = neg_k(div_k(a, pow_k(b, 2)));

    // downstream

    // "a->grad = mul_k(a_local, upstream)" overwrite's input grad, the below does "+=" to it
    maybe_init_grad(a);
    maybe_init_grad(b);

    tensor* a_grad = mul_k(a_local, upstream);
    tensor* b_grad = mul_k(b_local, upstream);
    // does "+="
    add_k_(a->grad, a_grad, a->grad);
    add_k_(b->grad, b_grad, b->grad);
}

void repeat_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];

    int axis = out->non_grad_inputs[0];
    if (axis==0){
        // sum together each column of upstream
        //  option 1: add "batched_reduce_sum_k(..., axis=0)"
        //  option 2: to avoid modifying batched_reduce_sum, use: transpose -> batched_reduce_sum -> transpose

        // todo: maybe a cleaner way is to add "axis" argument to batched_reduce_sum,
        //  but implement it in terms of transposes (no need to change the cuda kernel)

        tensor* transposed = transpose_k(upstream); // (num_repeats, N) -> (N, num_repeats)
        tensor* reduced = batched_reduce_sum_k(transposed); // (N, 1)
        a->grad = transpose_k(reduced); // (1, N)

    } else if (axis==1){
        // sum together each row of upstream
        a->grad = batched_reduce_sum_k(upstream);
    }
}

// reduce_sum: (B, N) -> (B, 1)
void batched_reduce_sum_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0]; // (B, N)

    // important to fill with 0's if we gonna "+=" to it below.
    // If we instead simply overwrite it, then wouldn't matter,
    // but bc we do "+=" it does matter (if there's any garbage
    // data, the grad will be += to it)
    maybe_init_grad(a);

    int N = a->shape[1];
    tensor* local = TensorLikeFill(a, 1.0); // (B, 1)
    tensor* upstream_broadcasted = repeat_k(upstream, /*axis = */ 1, /*num_repeats = */ N); // (B, 1) -> (B, N)
    tensor* a_grad = mul_k(local, upstream_broadcasted);

    // add to existing a grad
    add_k_(a->grad, a_grad, a->grad);
}

void pow_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    // 1. local
    // store local in grad in the grad field
    // todo-low: mem leak
    // todo: below assumes exponent=2 ?
    tensor* local = mul_k(TensorLikeFill(a, 2.0), a);
    // 2. wire local with upstream
    a->grad = mul_k(local, upstream);
}

void exp_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    tensor* local = out;
    tensor* curr_a_grad = mul_k(local, upstream);
    maybe_init_grad(a);
    add_k_(a->grad, curr_a_grad, a->grad);
}

void log_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    a->grad = TensorLikeFill(a, 0.0);

    // approximately 2.718282 C Math exp() Function e is the base of the natural system of logarithms (approximately 2.718282)
    // Some implementations of the <math. h> library include a constant M_E
    float log_e = logf(M_E);
    // non-vectorized form: "(1/a) * log_e;"
    tensor* local = mul_k(div_k(TensorLikeFill(a, 1), a), TensorLikeFill(a, log_e));
    mul_k_(local, upstream, a->grad);
}

void reduce_sum_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    // 1. local
    tensor* local = TensorLikeFill(a, 1.0);
    // 2. wire local with upstream
    // make upstream and local to be same shape (currently upstream is a scalar, while local is a 2d tensor)

    // Note: cannot just "upstream->data[0]", because the data can be on CUDA;
    // Also there's no need to call "COPY_TO_DEVICE", as it will be called from TensorLikeFill
    tensor* upstream_host = COPY_FROM_DEVICE(upstream);

    tensor* broadcasted_upstream = TensorLikeFill(a, upstream_host->data[0]);
    a->grad = mul_k(local, broadcasted_upstream);
}

// todo-high: it doesn't make sense to have a transpose_bwd bc it just calls transpose -- at the moment this fn is used bc calling convention (args signature) for _bwd funcs is different from the fwd funcs
// todo: bwd formula
void transpose_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    a->grad = transpose_k(upstream);
}

void neg_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    tensor* local = TensorLikeFill(a, -1.0);
    a->grad = mul_k(local, upstream);
}

// comment: shapes are same as matmul_bwd, but with additional (B,) dim first
void batched_matmul_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
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
}

/*
answer-now:
these are almost identical and it makes sense: select_bwd batched_reduce_max_bwd -- for reduce_max_bwd you just need to make one more step to get the idxs, and for select_bwd they are already provided as arguments

void select_bwd(tensor* upstream, tensor* out) {
    idxs = out->inputs[1];
    a->grad = TensorLikeFill(a, 0.0);
    select_set(a->grad, idxs, 1);
    mul_k(upstream, a->grad);
}

void batched_reduce_max_bwd(tensor* upstream, tensor* out) {
    idxs = // get_max_idx ...
    a->grad = TensorLikeFill(a, 0.0);
    select_set(a->grad, idxs, 1);
    mul_k(upstream, a->grad);
}
*/

// a(B, N), idx(B, 1) = out(B, 1)
void select_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    tensor* idx = out->inputs[1];
    int N = a->shape[1];

    // local
    tensor* local = TensorLikeFill(a, 0.0); // (B, N)
    select_set_(local, idx, 1.);

    // downstream
    tensor* upstream_broadcasted = repeat_k(upstream, /*axis = */ 1, /*num_repeats = */ N); // (B, 1) -> (B, N);
    tensor* a_grad = mul_k(local, upstream_broadcasted);

    // add to existing grad
    maybe_init_grad(a);
    add_k_(a->grad, a_grad, a->grad);

    // // todo: rm? If do, in autograd need to check "if inp->grad != NULL" before lprint'ing
    // grad wrt idx
    maybe_init_grad(idx);
    local = TensorLikeFill(idx, 1.0);
    tensor* idx_grad = mul_k(local, upstream);
    add_k_(idx->grad, idx_grad, idx->grad);
}

// made reduce_max_bwd shared between gpu and cpu by allowing kernels to record
// some additional info during fwd kernel, and simply access and use this field in bwd

// todo-high: too many copy_to/from
void reduce_max_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    // todo-now: use select_set? This will also avoid needing to call COPY_TO_DEVICE, FROM_DEVICE here (when setting local->data[idx]=1)
    int idx = (int)out->scratch_space[0]->data[0];
    // 1. local
    tensor* local = TensorLikeFill(a, 0.0);
    tensor* local_host = COPY_FROM_DEVICE(local);
    local_host->data[idx] = 1.0;
    COPY_TO_DEVICE(local_host); // semantically this is "local"

    // 2. wire local with upstream
    // copy to cpu before accessing t->data
    tensor* upstream_host = COPY_FROM_DEVICE(upstream);
    // make upstream and local to be same shape (currently upstream is a scalar, while local is a 2d tensor)
    tensor* broadcasted_upstream = TensorLikeFill(a, upstream_host->data[0]);

    tensor* curr_a_grad = mul_k(local_host, broadcasted_upstream);
    maybe_init_grad(a);
    add_k_(a->grad, curr_a_grad, a->grad);
}

// select: a(B, N), idx(B, 1) = out(B, 1)
void batched_reduce_max_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];

    int N = a->shape[1];
    tensor* idx = out->scratch_space[0]; // (B, 1)

    // local
    // need to set to ones at these idxs, bc previously repeat_k will return a new tensor
    // with these elements at indexes (from the original tensor) copied -- so it's not a
    // view on the original elements and therefore can't set it by modifying output of
    // select_k elements
    tensor* local = TensorLikeFill(a, 0.0); // (B, N)
    select_set_(local, idx, 1.);

    // downstream
    tensor* upstream_broadcasted = repeat_k(upstream, /*axis = */ 1, /*num_repeats = */ N); // (B, 1) -> (B, N);
    tensor* a_grad = mul_k(local, upstream_broadcasted);

    // add to existing grad
    maybe_init_grad(a);
    add_k_(a->grad, a_grad, a->grad);
}


void batched_flatten_bwd(tensor* upstream, tensor* out) {
    assert_input(upstream, out->num_dims);
    tensor* a = out->inputs[0];
    maybe_init_grad(a);
    // reshape upstream into the shape of a
    unsafe_add_k_(a->grad, upstream, a->grad);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fwd kernels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


tensor* batched_flatten_k(tensor* a) {
    // ugly, but uses assert_input for contiguity and device check
    assert_input(a, -1);
    if (!a->num_dims==3 && !a->num_dims==4){
        printf("[batched_flatten] Shape error\n");
        exit(1);
    }
    int B = a->shape[0];

    // collapsing all dims of the tensor except B (0-th dim);
    // this line is more concise then iterating over a.shape[1:] and multiplying them
    int out_dim = a->size / a->shape[0];

    // inputs to this kernel can be 3d, 4d -- but the output is always 2d (all dims flattened except for the batch dim)
    // todo: these copies aren't needed -- can just change strides on the upstream, but
    //   in this case the rest of kernels need to support strided input (output of this kernel)
    // todo: memory leak
    tensor* out = TensorLikeFill(Tensor(B, out_dim), 0.);

    // use add_k_ on zero initialized out, to avoid needing to
    // impl for-loop copy (_copy_arr) as separate primitive in both backends;
    // 
    // I ended up special casing this fn anyway (using "unsafe_add_k_" instead of using "add_k_"),
    // bc the former only needs to assert same input->size, but the latter needs to assert
    // same input->shape (more restrictive) -- but in general (for all other cuda kernels
    // using _launch_binary_elementwise) the more restrictive checks are helpful
    unsafe_add_k_(out, a, out);
    return out;
}
