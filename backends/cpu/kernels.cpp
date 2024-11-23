#include <math.h> // log, pow


tensor* neg_k(tensor*);
tensor* pow_k(tensor*, int);
tensor* transpose_k(tensor*);
tensor* mul_k(tensor*, tensor*);
tensor* batched_transpose_k(tensor*);
tensor* batched_reduce_sum_k(tensor*);
void reduce_sum_bwd(tensor*, tensor*);


void _strcp(char* src, char* dst, int size){
    for (int i=0; i<size; i++) {
        dst[i] = src[i];
        printf("%c\n", src[i]);
    }
}

void _copy_arr(float* src, float* dst, int size) {
    for (int i=0; i<size; i++)
        dst[i] = src[i];
}

// todo:
// - to preserve old behavior, change ops to:
//     - index from the right (inner most dims), NOT from the left (outer most)
//     - loop over the outer-most dim as well
// - IOW, in nn.h when increase shape from 3 to 4, all the existing ops (indexing from left) might break?
// - On the other hand, the kernels are specialized to particular num dims -- so will break anyway

// note: naming convention: assume "_elementwise", by default -- and ask to specify otherwise (if not elementwise) -- e.g. reduce_, matrix_, etc
// todo: ? use capital letters for functions that allocate heap mem, use lowercase otherwise
// note: below derivatives try to follow this order -- local * upstream

// **** kernels *****
//   - operate on tensors
//   - allocate ouput buff
//      - return new tensor


//    binary elementwise


tensor* add_k_(tensor* a, tensor* b, tensor* out) {
    // Previously was hardcoding a at for a specific index here -- conv_bwd uses add (which calls add_k_) for both 3d tensors (grads wrt input) AND 4d tensors (grads wrt kernels)
    // todo: here and in all ops, assert same size
    for (int i=0; i<out->size; i++)
        // comment: note incrementing index like this in a loop, assumes contiguous data
        // out->data[i] = a->data[i] + b->data[i];
        out->data[at(out, i)] = a->data[at(a, i)] + b->data[at(b, i)];
    return out;
}

// todo: use "const tensor*" instead of "tensor"
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
                // n*M, because for each "n" you skip "M" values

                // uses index_ instead of "a->data[n * M + m * 1]" as to not assume strides
                // printf("index_2d(a, n, m): %i\n", index_2d(a, n, m));
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


tensor* div_k(tensor* a, tensor* b) {

    // todo-now: add these checks to every function
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

    // 1 1 1
    // 2 2 2
    // 3 ...
    for (int b=0; b<B; b++){
        // points to the first element of the current b
        float* curr_a = a->data + b;
        // here indexing includes multiplying by "out->stride[0]" bc
        // out is a 2d tensor (for curr_a adding "out->stride[0]" is
        // not needed bc curr_a is (B, 1))
        float* curr_out = out->data + (b * out->stride[0]);
        for (int i=0; i<num_repeats; i++){
            *(curr_out+i) = *(curr_a);
        }
    }
    return out;
}


tensor* select_k(tensor* a, tensor* idx) {

    // expect shapes:
    //  a(s1, s2)
    //  idx(s1, 1)
    //      also each element of idx's first dim should be in range 0-s2
    //      (bc the first dim in idx will be used to index  the 2nd dim of a)
    //      this latter condition is not being checked here
    if (a->num_dims!=2 || idx->num_dims!=2 || idx->shape[1]!=1 || idx->shape[0]!=a->shape[0]) {
        printf("[select] Error shape");
        exit(1);
    }

    int B = a->shape[0];
    tensor* out = Tensor(B, 1);

    for (int b=0; b<B; b++){
        float* curr_a = a->data + (a->stride[0]*b);
        // note: additional constraint on input: idx->data should be ints;
        // below:
        //   1) take pointer to the *first* element of the current batch example in a (curr_a)
        //   2) add it idx for the current batch example (currently all tensors are floats, so need to cast idx->data to int before doing it)
        //   3) the result of the above is a pointer, so de-reference it
        out->data[b] = *(curr_a + (int)idx->data[b]);
        // printf("curr_idx: %i", (int)idx->data[b]);
    }
    return out;
}


tensor* select_set_(tensor* a, tensor* idx, int value) {
    if (a->num_dims!=2 || idx->num_dims!=2 || idx->shape[1]!=1 || idx->shape[0]!=a->shape[0]) {
        printf("[select_set_] Error shape");
        exit(1);
    }
    int B = a->shape[0];
    for (int b=0; b<B; b++){
        float* curr_a = a->data + (a->stride[0]*b);
        *(curr_a + (int)idx->data[b]) = value;
    }
    return out;
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
    for (int i=0; i<out->size; i++)
        out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
    return out;
}

void relu_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];
    // local
    // todo: avoid re-computing
    //  this is kind of gradient checkpointing;
    //  ofc can avoid by making original relu ouput the mask
    //  -- but I found recomputing is a bit cleaner to explain

    tensor* local = TensorLike(a);
    for (int i=0; i<local->size; i++) {
        local->data[i] = a->data[i] > 0.0 ? 1.0 : 0.0;
    }

    a->grad = mul_k(local, upstream);
    // free(local);
}


// todo-high:
// By modifying strides, for example, an array can be transposed
// or reshaped at zero cost (no memory needs to be copied).
// question-now: here and in batched_transpose_k, return a contiguous copy instead of original?
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


// logic for max and batched_max were copied from: reduce_sum and batched_reduce_sum
//  todo-high: add some common logic to abstract this repeated stuff away.
//    - e.g. lua-torch implements a macro, where you only need to specify body of the for loop (op-specific),
//      the rest (common logic) is in the code of the macro itself and thus doens't need to duplciated for each new op
tensor* reduce_max_k(tensor* a) {
    // set inital minimum to the first element
    tensor* out = TensorScalarFill(a->data[0]);
    // store at 0-th location in the scratch space array
    out->scratch_space[0] = TensorLikeFill(out, 0.);
    for (int i=0; i<a->size; i++) {
        if (a->data[i] > out->data[0]){
            out->data[0] = a->data[i];
            // will be used in reduce_max_bwd
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
        // todo: use tensor_like to preserve strides of original out? Well original curr_out is contiguous anyway
        // todo: use view_3d instead of manually doing it here, the problem is view_3d does not handle 2d view on 3d tensor (IOW here a dim actually needs to collapse, resulting in a 2d view)

        // comment: problem is out[i] is float, not Tensor
        //  curr_out is a scratch throw away tensor, needed bc matmul_k_ expects a Tensor struct for the output argument
        // todo: find a proper fix to the issue above: creating these throw away tensors creates memory leaks -- at least need to free them
        tensor* curr_out = TensorNoData(N, D);
        curr_out->data = out->data + (i * out->stride[0]);

        // comment: a[i], b[i], out[i] is incorrect. Bc such indexing would return i-th element of the tensor, but I don't want the i-th element, I want i-th row! So use 
        // todo: index_2d?

        tensor* curr_a = TensorNoData(N, M);
        curr_a->data = a->data + (i * a->stride[0]);

        tensor* curr_b = TensorNoData(M, D);
        curr_b->data = b->data + (i * b->stride[0]);

        matmul_k_(curr_a, curr_b, curr_out);
    }
    return out;
}


tensor* batched_flatten_k(tensor* a) {
    int B = a->shape[0], out_dim = -1;

    if (a->num_dims==3)
        out_dim = a->shape[1] * a->shape[2];
    else if (a->num_dims==4)
        out_dim = a->shape[1] * a->shape[2] * a->shape[3];
    else {
        printf("[batched_flatten] Error");
        exit(1);
    }

    // inputs to this kernel can be 3d, 4d -- but the output is always 2d (all dims flattened except for the batch dim)
    tensor* out = Tensor(B, out_dim);

    for (int i=0; i<a->size; i++)
        out->data[i] = a->data[i];
    return out;
}

void batched_flatten_bwd(tensor* upstream, tensor* out) {
    tensor* a = out->inputs[0];

    // todo: these copies aren't needed -- can just change strides and shapes on the upstream
    if (!a->num_dims==3 && !a->num_dims==4){
        printf("[batched_flatten] Error");
        exit(1);
    }
    a->grad = TensorLike(a);

    // reshape upstream into the shape of a
    for (int i=0; i<upstream->size; i++)
        a->grad->data[i] = upstream->data[i];

    // free(local);
}


tensor* batched_reduce_sum_k(tensor* a) {

    if (a->num_dims!=2){
        printf("[batched_reduce_sum] Error");
        exit(1);
    }

    int B = a->shape[0], N = a->shape[1];
    tensor* out = Tensor(B, 1);

    for (int b=0; b<B; b++){
        // a[i], b[i], out[i] is incorrect. Bc such indexing would return
        // i-th element of the tensor, but I don't want the i-th element,
        // I want i-th row! So use:
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

// todo-now: for all "batched" kernels can I just reshape input to (B*first_dim), run regular max kernel and then reshape back?

// todo-now: to re-use this fwd kernel for CUDA, need to use strides in cuda kernels (instead of indexing manually)
// tensor* local_a = transpose_k(b); // (B, M, D) -> (B, D, M)
// comment: this is equivalent to numpy's np.transpose(x, axes=(0, 2, 1))
tensor* batched_transpose_k(tensor* x){
    int shape_1 = x->shape[1];
    x->shape[1] = x->shape[2];
    x->shape[2] = shape_1;

    int stride_1 = x->stride[1];
    x->stride[1] = x->stride[2];
    x->stride[2] = stride_1;
    return x;
}
