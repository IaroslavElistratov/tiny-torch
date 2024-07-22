#include <iostream> // todo: use C only
// #include <time.h>
// #include <stdlib.h>


using namespace std;
#include <iomanip> // for  input-output manipulation

# define NUM_EP 5
# define LR 0.02
// #define DEBUG  1


#ifdef DEBUG
#define print(f_p, msg) _print(f_p, msg)
#else
#define print(f_p, msg)
#endif

struct tensor {
    float* data;
    int shape[2];
    // to avoid
    //  int size = x->shape[0] * x->shape[1];
    int size;
};

// forward declaration
float* GetRandomFloat(int num);

tensor* Tensor(int s1, int s2)
{
    tensor* t = (tensor*)malloc(sizeof(tensor));

    t->size = s1*s2;

    t->shape[0] = s1;
    t->shape[1] = s2;

    t->data = GetRandomFloat(s1*s2);

    return t;
}

/*
for convince to avoid:
    int N = x->shape[0], M = x->shape[1];
    tensor* out = Tensor(N, D);
*/
tensor* TensorLike(tensor* t)
{
    int s1 = t->shape[0], s2 = t->shape[1];
    return Tensor(s1, s2);
}

tensor* TensorLikeFill(tensor* t, float value)
{
    tensor* t_new = TensorLike(t);
    for (int i=0; i<t_new->size; i++)
        t_new->data[i] = value;
    return t_new;
}


tensor* ElementwiseMul(tensor* a, tensor* b)
{
    // todo: here and in all ops, assert same size
    tensor* out = TensorLike(a);
    for (int i=0; i<out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return out;
}


// x_max is number columns to get to next row
int index(int x, int y, int x_max) {
    return x * x_max + y;
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


void _print(tensor* t, const char* msg)
{
    printf("\n%s: ", msg);

    for (int i=0, row_len = t->shape[1]; i<t->size; i++) {
        if (i % row_len == 0) cout << endl;
        // easy numpy export:
        // if (i % row_len == 0) cout << "], " << endl << "[";

        // todo: use right justified, and print only 4 points of precision
        // cout << " " << setw(5) << right << t->data[i] << ", ";

        // %6.1f describes number at least six characters wide, with 1 digit after the decimal point
        printf("%8.4f, ", t->data[i]);
    }
    cout << endl;
}


float* GetRandomFloat(int num)
{
    float* f_ptr = (float*)malloc(sizeof(float) * num);

    for (int i=0; i<num; i++)
    {
        // https://linux.die.net/man/3/random
        // returns a pseudo-random int between 0 and RAND_MAX
        // normalize to: 0 - 1
        // shift to: -0.5 - 0.5
        // todo: maybe wrongly truncating to 0 due to int division? No bc C promotes args?
        f_ptr[i] = ((float)rand() / RAND_MAX) - 0.5;
    }


    return f_ptr;
}

void sgd(tensor* w, tensor* grad_w)
{
    for (int i=0; i<w->size; i++) {
        w->data[i] -= grad_w->data[i] * LR;
    }
}

float train_step(tensor* x, tensor* w1, tensor* w2)
{
    // *** FWD ***

    // x(N, M) @ w1(M, D) = out1(N, D)
    tensor* out1 = Matmul(x, w1);
    print(out1, "matmul_1");

    // out2(N, D)
    tensor* out2 = Relu(out1);
    print(out2, "relu");

    // out2(N, D) @ w2(D, O) = out3(N, O)
    tensor* out3 = Matmul(out2, w2);
    print(out3, "matmul_2");

    // loss
    tensor* y = TensorLikeFill(out3, 0.5); // dummy label
    float loss = mse(out3, y);
    // cout << "loss :" << loss << endl;

    // *** BWD ***

    // note: below derivatives try to follow this order -- local * upstream
    // question-now: non deterministic order of args?
    //  - dL_dpow, dL_dneg, dL_dw2, dL_out2

    float dL_dL = 1.0;
    // float dL_dsum = 1.0;

    // mse bwd

    // "reduce_sum" bwd
    tensor* dL_dpow = TensorLikeFill(y, dL_dL);

    // "**2" bwd
    // todo: avoid recomputing this during backward
    tensor* diff = TensorLike(y);
    for (int i=0; i<diff->size; i++) {
        diff->data[i] = y->data[i] - out3->data[i];
    }
    tensor* local_dpow = ElementwiseMul(TensorLikeFill(diff, 2.0), diff);
    tensor* dL_dneg = ElementwiseMul(local_dpow, dL_dpow);

    // "-" bwd
    tensor* dL_dout3 = ElementwiseMul(TensorLikeFill(dL_dneg, -1.0), dL_dneg);

    // matmul 2 bwd

    // dL_dout3(N, O)
    // out2(N, D)
    // w2(D, O)
    //   so: out2.T(D, N) @ dL_dout3(N, O) = dL_dw2(D, O)
    //   I went from the ouput shape (D, O) -- this has to be the same bc
    tensor* dL_dw2 = Matmul(Transpose(out2), dL_dout3);
    _print(dL_dw2, "dL_dw2");

    // out2(N, D)
    //   so: dL_dout3(N, O) @ w2.T(O, D) = dL_out2(N, D)
    tensor* dL_out2 = Matmul(dL_dout3, Transpose(w2));

    // relu bwd

    // note: this is kind of gradient checkpointing;
    //  ofc can avoid by making original relu ouput the mask
    //  -- but I found recomputing is a bit cleaner to explain
    tensor* relu_mask = GetReluMask(out1);
    tensor* dL_dout1 = ElementwiseMul(relu_mask, dL_out2);

    // matmul 1 bwd

    // dL_dout1(N, D)
    // x(N, M)
    // w1(M, D)
    //   so: x.T(M, N) @ dL_dout1(N, D) = dL_dw1(M, D)
    tensor* dL_dw1 = Matmul(Transpose(x), dL_dout1);
    _print(dL_dw1, "dL_dw1");

    // *** Optim Step ***
    sgd(w1, dL_dw1);
    sgd(w2, dL_dw2);

    return loss;
}


int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int N = 16;
    int M = 2;
    int D = 4;
    int O = 1;

    // *** Init ***
    tensor* x = Tensor(N, M);
    print(x, "x");

    tensor* w1 = Tensor(M, D);
    print(w1, "w1");

    tensor* w2 = Tensor(D, O);
    print(w2, "w2");

    // *** Train Step ***
    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        float loss = train_step(x, w1, w2);
        cout << "\nep: " << ep_idx << "; loss: " << loss << endl;

        print(w1, "w1");
        print(w2, "w2");
    }

    // todo: write to file
    return 0;
}
