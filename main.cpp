#include <iostream> // todo: use C only
// #include <time.h>
// #include <stdlib.h>


using namespace std;
#include <iomanip> // for  input-output manipulation


// note: assumes a, b are same shape
float* ElementwiseMul(float* a, float* b, int size)
{
    float* out = (float*)malloc(sizeof(float)* size);
    for (int i=0; i<size; i++) {
        out[i] = a[i] * b[i];
    }
    return out;
}


// x_max is number columns to get to next row
int index(int x, int y, int x_max) {
    return x * x_max + y;
}


// todo: add tests -- https://github.com/tensorflow/tensorflow/commit/6f4a0e96d853d1d8fe05a8dd8f7ba0cd0fb0e79b#diff-65511a88d2951377144d77a2de94c0f597c4664189d3d5ac730e653560b64f31R259-R342
float* Matmul(float* x, float* w,
        int N, int M, int D)
{
    // (N, M) @ (M, D) = (N, D)
    float* out = (float*)malloc(sizeof(float) * N * D);


    for (int n=0; n<N; n++) {
        for (int d=0; d<D; d++) {
            float sum = 0;
            for (int m=0; m<M; m++){
                // cout << "x_idx: " << n*M+m << endl;
                // cout << "y_idx: " << m*D+d << endl;
                sum += x[n*M+m] * w[m*D+d];
                // cout << endl;
            }
            // cout << "sum: " << sum << endl;
            out[n*D+d] = sum;
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
float* Relu(float* x, int size)
{
    float* out = (float*)malloc(sizeof(float) * size);
    for (int i=0; i<size; i++) {
        out[i] = x[i] > 0.0 ? x[i] : 0.0;
        // mask[i] = x[i] > 0.0 ? 1.0 : 0.0;
    }
    return out;
}

float* GetReluMask(float* x, int size)
{
    float* out = (float*)malloc(sizeof(float) * size);
    for (int i=0; i<size; i++) {
        out[i] = x[i] > 0.0 ? 1.0 : 0.0;
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
float mse(float* x, float* y, int size)
{
    float diff, out = 0;
    for (int i=0; i<size; i++) {
        diff = y[i] - x[i];
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
float* Transpose(float* x, int s1, int s2)
{
    // [S1, S2]
    //   - to go to next row, need to skip S2 elements
    //   - to go to next column, need to skip 1
    int stride_next_row = s2, stride_next_col = 1;

    float* f_ptr = (float*)malloc(sizeof(float) * s1*s2);

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
            f_ptr[idx_orig++] = x[idx_trans];
            // cout << "; idx_trans: " << idx_trans << endl;
        }
    }

    return f_ptr;
}


void print(float* a, int s0, int s1)
{
    for (int i=0, size=s0*s1; i<size; i++) {
        if (i % s1 == 0) cout << endl;
        // easy numpy export:
        // if (i % s1 == 0) cout << "], " << endl << "[";

        // todo: use right justified, and print only 4 points of precision
        // cout << " " << setw(5) << right << a[i] << ", ";

        // %6.1f describes number at least six characters wide, with 1 digit after the decimal point
        printf("%8.4f, ", a[i]);
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

// float* Zeros(int size)
// {
//     float* out = (float*)malloc(sizeof(float) * size);
//     for (int i=0; i<size; i++)
//         out[i] = 0.0;
//     return out;
// }

float* BroadcastScalar(float scalar, int size)
{
    float* f_ptr = (float*)malloc(sizeof(float) * size);
    for (int i=0; i<size; i++)
        f_ptr[i] = scalar;
    return f_ptr;
}

int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int N = 16;
    int M = 2;
    int D = 4;
    int O = 1;

    // *** INIT ***

    float* x = GetRandomFloat(N*M);
    cout << "x: ";
    print(x, N, M);

    float* w1 = GetRandomFloat(M*D);
    cout << "\nw1: ";
    print(w1, M, D);

    float* w2 = GetRandomFloat(D*O);
    cout << "\nw2: ";
    print(w2, D, O);

    // *** FWD ***

    // x(N, M) @ w1(M, D) = out1(N, D)
    float* out1 = Matmul(x, w1, N, M, D);
    cout << "\nmatmul_1: ";
    print(out1, N, D);

    // out2(N, D)
    float* out2 = Relu(out1, N*D);
    cout << "\nrelu: ";
    print(out2, N, D);

    // out2(N, D) @ w2(D, O) = out3(N, O)
    float* out3 = Matmul(out2, w2, N, D, O);
    cout << "\nmatmul_2: ";
    print(out3, N, O);

    // loss
    float* y = BroadcastScalar(0.5, N*O); // dummy label
    float loss = mse(out3, y, N*O);
    cout << "loss :" << loss << endl;

    // *** BWD ***

    // note: below derivatives try to follow this order -- local * upstream

    float dL_dL = 1.0;

    // mse bwd

    // question-now: non deterministic order of args?
    //  - dL_dpow, dL_dneg, dL_w2, dL_out2
    float* dL_dneg = ElementwiseMul(BroadcastScalar(2.0, N*O), BroadcastScalar(dL_dL, N*O), N*O);
    float* dL_dout3 = ElementwiseMul(BroadcastScalar(1.0, N*O), dL_dneg, N*O);

    // matmul 2 bwd

    // dL_dout3(N, O)
    // out2(N, D)
    // w2(D, O)
    //   so: out2.T(D, N) @ dL_dout3(N, O) = dL_w2(D, O)
    //   I went from the ouput shape (D, O) -- this has to be the same bc
    // note: Transpose inputs dims **before** transpose
    float* dL_w2 = Matmul(Transpose(out2, N, D), dL_dout3, D, N, O);
    cout << "\ndL_w2: ";
    print(dL_w2, D, O);

    // out2(N, D)
    //   so: dL_dout3(N, O) @ w2.T(O, D) = dL_out2(N, D)
    float* dL_out2 = Matmul(dL_dout3, Transpose(w2, D, O), N, O, D);

    // relu bwd

    // note: this is kind of gradient checkpointing;
    //  ofc can avoid by making original relu ouput the mask
    //  -- but I found recomputing is a bit cleaner to explain
    float* relu_mask = GetReluMask(out1, N*D);
    float* dL_dout1 = ElementwiseMul(relu_mask, dL_out2, N*D);

    // matmul 1 bwd

    // dL_dout1(N, D)
    // x(N, M)
    // w1(M, D)
    //   so: x.T(M, N) @ dL_dout1(N, D) = dL_dw1(M, D)
    float* dL_dw1 = Matmul(Transpose(x, N, M), dL_dout1, M, N, D);
    cout << "\ndL_dw1: ";
    print(dL_dw1, M, D);

    // todo: write to file
    return 0;
}
