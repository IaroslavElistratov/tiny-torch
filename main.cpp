#include <iostream> // todo: use C only
// #include <time.h>
// #include <stdlib.h>


using namespace std;


// // note: assumes a, b are same shape
// float* ElementwiseMul(float* a, float* b, int size)
// {
//     float* out = (float*)malloc(sizeof(float)* size);
//     for (int i=0; i<size; i++) {
//         out[i] = a[i] * b[i];
//     }
//     return out;
// }


// x_max is number columns to get to next row
int index(int x, int y, int x_max) {
    return x * x_max + y;
}


// todo: add tests
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

void Print(float* a, int s0, int s1)
{
    int size = s0 * s1;

    for (int i=0; i<size; i++) {
        if (i % s1 == 0) cout << endl;
        // todo: use right justified, and print only 4 points of precision
        cout << " " << a[i] << ", ";
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


int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    // (N, M) @ (M, D) = (N, D)
    int N = 2;
    int M = 3;
    int D = 1;

    int x_size = N*M;
    float* x = GetRandomFloat(x_size);
    cout << "x: ";
    Print(x, N, M);

    int w_size = M*D;
    float* w = GetRandomFloat(w_size);
    cout << "w: ";
    Print(w, M, D);

    cout << endl;
    float* out1 = Matmul(x, w, N, M, D);
    Print(out1, N, D);

    // todo: write to file
    return 0;
}
