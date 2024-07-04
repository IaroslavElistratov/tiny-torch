#include <iostream> // todo: use C only
// #include <time.h>
// #include <stdlib.h>


using namespace std;


// note: assumes a, b are same shape
float* ElementwiseMul(float* a, float* b, int size)
{
    float* out = (float*)malloc(sizeof(float)* size);
    for (int i=0; i<size; i++) {
        out[i] = a[i] * b[i];
    }
    return out;
}

float* Matmul(float* a, float* b, int size)
{
    float* out = (float*)malloc(sizeof(float));;
    for (int i=0; i<size; i++) {
        // de-reference and compute
        *out += a[i] * b[i];
    }
    return out;
}

void Print(float* a, int size)
{
    for (int i=0; i<size; i++) {
        cout << a[i] << ", ";
    }
    cout << endl;
}


float* GenRandomFloat(int num)
{
    float* f_ptr = (float*)malloc(sizeof(float) * num);

    for (int i=0; i<num; i++)
    {
        // https://linux.die.net/man/3/random
        // returns a pseudo-random int between 0 and RAND_MAX
        // normalize to: 0 - 1
        // shift to: -0.5 - 0.5
        f_ptr[i] = ((float)rand() / RAND_MAX) - 0.5;
    }


    return f_ptr;
}


int main() {
    // random num generator init, must be called once
    srand(time(NULL));
    // srand(123);

    int size = 5;

    float* x = GenRandomFloat(size);
    Print(x, size);
    float* w = GenRandomFloat(size);
    Print(w, size);

    float* out = ElementwiseMul(x, w, size);
    Print(out, size);

    float* out1 = Matmul(x, w, size);
    Print(out1, 1);

    // int* out = Matmul1d(x, w, size);
    // cout << *out << endl;
    return 0;
}