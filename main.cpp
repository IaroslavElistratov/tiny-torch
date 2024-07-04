#include <iostream> // todo: use C only
// #include <time.h>
// #include <stdlib.h>


using namespace std;


// note: assumes a, b are same shape
int* ElementwiseMul(int* a, int* b, int size)
{
    int* out = (int*)malloc(size * sizeof(int));
    for (int i=0; i<size; i++) {
        out[i] = a[i] * b[i];
    }
    return out;
}

int* Matmul1d(int* a, int* b, int size)
{
    int* out = (int*)malloc(sizeof(int));;
    for (int i=0; i<size; i++) {
        // de-reference and compute
        *out += a[i] * b[i];
    }
    return out;
}

void Print(int* a, int size)
{
    for (int i=0; i<size; i++) {
        cout << a[i] << ", ";
    }
    cout << endl;
}


float* GenRandomFloat()
{
    float* f_prt = (float*)malloc(sizeof(float));

    // https://linux.die.net/man/3/random
    // returns a pseudo-random int between 0 and RAND_MAX
    // normalize to: 0 - 1
    // shift to: -0.5 - 0.5
    float r = ((float)rand() / RAND_MAX) - 0.5;

    // write to heap memory
    *f_prt = r;

    return f_prt;
}


int main() {
    // random num generator init, must be called once
    srand(time(NULL));
    // srand(123);

    float* f = GenRandomFloat();
    cout << "GenRandomFloat: " << *f << endl;;

    // // todo: make it random numbers
    // int x[5] = {2, 4, 5, 8, 1};
    // int w[5] = {2, 2, 2, 2, 2};

    // int size = sizeof(x) / sizeof(int);
    // // cout << size;

    // // int* out = ElementwiseMul(x, w, size);
    // // Print(out, size);

    // int* out = Matmul1d(x, w, size);
    // cout << *out << endl;
    return 0;
}