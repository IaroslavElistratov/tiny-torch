#include <iostream>
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

int main() {
    int x[5] = {2, 4, 5, 8, 1};
    int w[5] = {2, 2, 2, 2, 2};

    int size = sizeof(x) / sizeof(int);
    // cout << size;

    // int* out = ElementwiseMul(x, w, size);
    // Print(out, size);

    int* out = Matmul1d(x, w, size);
    cout << *out << endl;
    return 0;
}