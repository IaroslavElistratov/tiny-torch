// #include <iostream> // todo: use C only
// using namespace std;

#include "nn.h"

// have split _print into these 2 fns so that I can use
// the inner fn (print_kernel) with float arrays
void print_kernel(float* data, int size, int row_len, char* msg)
{
    printf("\n%s: ", msg);

    for (int i=0; i<size; i++) {
        if (i % row_len == 0) cout << endl;
        // easy numpy export:
        // if (i % row_len == 0) cout << "], " << endl << "[";

        // todo: use right justified, and print only 4 points of precision
        // cout << " " << setw(5) << right << t->data[i] << ", ";

        // %6.1f describes number at least six characters wide, with 1 digit after the decimal point
        printf("%8.4f, ", data[i]);
    }
    cout << endl;
}

void _print(tensor* t, char* msg)
{
    print_kernel(t->data, t->size, t->shape[1], msg);
}
