// #include <iostream> // todo: use C only
// using namespace std;

#include "nn.h"

void _print(tensor* t, char* msg)
{
    printf("\n%s: ", msg);

    for (int i=0, row_len=t->shape[1]; i<t->size; i++) {
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
