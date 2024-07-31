#include <iostream> // todo: use C only
// #include <stdlib.h>
// #include <iomanip> // for  input-output manipulation
using namespace std;

// #include "nn.h"
#include "tensor.cpp"
#include "ops.cpp"
#include "utils.cpp"


#define NUM_EP 2
#define LR 0.02
#define DEBUG  1

#ifdef DEBUG
#define print(f_p) _print(f_p)
#else
#define print(f_p)
#endif

void sgd(tensor* w, tensor* grad_w)
{
    for (int i=0; i<w->size; i++)
        w->data[i] -= grad_w->data[i] * LR;
}

float train_step(tensor* x, tensor* w1, tensor* w2)
{
    // *** FWD ***

    // x(N, M) @ w1(M, D) = out1(N, D)
    tensor* out1 = matmul(x, w1);
    set_name(out1, "matmul_1"); print(out1);

    // out2(N, D)
    tensor* out2 = relu(out1);
    set_name(out2, "relu"); print(out2);

    // out2(N, D) @ w2(D, O) = out3(N, O)
    tensor* out3 = matmul(out2, w2);
    set_name(out3, "matmul_2"); print(out3);

    // loss
    tensor* y = TensorLikeFill(out3, 0.5); // dummy label
    tensor* loss = reduce_sum(pow(sub(y, out3), 2));
    cout << "loss :" << loss->data[0] << endl;

    // todo: non deterministic order of args?

    // *** Backward ***
    loss->backward(loss);

    // *** Optim Step ***
    sgd(w1, w1->grad);
    sgd(w2, w2->grad);

    graphviz(loss);

    return loss->data[0];
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
    set_name(x, "x"); print(x);

    tensor* w1 = Tensor(M, D);
    set_name(w1, "w1"); print(w1);

    tensor* w2 = Tensor(D, O);
    set_name(w2, "w2"); print(w2);

    // *** Train Step ***
    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        float loss = train_step(x, w1, w2);
        cout << "\nep: " << ep_idx << "; loss: " << loss << endl;

        print(w1);
        print(w2);
    }

    // todo: write to file
    return 0;
}


int _main() {
    srand(123);

    cout << "sizeof(tensor): " << sizeof(tensor) << endl << endl;

    // only used for shape inference
    tensor* _ = Tensor(2, 2);

    tensor* a = TensorLikeFill(_, 2.0);
    set_name(a, "a"); print(a);

    tensor* b = TensorLikeFill(_, 2.0);
    set_name(b, "b"); print(b);

    tensor* c = add(a, b);
    set_name(c, "c"); print(c);

    tensor* d = TensorLikeFill(_, 3.0);
    set_name(d, "d"); print(d);

    tensor* e = mul(c, d);
    set_name(e, "e"); print(e);

    // e(2,2) @ f(2,5) = (2,5)
    tensor* f = Tensor(2, 5);
    set_name(f, "f"); print(f);

    tensor* g = matmul(e, f);
    set_name(g, "g"); print(g);

    // mse
    tensor* y = TensorLikeFill(g, 0.0);
    tensor* loss = reduce_sum(pow(sub(y, g), 2));

    cout << "loss: " << loss->data[0] << endl;

    loss->backward(loss);
    graphviz(loss);
    return 0;
}
