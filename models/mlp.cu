#include <iostream> // todo: use C only
using namespace std;

#define DEVICE CPU


#include "../nn.h"
#include "../tensor.cpp"
#include "../ops.cpp"
#include "../composite_ops.cpp"
#include "../cifar10.cpp"
#include "../print.cpp"


#define NUM_EP 10
#define LR 0.02


void sgd(tensor* w) {
    tensor* w_local = COPY_FROM_DEVICE(w);
    tensor* w_grad_local = COPY_FROM_DEVICE(w->grad);

    for (int i=0; i<w->size; i++)
        w_local->data[i] -= w_grad_local->data[i] * LR;

    COPY_TO_DEVICE(w_local);
    w->data = w_local->data;
}


float train_step(tensor* x, tensor* w1, tensor* w2)
{
    // *** FWD ***

    // x(N, M) @ w1(M, D) = out1(N, D)
    tensor* out1 = matmul(x, w1);
    set_name(out1, "matmul_1"); // print(out1);

    // out2(N, D)
    tensor* out2 = relu(out1);
    set_name(out2, "relu"); // print(out2);

    // out2(N, D) @ w2(D, O) = out3(N, O)
    tensor* out3 = matmul(out2, w2);
    set_name(out3, "matmul_2"); // print(out3);

    // loss
    tensor* y = TensorLikeFill(out3, 0.5); // dummy label
    tensor* loss = reduce_sum(pow(sub(y, out3), 2));

    // *** Backward ***
    loss->backward(loss);

    // *** Optim Step ***
    sgd(w1);
    sgd(w2);

    graphviz(loss);

    return COPY_FROM_DEVICE(loss)->data[0];
}


int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);
    set_backend_device();

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

    // *** Train ***
    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        float loss = train_step(x, w1, w2);
        cout << "\nep: " << ep_idx << "; loss: " << loss << endl;
    }

    return 0;
}
