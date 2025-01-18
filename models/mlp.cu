#include <iostream> // todo: use C only

#define DEVICE CPU


#include "../nn.h"
#include "../tensor.cpp"
#include "../ops.cpp"
#include "../composite_ops.cpp"
#include "../cifar10.cpp"
#include "../print.cpp"
#include "../optim.cpp"
#include "../codegen.cpp"

#define NUM_EP 1
#define LR 0.02


float train_step(tensor* x, tensor* w1, tensor* w2)
{
    // *** Net ***

    // x(N, M) @ w1(M, D) = out1(N, D)
    tensor* out1 = matmul(x, w1);
    set_name(out1, "matmul_1");

    // out2(N, D)
    tensor* out2 = relu(out1);
    set_name(out2, "relu");

    // out2(N, D) @ w2(D, O) = out3(N, O)
    tensor* out3 = matmul(out2, w2);
    set_name(out3, "matmul_2");

    // *** Loss fn ***
    tensor* y = TensorLikeFill(out3, 0.5); // dummy label
    tensor* loss = reduce_sum(pow(sub(y, out3), 2));

    // *** Zero-out grads ***
    zero_grads();

    // *** Backward ***
    save_num_uses(loss);
    loss->backward(loss);

    generate_test(loss);

    // *** Optim Step ***
    sgd(LR);

    graphviz(loss);

    return COPY_FROM_DEVICE(loss)->data[0];
}


int main(void) {
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
    add_param(w1);

    tensor* w2 = Tensor(D, O);
    set_name(w2, "w2"); print(w2);
    add_param(w2);

    // *** Train ***
    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        float loss = train_step(x, w1, w2);
        cout << "\nep: " << ep_idx << "; loss: " << loss << endl;
    }

    return 0;
}
