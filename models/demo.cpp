#include <iostream> // todo: use C only
using namespace std;

#include "../tensor.cpp"
#include "../ops.cpp"
#include "../conv.cpp"
#include "../utils.cpp"
#include "../cifar10.cpp"
#include "../print.cpp"

#define NUM_EP 10
#define LR 0.02


void sgd(tensor* w) {
    for (int i=0; i<w->size; i++)
        w->data[i] -= w->grad->data[i] * LR;
}

struct state {
    tensor* kernel1;
    tensor* kernel2;
    tensor* w1;
    tensor* w2;
    tensor* w3;
};

void zero_grads(state* params){
    params->kernel1->grad = NULL;
    params->kernel2->grad = NULL;
    params->w1->grad = NULL;
    params->w2->grad = NULL;
    params->w3->grad = NULL;
}

tensor* forward(tensor* input, state* params) {

    // *** Net ***

    tensor* conv1 = batched_conv(input, params->kernel1);
    tensor* relu1 = relu(conv1);
    tensor* mp1 = batched_maxpool(relu1);

    tensor* conv2 = batched_conv(mp1, params->kernel2);
    tensor* relu2 = relu(conv2);
    tensor* mp2 = batched_maxpool(relu2);

    tensor* flat = batched_flatten(mp2);

    tensor* mm1 = matmul(flat, params->w1);
    tensor* relu3 = relu(mm1);

    tensor* mm2 = matmul(relu3, params->w2);
    tensor* relu4 = relu(mm2);

    tensor* mm3 = matmul(relu4, params->w3);

    // *** Softmax ***

    //  min-max trick for numerical stability
    int n_repeats = mm3->shape[1];
    tensor* maxes = repeat(batched_max(mm3), n_repeats);
    tensor* su = sub(mm3, maxes);

    tensor* ex = exp(su);
    tensor* denom = batched_reduce_sum(ex);
    n_repeats = ex->shape[1];
    tensor* denom_broadcasted = repeat(denom, n_repeats);
    tensor* sm = div(ex, denom_broadcasted);

    return sm;
}

tensor* NLL(tensor* probs, tensor* label){
    int B = label->shape[0];
    tensor* se = select(probs, label);
    tensor* lg = log(se);
    tensor* lgsum = reduce_sum(lg);
    tensor* nll = neg(lgsum);
    tensor* loss = div(nll, TensorScalarFill(B));
    set_name(loss, "loss");
    return loss;
}

tensor* train_step(cifar10* data, state* params) {

    tensor* probs = forward(data->input, params);
    tensor* loss = NLL(probs, data->label);
    zero_grads(params);

    // backward
    loss->backward(loss);

    // optim step
    sgd(params->kernel1);
    sgd(params->kernel2);
    sgd(params->w1);
    sgd(params->w2);
    sgd(params->w3);

    return loss;
}

int main() {
    // random num generator init, must be called once
    srand(123);

    int C = 3;
    int F = 6;
    int HH1 = 2;
    int WW1 = 2;
    int HH2 = 2;
    int WW2 = 2;

    cifar10* data = get_cifar10();

    // *** Init ***

    tensor* kernel1 = Tensor4d(F, C, HH1, WW1);
    set_name(kernel1, "kernel1"), sprint_4d(kernel1);

    tensor* kernel2 = Tensor4d(F, F, HH2, WW2);
    set_name(kernel2, "kernel2"), sprint_4d(kernel2);

    tensor* w1 = Tensor(24, 32);
    set_name(w1, "w1"), sprint_2d(w1);

    tensor* w2 = Tensor(32, 16);
    set_name(w2, "w2"), sprint_2d(w2);

    tensor* w3 = Tensor(16, 10);
    set_name(w3, "w3"), sprint_2d(w3);

    state params = {kernel1, kernel2, w1, w2, w3};

    // *** Train ***

    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        tensor* loss = train_step(data, &params);
        if (ep_idx==0)
            graphviz(loss);
        printf("ep: %i; loss: %f;\n", ep_idx, loss->data[0]);
    }

    print_4d(kernel1);
    print_4d(kernel2);
    print_2d(w1);
    print_2d(w2);
    print_2d(w3);
    return 0;
}
