#include <iostream> // todo: use C only
using namespace std;

#define DEVICE CUDA


#include "../nn.h"
#include "../tensor.cpp"
#include "../ops.cpp"
#include "../cifar10.cpp"
#include "../print.cpp"


#define NUM_EP 1
#define LR 0.02
#define DEBUG  1



// todo-low: mv to optim.cpp
void sgd(tensor* w) {
    tensor* w_local = COPY_FROM_DEVICE(w);
    tensor* w_grad_local = COPY_FROM_DEVICE(w->grad);

    for (int i=0; i<w->size; i++)
        w_local->data[i] -= w_grad_local->data[i] * LR;

    COPY_TO_DEVICE(w_local);
    // todo: memory leak
    w->data = w_local->data;
}


/*
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def forward(self, x):
    x = self.conv1(x)   // Conv2d(in_channels=3, out_channels=6, kernel_size=5)
    x = F.relu(x)
    x = self.pool(x)    // MaxPool2d(2, 2)

    x = self.conv2(x)   // Conv2d(in_channels=6, out_channels=16, kernel_size=5)
    x = F.relu(x)
    x = self.pool(x)

    x = torch.flatten(x, 1) // flatten all dimensions except batch

    x = self.fc1(x)     // Linear(in_features=16 * 5 * 5, out_features=120)
    x = F.relu(x)

    x = self.fc2(x)     // Linear(in_features=120, out_features=84)
    x = F.relu(x)

    x = self.fc3(x)     // Linear(in_features=84, out_features=10)
    return x
*/


struct state
{
    tensor* kernel1;
    tensor* kernel2;
    tensor* w1;
    tensor* w2;
    tensor* w3;
};


tensor* forward(tensor* input, state* params) {

    // *** Net ***

    tensor* conv1 = batched_conv(input, params->kernel1);
    set_name(conv1, "conv1"); // sprint(conv1);
    tensor* relu1 = relu(conv1);
    set_name(relu1, "relu1"); // sprint(relu1);
    tensor* mp1 = batched_maxpool(relu1);
    set_name(mp1, "mp1"); // sprint(mp1);

    tensor* conv2 = batched_conv(mp1, params->kernel2);
    set_name(conv2, "conv2"); // sprint(conv2);
    tensor* relu2 = relu(conv2);
    set_name(relu2, "relu2"); // sprint(relu2);
    tensor* mp2 = batched_maxpool(relu2);
    set_name(mp2, "mp2"); // sprint(mp2);

    tensor* flat = batched_flatten(mp2);
    set_name(flat, "flat"); // sprint(flat);

    tensor* mm1 = matmul(flat, params->w1);
    set_name(mm1, "mm1"); // sprint(mm1);
    tensor* relu3 = relu(mm1);
    set_name(relu3, "relu3"); // sprint(relu3);

    tensor* mm2 = matmul(relu3, params->w2);
    set_name(mm2, "mm2"); // sprint(mm1);
    tensor* relu4 = relu(mm2);
    set_name(relu4, "relu4"); // sprint(relu4);

    tensor* mm3 = matmul(relu4, params->w3);
    set_name(mm3, "mm3"); // sprint(mm3);

    // *** Softmax ***

    // min-max trick for numerical stability, python: "mm3 -= np.max(mm3, axis=1, keepdims=True)"
    int n_repeats = mm3->shape[1];
    tensor* maxes = repeat(batched_reduce_max(mm3), n_repeats);
    set_name(maxes, "maxes"); // sprint(maxes);
    tensor* su = sub(mm3, maxes);
    set_name(su, "su"); // sprint(su);

    tensor* ex = exp(su);                       // (B, ?)
    set_name(ex, "ex"); // sprint(ex);
    tensor* denom = batched_reduce_sum(ex);     // (B, 1)
    set_name(denom, "denom"); // sprint(denom);
    n_repeats = ex->shape[1];
    tensor* denom_broadcasted = repeat(denom, n_repeats);
    set_name(denom_broadcasted, "denom_broadcasted"); // sprint(denom_broadcasted);
    tensor* sm = div(ex, denom_broadcasted);    // (B, ?)
    set_name(sm, "sm"); // print(sm);

    return sm;
}

tensor* NLL(tensor* probs, tensor* label){
    int B = label->shape[0];
    set_name(label, "label"); // print(data->label);
    tensor* se = select(probs, label);   // (B, 1)
    set_name(se, "se"); // sprint(se);
    tensor* lg = log(se);               // (B, 1)
    set_name(lg, "lg"); // sprint(lg);
    tensor* lgsum = reduce_sum(lg);         // (, )
    set_name(lgsum, "lgsum"); // sprint(lgsum);
    tensor* nll = neg(lgsum);               // (, )
    set_name(nll, "nll"); // print(nll);
    // divide by the batch size
    tensor* nll_normalized = div(nll, TensorScalarFill(B));               // (, )
    set_name(nll_normalized, "nll_normalized"); // print(nll_normalized);
    return nll_normalized;
}

tensor* train_step(cifar10* data, state* params) {

    tensor* probs = forward(data->input, params);
    tensor* loss = NLL(probs, data->label);

    // *** Zero-out grads ***
    params->kernel1->grad = NULL;
    params->kernel2->grad = NULL;
    params->w1->grad = NULL;
    params->w2->grad = NULL;
    params->w3->grad = NULL;

    // *** Backward ***
    loss->backward(loss);

    // *** Optim Step ***
    sgd(params->kernel1);
    sgd(params->kernel2);
    sgd(params->w1);
    sgd(params->w2);
    sgd(params->w3);

    return COPY_FROM_DEVICE(loss);
}

int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);
    set_backend_device();

    int C = 3;
    int F = 6;
    int HH1 = 2;
    int WW1 = 2;

    int HH2 = 2;
    int WW2 = 2;

    cifar10* data = get_cifar10();

    // *** Init ***

    tensor* kernel1 = Tensor(F, C, HH1, WW1);
    set_name(kernel1, "kernel1"), sprint(kernel1);

    tensor* kernel2 = Tensor(F, F, HH2, WW2);
    set_name(kernel2, "kernel2"), sprint(kernel2);

    // todo-low: when define weights (w1, w2, w3) in forward, can use runtime shapes to create these weights.
    // But when creating weights in main (in main fn), needed to hardcode these shapes, copying from train_step.
    // w1 = Tensor(flat->shape[1], 32);
    // w2 = Tensor(relu3->shape[1], 16);
    // w3 = Tensor(relu4->shape[1], 10);

    tensor* w1 = Tensor(24, 32);
    set_name(w1, "w1"), sprint(w1);

    tensor* w2 = Tensor(32, 16);
    set_name(w2, "w2"), sprint(w2);

    tensor* w3 = Tensor(16, 10);
    set_name(w3, "w3"), sprint(w3);

    state params = {kernel1, kernel2, w1, w2, w3};

    // *** Train ***

    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        // passes loss sanity check -- 10 classes, if model is random (predicting each cls equally)
        // log(0.1) = -2.3
        tensor* loss = train_step(data, &params);
        if (ep_idx==0)
            graphviz(loss);
        printf("ep: %i; loss: %f;\n", ep_idx, loss->data[0]);
    }

    print(params.kernel1->grad);
    print(kernel1);
    print(kernel2);
    print(w1);
    print(w2);
    print(w3);
    return 0;
}
