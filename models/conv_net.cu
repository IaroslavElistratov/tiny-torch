#include <iostream> // todo: use C only
using namespace std;


#define DEVICE CUDA

#include "../nn.h"
#include "../tensor.cpp"
#include "../ops.cpp"
#include "../composite_ops.cpp"
#include "../cifar10.cpp"
#include "../print.cpp"
#include "../codegen.cpp"


#define NUM_EP 1 // 20
#define LR 0.001 // torch tutorial
#define DEBUG  1



// todo-low: mv to optim.cpp;
// todo: to avoid data transfer impl sgd as composite_op: sub_(w->grad, mul(w->grad, TensorLikeFill(w, 0.2)), w->grad)
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



tensor* forward(tensor* input, state* params) {

    // *** Net ***

    tensor* conv1 = batched_conv(input, params->kernel1, params->bias_kernel1);
    set_name(conv1, "conv1"); // sprint(conv1);
    tensor* relu1 = relu(conv1);
    set_name(relu1, "relu1"); // sprint(relu1);
    tensor* mp1 = batched_maxpool(relu1);
    set_name(mp1, "mp1"); // sprint(mp1);

    tensor* conv2 = batched_conv(mp1, params->kernel2, params->bias_kernel2);
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
    set_name(mm2, "mm2"); // sprint(mm2);
    tensor* relu4 = relu(mm2);
    set_name(relu4, "relu4"); // sprint(relu4);

    tensor* mm3 = matmul(relu4, params->w3);
    set_name(mm3, "mm3"); // sprint(mm3);

    return mm3;
}



void accuracy(tensor* log_probs, tensor* label){
    // pred idxs
    tensor* probs = exp(log_probs);
    set_name(probs, "probs");
    tensor* pred = batched_reduce_max(probs)->scratch_space[0];
    set_name(pred, "pred");

    pred = COPY_FROM_DEVICE(pred);
    label = COPY_FROM_DEVICE(label);

    // it's not a binary elementwise but same checks
    // assert_binary_elementwise(pred, label);

    int B = pred->shape[0];
    float correct = 0.0;
    for (int b=0; b<B; b++){
        (pred->data[b] == label->data[b]) ? correct++ : 0;
    }
    printf("accuracy: %f (%i/%i)\n", correct/B, (int)correct, B);
};

tensor* train_step(cifar10* data, state* params) {

    tensor* logits = forward(data->input, params);
    tensor* log_probs = log_softmax(logits);
    tensor* loss = NLL(log_probs, data->label);

    // *** Zero-out grads ***
    params->kernel1->grad = NULL;
    params->bias_kernel1->grad = NULL;
    params->kernel2->grad = NULL;
    params->bias_kernel2->grad = NULL;
    params->w1->grad = NULL;
    params->w2->grad = NULL;
    params->w3->grad = NULL;

    loss->num_uses = 0;
    save_num_uses(loss);

    // *** Backward ***
    loss->backward(loss);

    // must call generate test BEFORE param update, otherwise asserts
    // on runtime values don't make sense -- bc SGD mutates weights inplace
    generate_test(loss, params);

    // *** Optim Step ***
    sgd(params->kernel1);
    sgd(params->bias_kernel1);
    sgd(params->kernel2);
    sgd(params->bias_kernel2);
    sgd(params->w1);
    sgd(params->w2);
    sgd(params->w3);

    accuracy(log_probs, data->label);
    return loss;
}

// todo-low: when define weights (w1, w2, w3) in forward, can use runtime shapes to create these weights.
// But when creating weights in main (in main fn), needed to hardcode these shapes, copying from train_step.
// w1 = Tensor(flat->shape[1], 32);
// w2 = Tensor(relu3->shape[1], 16);
// w3 = Tensor(relu4->shape[1], 10);
int main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);
    set_backend_device();

    fclose(fopen("./generated/log.txt", "w"));


    // *** Init ***

    int C = 3;
    int F = 6;
    int HH1 = 7;
    int WW1 = 7;

    int HH2 = 6;
    int WW2 = 6;

    tensor* kernel1 = Tensor(F, C, HH1, WW1);
    set_name(kernel1, "kernel1");
    tensor* kernel2 = Tensor(F, F, HH2, WW2);
    set_name(kernel2, "kernel2");

    tensor* w1 = Tensor(96, 64);
    set_name(w1, "w1");
    tensor* w2 = Tensor(64, 32);
    set_name(w2, "w2");
    tensor* w3 = Tensor(32, 10);
    set_name(w3, "w3");

    tensor* bias_kernel1 = Tensor(F, 1);
    set_name(bias_kernel1, "bias_kernel1");

    tensor* bias_kernel2 = Tensor(F, 1);
    set_name(bias_kernel2, "bias_kernel2");

    // tensor* b1 = Tensor(128, 1);
    // set_name(b1, "b1");

    // tensor* b2 = Tensor(64, 1);
    // set_name(b2, "b2");

    // tensor* b3 = Tensor(10, 1);
    // set_name(b3, "b3");

    state params = {kernel1, bias_kernel1, kernel2, bias_kernel2, w1, w2, w3};



    // todo: somehow if having prints in-between initialization of later weights print produces differently
    // initialized later tensors (via print->copy_from_cuda) even though the constructor called from
    // copy_from_cuda does not explicitly advacne the RNG state (does not call GetRandomFloat).
    // question-now: is it bc of "copy_from_cuda -> malloc(size)" ?
    // So moved prints and "get_cifar10" after initializing the tensors -- this way tensors will get initialized to
    // the same values regardless of wether there are prints or not

    lprint(kernel1);
    lprint(bias_kernel1);
    lprint(kernel2);
    lprint(bias_kernel2);
    lprint(w1);
    lprint(w2);
    lprint(w3);

    cifar10* data = get_cifar10();

    // *** Train ***

    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        // passes loss sanity check -- 10 classes, if model is random (predicting each cls equally)
        // log(0.1) = -2.3
        tensor* loss = train_step(data, &params);
        if (ep_idx==0){
            graphviz(loss);
        }
        printf("ep: %i; loss: %f;\n\n", ep_idx, COPY_FROM_DEVICE(loss)->data[0]);
    }

    // lprint(params.w3->grad);
    // lprint(params.w2->grad);
    // lprint(params.w1->grad);
    // lprint(params.kernel2->grad);
    // lprint(params.bias_kernel2->grad);
    // lprint(params.kernel1->grad);
    // lprint(params.bias_kernel1->grad);

    // lprint(kernel1);
    // lprint(kernel2);
    // lprint(w1);
    // lprint(w2);
    // lprint(w3);
    return 0;
}
