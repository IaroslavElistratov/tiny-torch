#include <iostream> // todo: use C only
using namespace std;


#define DEVICE CUDA

#include "../nn.h"
#include "../tensor.cpp"
#include "../ops.cpp"
#include "../composite_ops.cpp"
#include "../cifar10.cpp"
#include "../print.cpp"
#include "../optim.cpp"
#include "../codegen.cpp"


#define NUM_EP 1 // 20
#define LR 0.001 // torch tutorial
#define DEBUG  1



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



tensor* forward(tensor* input) {

    tensor* conv1 = batched_conv(input, get_param("kernel1"), get_param("bias_kernel1"));
    set_name(conv1, "conv1");
    tensor* relu1 = relu(conv1);
    set_name(relu1, "relu1");
    tensor* mp1 = batched_maxpool(relu1);
    set_name(mp1, "mp1");

    tensor* conv2 = batched_conv(mp1, get_param("kernel2"), get_param("bias_kernel2"));
    set_name(conv2, "conv2");
    tensor* relu2 = relu(conv2);
    set_name(relu2, "relu2");
    tensor* mp2 = batched_maxpool(relu2);
    set_name(mp2, "mp2");

    tensor* flat = batched_flatten(mp2);
    set_name(flat, "flat");

    tensor* mm1 = matmul(flat, get_param("w1"));
    set_name(mm1, "mm1");
    tensor* relu3 = relu(mm1);
    set_name(relu3, "relu3");

    tensor* mm2 = matmul(relu3, get_param("w2"));
    set_name(mm2, "mm2");
    tensor* relu4 = relu(mm2);
    set_name(relu4, "relu4");

    tensor* mm3 = matmul(relu4, get_param("w3"));
    set_name(mm3, "mm3");

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


tensor* train_step(cifar10* data) {

    // *** Net ***
    tensor* logits = forward(data->input);

    // *** Loss fn ***
    tensor* log_probs = log_softmax(logits);
    tensor* loss = NLL(log_probs, data->label);

    // *** Zero-out grads ***
    zero_grads();

    // *** Backward ***
    save_num_uses(loss);
    loss->backward(loss);

    // must call generate test BEFORE param update, otherwise asserts
    // on runtime values don't make sense -- bc SGD mutates weights inplace
    generate_test(loss);

    // *** Optim Step ***
    sgd(LR);

    // todo-high: need smt like torch.detach?
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
    add_param(kernel1);

    tensor* bias_kernel1 = Tensor(F, 1);
    set_name(bias_kernel1, "bias_kernel1");
    add_param(bias_kernel1);

    tensor* kernel2 = Tensor(F, F, HH2, WW2);
    set_name(kernel2, "kernel2");
    add_param(kernel2);

    tensor* bias_kernel2 = Tensor(F, 1);
    set_name(bias_kernel2, "bias_kernel2");
    add_param(bias_kernel2);

    tensor* w1 = Tensor(96, 64);
    set_name(w1, "w1");
    add_param(w1);

    tensor* w2 = Tensor(64, 32);
    set_name(w2, "w2");
    add_param(w2);

    tensor* w3 = Tensor(32, 10);
    set_name(w3, "w3");
    add_param(w3);

    // tensor* b1 = Tensor(128, 1);
    // set_name(b1, "b1");

    // tensor* b2 = Tensor(64, 1);
    // set_name(b2, "b2");

    // tensor* b3 = Tensor(10, 1);
    // set_name(b3, "b3");

    // todo-low: change add_param to accept array of all prams "add_param({kernel1, bias_kernel1, kernel2, bias_kernel2, w1, w2, w3})"?


    // todo: somehow if having prints in-between initialization of later weights print produces differently
    // initialized later tensors (via print->copy_from_cuda) even though the constructor called from
    // copy_from_cuda does not explicitly advacne the RNG state (does not call GetRandomFloat).
    // question-now: is it bc of "copy_from_cuda -> malloc(size)" ?
    // So moved prints and "get_cifar10" after initializing the tensors -- this way tensors will get initialized to
    // the same values regardless of wether there are prints or not

    // lprint(kernel1);
    // lprint(bias_kernel1);
    // lprint(kernel2);
    // lprint(bias_kernel2);
    // lprint(w1);
    // lprint(w2);
    // lprint(w3);

    cifar10* data = get_cifar10();

    // *** Train ***

    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        // passes loss sanity check -- 10 classes, if model is random (predicting each cls equally)
        // log(0.1) = -2.3
        tensor* loss = train_step(data);
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
