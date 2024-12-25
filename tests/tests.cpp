#include <iostream> // todo: use C only
using namespace std;

#define DEVICE CPU

// #include "nn.h"
#include "tensor.cpp"
#include "ops.cpp"
#include "conv.cpp"
#include "utils.cpp"
#include "../print.cpp"


// todo: add tests -- https://github.com/tensorflow/tensorflow/commit/6f4a0e96d853d1d8fe05a8dd8f7ba0cd0fb0e79b#diff-65511a88d2951377144d77a2de94c0f597c4664189d3d5ac730e653560b64f31R259-R342
//  - https://github.com/rui314/8cc/blob/b480958396f159d3794f0d4883172b21438a8597/test/typeof.c#L23

/*
    out2[1] = 0.123;
    print(ReluBackward(out2, N*D), N, D);
*/

/* Transpose tests
    float arr[] = {0., 1., 2., 3.};
    cout << "\narr: ";
    Print(arr, 2, 2);
    float* arr_T = Transpose(arr, 2, 2);
    cout << "\ntransposed: ";
    Print(arr_T, 2, 2);

    float arr[] = {0., 1., 2.,
                    3., 4., 5.};
    cout << "\narr: ";
    Print(arr, 2, 3);
    float* arr_T = Transpose(arr, 2, 3);
    cout << "\ntransposed: ";
    Print(arr_T, 3, 2);
*/

/* pow tests
    cout << "pow(2, 2): " << pow(2, 2) << endl;
    cout << "pow(2, 2): " << pow(4, 8) << endl;
*/

/* autograd
    tensor* a = Tensor(2, 2);
    a->backward(a);
    // >>> [autograd engine] Error: tensor has no grad_fn
*/



int test_net() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int B = 2;
    int C = 3;
    int H = 32;
    int W = 32;

    int F = 5;
    int HH = 2;
    int WW = 2;

    // *** Init ***
    tensor* input = Tensor(B, C, H, W);
    set_name(input, "input"); sprint(input);

    // cifar10* data = get_cifar10();
    // tensor* input = data->input;

    tensor* kernel = Tensor(F, C, HH, WW);
    set_name(kernel, "kernel"); sprint(kernel);
    tensor* kernel2 = Tensor(F, F, HH, WW);
    set_name(kernel2, "kernel2"); sprint(kernel2);


    // *** Net ***
    tensor* out_conv1 = batched_conv(input, kernel);
    set_name(out_conv1, "out_conv1"); sprint(out_conv1);
    tensor* out_relu1 = relu(out_conv1);
    set_name(out_relu1, "out_relu1"); sprint(out_relu1);
    tensor* out_mp1 = batched_maxpool(out_relu1);
    set_name(out_mp1, "out_mp1"); sprint(out_mp1);

    tensor* out_conv2 = batched_conv(out_mp1, kernel2);
    set_name(out_conv2, "out_conv2"); sprint(out_conv2);
    tensor* out_relu2 = relu(out_conv2);
    set_name(out_relu2, "out_relu2"); sprint(out_relu2);
    tensor* out_mp2 = batched_maxpool(out_relu2);
    set_name(out_mp2, "out_mp2"); sprint(out_mp2);

    tensor* out_flat = batched_flatten(out_mp2);
    set_name(out_flat, "out_flat"); sprint(out_flat);

    tensor* w1 = Tensor(out_flat->shape[1], 32);
    set_name(w1, "w1"); print(w1);
    tensor* out_mm1 = matmul(out_flat, w1);
    set_name(out_mm1, "out_mm1"); sprint(out_mm1);
    tensor* out_relu3 = relu(out_mm1);
    set_name(out_relu3, "out_relu3"); sprint(out_relu3);

    tensor* w2 = Tensor(out_relu3->shape[1], 16);
    set_name(w2, "w2"); print(w2);
    tensor* out_mm2 = matmul(out_relu3, w2);
    set_name(out_mm2, "out_mm2"); sprint(out_mm1);
    tensor* out_relu4 = relu(out_mm2);
    set_name(out_relu4, "out_relu4"); sprint(out_relu4);

    tensor* w3 = Tensor(out_relu4->shape[1], 10);
    set_name(w3, "w3"); print(w3);
    tensor* out = matmul(out_relu4, w3);
    set_name(out, "out"); print(out);

    out->backward(out);
    graphviz(out);


    return 0;
}


int test_select() {
    srand(123);
    tensor* a = Tensor(4, 2);
    set_name(a, "a"), print(a);
    tensor* idx = Tensor(4, 1);
    idx->data[0] = 1.0;
    idx->data[1] = 0.0;
    idx->data[2] = 0.0;
    idx->data[3] = 1.0;

    tensor* out = select(a, idx);

    tensor* w = Tensor(1, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    print(out2);

    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);

}

// test_max
int test_max() {
    srand(123);
    tensor* a = Tensor(4, 3);
    set_name(a, "a"), print(a);

    tensor* out = batched_max(a);
    set_name(out, "out"), print(out);

    tensor* w = Tensor(1, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    set_name(out2, "out2"), print(out2);

    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);
    // graphviz(out2);

    return 0;
}

int test_batched_reduce() {
    srand(123);
    tensor* a = Tensor(4, 10);
    set_name(a, "a"), print(a);

    tensor* out = batched_reduce_sum(a);
    set_name(out, "out"), print(out);

    tensor* w = Tensor(1, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    set_name(out2, "out2"), print(out2);



    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);
    return 0;
}

int test_exp() {
    srand(123);
    tensor* a = Tensor(4, 10);
    set_name(a, "a"), print(a);

    tensor* out = exp(a);
    set_name(out, "out"), print(out);

    tensor* w = Tensor(10, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    set_name(out2, "out2"), print(out2);



    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);
    graphviz(out2);

    return 0;
}

int test_log() {
    srand(123);
    tensor* a = Tensor(4, 10);
    set_name(a, "a"), print(a);

    tensor* out = log(a);
    set_name(out, "out"), print(out);

    tensor* w = Tensor(10, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    set_name(out2, "out2"), print(out2);



    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);
    graphviz(out2);

    return 0;
}



int test_repeat() {
    srand(123);
    tensor* a = Tensor(4, 1);
    set_name(a, "a"), print(a);

    tensor* out = repeat(a, 3);
    set_name(out, "out"), print(out);

    tensor* w = Tensor(3, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    set_name(out2, "out2"), print(out2);



    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);
    graphviz(out2);

    return 0;
}



int test_neg() {
    srand(123);
    tensor* a = Tensor(4, 1);
    set_name(a, "a"), print(a);

    tensor* out = neg(a);
    set_name(out, "out"), print(out);

    tensor* w = Tensor(1, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    set_name(out2, "out2"), print(out2);



    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);
    graphviz(out2);

    return 0;
}




int test_div() {
    srand(123);
    tensor* a = Tensor(4, 3);
    set_name(a, "a"), print(a);

    tensor* b = Tensor(4, 3);
    set_name(b, "b"), print(b);

    tensor* out = div(a, b);
    set_name(out, "out"), print(out);

    tensor* w = Tensor(3, 5);
    set_name(w, "w"), print(w);

    tensor* out2 = matmul(out, w);
    set_name(out2, "out2"), print(out2);


    out2->backward(out2);
    set_name(a->grad, "a_grad"), print(a->grad);
    set_name(b->grad, "b_grad"), print(b->grad);
    graphviz(out2);

    return 0;
}

// int main(){
//     srand(123);
//     tensor* a = Tensor(4, 3);
//     set_name(a, "a"), print(a);

//     tensor* b = Tensor(4, 3);
//     set_name(b, "b"), print(b);

//     tensor* c = pow_k(b, 2);
//     set_name(c, "c"), print(c);

//     tensor* d = div_k(a, c);
//     set_name(d, "d"), print(d);

//     tensor* out = neg_k(d);
//     set_name(out, "out"), print(out);
// }



// testing maxpool and its bwd
int test_maxpool() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int B = 2;
    int C = 3;
    int H = 4;
    int W = 4;

    int F = 5;
    int HH = 2;
    int WW = 2;

    // // *** Init ***
    // tensor* x = Tensor(C, H, W);
    // set_name(x, "x"); print(x);

    // tensor* out = maxpool_k(x);
    // set_name(out, "out"); print(out);

    // // printf("\n\n\n\n\n\n\n\n\n\n\n\n");

    // // temporary:
    // out->inputs[0] = x;

    // tensor* upstream = TensorLikeFill(out, 1.0);
    // tensor* grad_x = bwd_maxpool_k(upstream, out);
    // set_name(grad_x, "grad_x"); print(grad_x);



    // // *** Init ***
    // tensor* x = Tensor(B, C, H, W);
    // set_name(x, "x"); print(x);

    // tensor* out = batched_maxpool_k(x);
    // set_name(out, "out"); print(out);

    // // printf("\n\n\n\n\n\n\n\n\n\n\n\n");

    // // temporary:
    // out->inputs[0] = x;

    // tensor* upstream = TensorLikeFill(out, 1.0);
    // tensor* grad_x = bwd_batched_maxpool_k(upstream, out);
    // set_name(grad_x, "grad_x"); print(grad_x);


    // *** Init ***
    tensor* x = Tensor(B, C, H, W);
    set_name(x, "x"); print(x);

    tensor* out = batched_maxpool(x);
    set_name(out, "out"); print(out);

    out->backward(out);

    return 0;
}


int test_flatten() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int B = 2;
    int C = 2;
    int H = 8;
    int W = 8;

    int F = 2;
    int HH = 2;
    int WW = 2;

    // *** Init ***
    tensor* input = Tensor(B, C, H, W);
    set_name(input, "input"); print(input);

    tensor* kernel = Tensor(F, C, HH, WW);
    set_name(kernel, "kernel"); print(kernel);

    tensor* out_conv1 = batched_conv(input, kernel);
    set_name(out_conv1, "out_conv1"); sprint(out_conv1);

    tensor* out_flat = batched_flatten(out_conv1);
    set_name(out_flat, "out_flat"); print(out_flat);


    out_flat->backward(out_flat);

    return 0;
}



int test_conv() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int B = 2;
    int C = 3;
    int H = 4;
    int W = 4;

    int F = 5;
    int HH = 2;
    int WW = 2;

    // *** Init ***
    tensor* input = Tensor(B, C, H, W);
    set_name(input, "input"); print(input);

    tensor* kernel = Tensor(F, C, HH, WW);
    set_name(kernel, "kernel"); print(kernel);

    tensor* out_conv1 = batched_conv(input, kernel);
    set_name(out_conv1, "out_conv1"); print(out_conv1);
    out_conv1->backward(out_conv1);

    return 0;
}


int test_conv() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int C = 3;
    int H = 4;
    int W = 4;

    int F = 5;
    int HH = 2;
    int WW = 2;

    // *** Init ***
    tensor* input = Tensor(C, H, W);
    set_name(input, "input"); print(input);

    tensor* kernel = Tensor(F, C, HH, WW);
    set_name(kernel, "kernel"); print(kernel);

    tensor* out = conv_k(input, kernel);
    set_name(out, "out"); print(out);

    printf("\n\n\n\n\n\n\n\n\n\n\n\n");

    // temporary:
    out->inputs[0] = input;
    out->inputs[1] = kernel;

    tensor* upstream = TensorLikeFill(out, 1.0);
    bwd_conv_k(upstream, out);
    tensor* grad_kernels = kernel->grad; // set by bwd_conv_k
    set_name(grad_kernels, "grad_kernels"); print(grad_kernels);

    return 0;
}



int test_bmm() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int B = 8;
    int N = 3;
    int M = 2;
    int D = 4;


    tensor* input = Tensor(B, N, M);
    set_name(input, "input"); print(input);

    tensor* weight = Tensor(B, M, D);
    set_name(weight, "weight"); print(weight);

    // (B, N, D)
    tensor* out = batched_matmul(input, weight);
    set_name(out, "out"); print(out);

    graphviz(out);

    return 0;
}



int test_bmm_k() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int B = 8;
    int N = 3;
    int M = 2;
    int D = 4;


    tensor* input = Tensor(B, N, M);
    set_name(input, "input"); print(input);

    tensor* weight = Tensor(B, M, D);
    set_name(weight, "weight"); print(weight);

    // (B, N, D)
    tensor* out = batched_matmul_k(input, weight);
    set_name(out, "out"); print(out);

    return 0;
}

int test_bt_k() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int B = 8;
    int N = 3;
    int M = 2;
    int D = 4;


    tensor* input = Tensor(B, N, M);
    set_name(input, "input"); print(input);

    tensor* transposed = batched_transpose_k(input);
    set_name(transposed, "transposed"); print(transposed);

    return 0;
}




int test_indexing() {
    srand(123);

    tensor* x = Tensor(3, 7);
    set_name(x, "orig. x"); print(x);

    tensor* x_slice = slice(x, "1:3, 3:6");
    set_name(x_slice, "x_slice");
    print(x_slice);

    tensor* x_view = view(x, "1:3, 3:6");
    set_name(x_view, "x_view");
    print(x_view);

    cout << "\n19th element of x:" << endl;
    cout << x->data[at(x, 19)] << endl;

    tensor* y = Tensor(4, 3, 7);
    set_name(y, "orig. y");
    print(y);

    tensor* y_slice = slice(y, "2:4, 1:3, 3:6");
    set_name(y_slice, "y_slice");
    print(y_slice);

    tensor* y_view = view(y, "2:4, 1:3, 3:6");
    set_name(y_view, "y_view");
    print(y_view);

    cout << "\n54th element of y:" << endl;
    cout << y->data[at(y, 54)] << endl;

    return 0;
}


// test_at
//    (3, 7).at(20) = 
//    y_idx = 20 / 7 = 2
//    z_idx = 20 % 7 = 6

