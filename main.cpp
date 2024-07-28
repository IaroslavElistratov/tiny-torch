#include <iostream> // todo: use C only
// #include <stdlib.h>
// #include <iomanip> // for  input-output manipulation
using namespace std;

// #include "nn.h"
#include "tensor.cpp"
#include "ops.cpp"
#include "utils.cpp"


# define NUM_EP 5
# define LR 0.02
#define DEBUG  1

#ifdef DEBUG
#define print(f_p, msg) _print(f_p, msg)
#else
#define print(f_p, msg)
#endif

void sgd(tensor* w, tensor* grad_w)
{
    for (int i=0; i<w->size; i++) {
        w->data[i] -= grad_w->data[i] * LR;
    }
}

float train_step(tensor* x, tensor* w1, tensor* w2)
{
    // *** FWD ***

    // x(N, M) @ w1(M, D) = out1(N, D)
    tensor* out1 = Matmul(x, w1);
    print(out1, "matmul_1");

    // out2(N, D)
    tensor* out2 = Relu(out1);
    print(out2, "relu");

    // out2(N, D) @ w2(D, O) = out3(N, O)
    tensor* out3 = Matmul(out2, w2);
    print(out3, "matmul_2");

    // loss
    tensor* y = TensorLikeFill(out3, 0.5); // dummy label
    float loss = mse(out3, y);
    // cout << "loss :" << loss << endl;

    // *** BWD ***

    // note: below derivatives try to follow this order -- local * upstream
    // question-now: non deterministic order of args?
    //  - dL_dpow, dL_dneg, dL_dw2, dL_out2

    float dL_dL = 1.0;
    // float dL_dsum = 1.0;

    // mse bwd

    // "reduce_sum" bwd
    tensor* dL_dpow = TensorLikeFill(y, dL_dL);

    // "**2" bwd
    // todo: avoid recomputing this during backward
    tensor* diff = TensorLike(y);
    for (int i=0; i<diff->size; i++) {
        diff->data[i] = y->data[i] - out3->data[i];
    }
    tensor* local_dpow = ElementwiseMul(TensorLikeFill(diff, 2.0), diff);
    tensor* dL_dneg = ElementwiseMul(local_dpow, dL_dpow);

    // "-" bwd
    tensor* dL_dout3 = ElementwiseMul(TensorLikeFill(dL_dneg, -1.0), dL_dneg);

    // matmul 2 bwd

    // dL_dout3(N, O)
    // out2(N, D)
    // w2(D, O)
    //   so: out2.T(D, N) @ dL_dout3(N, O) = dL_dw2(D, O)
    //   I went from the ouput shape (D, O) -- this has to be the same bc
    tensor* dL_dw2 = Matmul(Transpose(out2), dL_dout3);
    _print(dL_dw2, "dL_dw2");

    // out2(N, D)
    //   so: dL_dout3(N, O) @ w2.T(O, D) = dL_out2(N, D)
    tensor* dL_out2 = Matmul(dL_dout3, Transpose(w2));

    // relu bwd

    // note: this is kind of gradient checkpointing;
    //  ofc can avoid by making original relu ouput the mask
    //  -- but I found recomputing is a bit cleaner to explain
    tensor* relu_mask = GetReluMask(out1);
    tensor* dL_dout1 = ElementwiseMul(relu_mask, dL_out2);

    // matmul 1 bwd

    // dL_dout1(N, D)
    // x(N, M)
    // w1(M, D)
    //   so: x.T(M, N) @ dL_dout1(N, D) = dL_dw1(M, D)
    tensor* dL_dw1 = Matmul(Transpose(x), dL_dout1);
    _print(dL_dw1, "dL_dw1");

    // *** Optim Step ***
    sgd(w1, dL_dw1);
    sgd(w2, dL_dw2);

    return loss;
}


int _main() {
    // random num generator init, must be called once
    // srand(time(NULL));
    srand(123);

    int N = 16;
    int M = 2;
    int D = 4;
    int O = 1;

    // *** Init ***
    tensor* x = Tensor(N, M);
    print(x, "x");

    tensor* w1 = Tensor(M, D);
    print(w1, "w1");

    tensor* w2 = Tensor(D, O);
    print(w2, "w2");

    // *** Train Step ***
    for (int ep_idx=0; ep_idx<NUM_EP; ep_idx++) {
        float loss = train_step(x, w1, w2);
        cout << "\nep: " << ep_idx << "; loss: " << loss << endl;

        print(w1, "w1");
        print(w2, "w2");
    }

    // todo: write to file
    return 0;
}


#include <deque> // deque from standard template library (STL)


int main() {
    srand(123);

    cout << "sizeof(tensor): " << sizeof(tensor) << endl << endl;

    // only used for shape inference
    tensor* _ = Tensor(2, 2);

    tensor* a = TensorLikeFill(_, 2.0);
    a->name = 'a';
    print(a, &a->name);
    tensor* b = TensorLikeFill(_, 2.0);
    b->name = 'b';
    print(b, &b->name);
    tensor* c = add(a, b);
    c->name = 'c';
    print(c, &c->name);

    tensor* d = TensorLikeFill(_, 3.0);
    d->name = 'd';
    tensor* e = mul(c, d);
    e->name = 'e';

    // e(2,2) @ f(2,5) = (2,5)
    tensor* f = Tensor(2, 5);
    f->name = 'f';
    print(f, "f");

    tensor* g = matmul(e, f);
    g->name = 'g';

    // tensor* loss = e;

    // mse
    tensor* y = TensorLikeFill(g, 0.0);
    tensor* loss = reduce_sum(pow(sub(y, g), 2));

    cout << "loss: " << loss->data[0] << endl;

    // bwd

    // todo: in autograd, don't overwrite grad instead do +=

    // todo: allocating grad buff[s] in ops previously led to err where, grad on
    //  the last node was set to start backprop loop "*e->grad = 1.0;"
    //  but bc buffer for grad is only allocated inside an Op, but e is never
    //  used by an op (e is last node in the computational graph) -- "*e->grad = 1.0;"
    //  is illegal as it the buffer hasn't ben allocated
    //   - one way to fix is allocate grad buff for all tensors in Tensor constructor
    //   - however, I do like that grad buff[s] are lazily created only when tensor is used.
    //     Which amounts to creating it here (in ops).

    // need to explicitly broadcast loss to the output shape
    //  of the op before it -- bc when chaining grads in the
    //  loop below with mul_k, it's assumed that shapes (of
    //  upstream and local) are the same
    loss->grad = FloatLikeFill(loss->inputs[0], 1.0);

    deque <tensor*> ready;
    ready.push_front(loss);
    while (ready.size() > 0) {
        tensor* t = ready.back(); ready.pop_back();

        printf("%s", t->op_name);
        // cout << "[autograd] " << t->op_name << endl;

        // each input of this op will have this as an upstream grad
        float* upstream = t->grad;

        // step once for the op -- propagates grad wrt all inputs of the op
        t->grad_fn(upstream, t);

        for (int i=0; i<t->num_inputs; i++){
            tensor* inp = t->inputs[i];
            // leaf tensors have no grad_fn, so don't push them on the queue
            // bc for each value pop'ed from the queue at later iterations,
            // this value's grad_fn will be called
            if (!inp->is_leaf) {
                ready.push_front(inp);
            }

            print_kernel(inp->grad, inp->size, inp->shape[1], &inp->name);
        }
    }

    return 0;
}
