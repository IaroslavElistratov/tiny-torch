#include <deque> // deque from standard template library (STL)

#include "nn.h"

#define print(f_p) _print(f_p)
void _print(tensor* t);

void backward(tensor* loss){

    if (loss->grad_fn == NULL) {
        cout << "[autograd engine] Error: tensor has no grad_fn" << endl;
        return;
    }

    // todo: in autograd, don't overwrite grad instead do +=

    // allocating grad buff[s] in ops previously led to err where, grad on
    //  the last node was set to start backprop loop "*e->grad = 1.0;"
    //  but bc buffer for grad is only allocated inside an Op, but e is never
    //  used by an op (e is last node in the computational graph) -- "*e->grad = 1.0;"
    //  is illegal as it the buffer hasn't ben allocated
    //   - one way to fix is allocate grad buff for all tensors in Tensor constructor
    //   - however, I do like that grad buff[s] are lazily created only when tensor is used.
    //     Which amounts to creating it here (in ops).

    // todo: make more general?
    // need to explicitly broadcast loss to the output shape
    //  of the op before it -- bc when chaining grads in the
    //  loop below with mul_k, it's assumed that shapes (of
    //  upstream and local) are the same
    loss->grad = TensorLikeFill(loss->inputs[0], 1.0);

    deque <tensor*> ready;
    ready.push_front(loss);
    while (ready.size() > 0) {
        tensor* t = ready.back(); ready.pop_back();

        const char* op_name = OP_NAMES[t->op_type];
        printf("%s", op_name);
        // cout << "[autograd] " << op_name << endl;

        // each input of this op will have this as an upstream grad
        tensor* upstream = t->grad;

        if (t->grad_fn == NULL) {
            cout << "[autograd engine] Error: tensor has no grad_fn" << endl;
            return;
        }

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

            printf("\n%s's grad", inp->name);
            print(inp->grad);
        }
    }
}
