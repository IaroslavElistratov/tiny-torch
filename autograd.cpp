#include <deque> // deque from standard template library (STL)
#include "nn.h"

#define IS_DEBUG_AG true
#define IS_DEBUG_BRANCHES false



void backward(tensor* loss){

    // open and close in write mode to clear the file, doing it here avoid needing to modify interface of lprint (which in general should NOT clear the file)
    // fclose(fopen("./generated/log.txt", "w"));

    if (!loss->grad_fn) {
        printf("[autograd engine] Error: tensor has no grad_fn\n");
        exit(1);
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

    // for the check below to pass
    loss->num_uses = 0;
    loss->grad = TensorLikeFill(loss, 1.0);

    deque <tensor*> ready;
    ready.push_front(loss);
    while (ready.size() > 0) {
        tensor* t = ready.back(); ready.pop_back();

        if (t->num_uses!=0){
            // push to the end the same thing we popped
            ready.push_front(t);

            if (IS_DEBUG_BRANCHES){
                printf("[autograd engine] pushed again %s (num_uses: %i)\n", t->name, t->num_uses);
                if (t->num_uses==-1) // -1
                    return;
            }
            continue;
        }

        if (IS_DEBUG_BRANCHES)
            printf("[autograd engine] done %s's grad\n", t->name);

        const char* op_name = OP_NAMES[t->op_type];
        if (IS_DEBUG_AG)
            printf("[autograd engine] %s\n", op_name);

        // comment:
        // when a tensor used by multiple ops, it's incorrect to access t->grad OR call t->grad_fn,
        // until output->grad_fn was not called on both of the outputs of the current op

        if (!t->grad_fn || !t->grad) {
            printf("[autograd engine] Error: tensor has no grad_fn\n");
            exit(1);
        }

        // each input of this op will have this as an upstream grad
        tensor* upstream = t->grad;
        // step once for the op -- propagates grad wrt all inputs of the op
        t->grad_fn(upstream, t);

        for (int i=0; i<t->num_inputs; i++){
            tensor* inp = t->inputs[i];

            // this condition can be used if decide to not compute ->grad
            // for some of the inputs to an op (e.g. idx->grad in select_bwd)
            // printf("[autograd engine] pushing: %s\n", inp->name);
            // if (!inp->grad){
            //     printf("[autograd engine] %s has no ->grad field\n", inp->name);
            //     continue;
            // }

            // will record pointers to all seen names -- to avid visiting same nodes twice, when
            //       exp
            //     /    \
            //    x1     x2
            //
            // check if we already visited this node
            bool is_pushed = false;
            // iterate over all quese and see if the "inp" is already pushed
            for (size_t ii=0; ii<ready.size(); ii++){
                if (ready.at(ii)->name == inp->name){
                    is_pushed = true;
                    break;
                }
            }

            // leaf tensors have no grad_fn, so don't push them on the queue
            // bc for each value pop'ed from the queue at later iterations,
            // this value's grad_fn will be called
            if (!inp->is_leaf && !is_pushed) {
                ready.push_front(inp);
            }

            // bc just called grad_fn of one of the outputs (t) of this tensor (inp)
            inp->num_uses--;

            // printf("\n%s's grad", inp->name);

            if (IS_DEBUG_AG){
                char buffer[30];
                sprintf(buffer, "%s_grad", inp->name);
                set_name(inp->grad, buffer);
                lprint(inp->grad);
            }
        }
    }
}
