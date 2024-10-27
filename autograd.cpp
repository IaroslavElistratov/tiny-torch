#include <deque> // deque from standard template library (STL)

#include "nn.h"

#define print(f_p) _print(f_p)
void _print(tensor* t);

#define IS_DEBUG_AG true


void print_grad(tensor* inp){
    char buffer[30];
    sprintf(buffer, "%s_grad", inp->name);
    set_name(inp->grad, buffer);

    // todo: move this into lprint
    if (inp->grad->num_dims==2) {
        lprint_2d(inp->grad);
    } else if (inp->grad->num_dims==3) {
        lprint_3d(inp->grad);
    } else if (inp->grad->num_dims==4) {
        lprint_4d(inp->grad);
    } else {
        printf("[autograd] Error");
    }
}


tensor* tensor_like(tensor* loss){
    tensor* grad;
    if (loss->num_dims==2)
        grad = TensorLikeFill(loss, 1.0);
    else if (loss->num_dims==3)
        grad = TensorLikeFill3d(loss, 1.0);
    else if (loss->num_dims==4)
        grad = TensorLikeFill4d(loss, 1.0);
    else {
        printf("[autograd engine] Error");
    }
    return grad;
}



void backward(tensor* loss){

    if (!loss->grad_fn) {
        printf("[autograd engine] Error: tensor has no grad_fn\n");
        return;
    }

    loss->num_uses = 0;
    loss->grad = tensor_like(loss);

    deque <tensor*> ready;
    ready.push_front(loss);
    while (ready.size() > 0) {
        tensor* t = ready.back(); ready.pop_back();

        if (t->num_uses!=0){
            // push to the end the same thing we popped
            ready.push_front(t);
            continue;
        }

        const char* op_name = OP_NAMES[t->op_type];
        if (IS_DEBUG_AG)
            printf("[autograd engine] %s\n", op_name);

        if (!t->grad_fn || !t->grad) {
            printf("[autograd engine] Error: tensor has no grad_fn\n");
            return;
        }

        // each input of this op will have this as an upstream grad
        tensor* upstream = t->grad;
        // step once for the op -- propagates grad wrt all inputs of the op
        t->grad_fn(upstream, t);

        for (int i=0; i<t->num_inputs; i++){
            tensor* inp = t->inputs[i];

            bool is_pushed = false;
            for (size_t ii=0; ii<ready.size(); ii++){
                if (ready.at(ii)->name == inp->name){
                    is_pushed = true;
                    break;
                }
            }

            if (!inp->is_leaf && !is_pushed) {
                ready.push_front(inp);
            }

            inp->num_uses--;

            if (IS_DEBUG_AG){
                print_grad(inp);
            }
        }
    }
}
