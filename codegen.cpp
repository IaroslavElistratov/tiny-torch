#include "nn.h"


// todo-med: rm cprint[s] and re-use lprint[s] funcs -- modify a global pointer to whatever filenme a lprint should write to, set this pointer to "test.py" before calling lprint then set it back to "log.txt"




extern int index(tensor* t, ...);
extern void (*COPY_TO_DEVICE)(tensor*);
extern tensor* (*COPY_FROM_DEVICE)(tensor*);

// extern bool IS_CODEGEN;
#include <stdio.h> // structure declaration called FILE
FILE *fopen(char *name, char *mode);

void cprint_1d(tensor* t, FILE *f){
    tensor* t_copy = COPY_FROM_DEVICE(t);

    fprintf(f, "    [%12.8f, ]", t_copy->data[0]);
}

void cprint_2d(tensor* t, FILE *f){
    tensor* t_copy = COPY_FROM_DEVICE(t);

    for (int y=0; y<t->shape[0]; y++){
        fprintf(f, "    [");
        for (int z=0; z<t->shape[1]; z++){
            int idx = index(t_copy, y, z);
            fprintf(f, "%12.8f, ", t_copy->data[idx]);
        }
        fprintf(f, "],\n");
    }
}

void cprint_3d(tensor* t, FILE *f){
    tensor* t_copy = COPY_FROM_DEVICE(t);

    for (int x=0; x<t->shape[0]; x++){
        for (int y=0; y<t->shape[1]; y++){
            fprintf(f, "    [");
            for (int z=0; z<t->shape[2]; z++){
                int idx = index(t_copy, x, y, z);
                fprintf(f, "%12.8f, ", t_copy->data[idx]);
            }
            fprintf(f, "],\n");
        }
        // if (x < t->shape[0]-1)   // avoid empty lines after the last matrix
        fprintf(f, "\n");
    }
}

void cprint_4d(tensor* t, FILE *f){
    tensor* t_copy = COPY_FROM_DEVICE(t);

    for (int o=0; o<t->shape[0]; o++){
        for (int x=0; x<t->shape[1]; x++){
            for (int y=0; y<t->shape[2]; y++){
                fprintf(f, "    [");
                for (int z=0; z<t->shape[3]; z++){
                    int idx = index(t_copy, o, x, y, z);
                    fprintf(f, "%12.8f, ", t_copy->data[idx]);
                }
                fprintf(f, "],\n");
            }
            // if (x < t->shape[1]-1)
            fprintf(f, "\n");
        }
        // if (o < t->shape[0]-1)
        fprintf(f, "\n");
    }
}

void cprint(tensor* t, FILE *f){
    // handles Scalar tensors
    if (t->num_dims==2 && t->shape[0] == 1 && t->shape[1] == 1) cprint_1d(t, f);
    else if (t->num_dims==2) cprint_2d(t, f);
    else if (t->num_dims==3) cprint_3d(t, f);
    else if (t->num_dims==4) cprint_4d(t, f);
    else {
        printf("[cprint] Error");
        exit(1);
    }
}




void codegen_op_call(tensor* t){
    FILE *f = fopen("./generated/test.py", "a");
    switch (t->op_type) {
        case 0:
            fprintf(f, "%s = %s + %s\n", t->name, t->inputs[0]->name, t->inputs[1]->name);
            break;
        case 1:
            fprintf(f, "%s = %s - %s\n", t->name, t->inputs[0]->name, t->inputs[1]->name);
            break;
        case 2:
            fprintf(f, "%s = %s * %s\n", t->name, t->inputs[0]->name, t->inputs[1]->name);
            break;
        case 3:
            fprintf(f, "%s = %s @ %s\n", t->name, t->inputs[0]->name, t->inputs[1]->name);
            break;
        case 20:
            fprintf(f, "%s = %s / %s\n", t->name, t->inputs[0]->name, t->inputs[1]->name);
            break;
        case 18:
        {
            int axis = t->non_grad_inputs[0];
            int num_repeats = t->non_grad_inputs[1];
            if (axis == 0){
                fprintf(f, "%s = %s.repeat(%i, 1)\n", t->name, t->inputs[0]->name, num_repeats);
            } else if (axis == 1){
                fprintf(f, "%s = %s.repeat(1, %i)\n", t->name, t->inputs[0]->name, num_repeats);
            }
            break;
        }
        case 14:
            fprintf(f, "%s = torch.gather(%s, dim=1, index=%s.long())\n", t->name, t->inputs[0]->name, t->inputs[1]->name);
            break;
        case 4:
            fprintf(f, "%s = torch.pow(%s, %i)\n", t->name, t->inputs[0]->name, t->non_grad_inputs[0]);
            break;
        case 6:
            fprintf(f, "%s = F.relu(%s)\n", t->name, t->inputs[0]->name);
            break;
        case 7:
            fprintf(f, "%s = torch.transpose(%s, 0, 1)\n", t->name, t->inputs[0]->name);
            break;
        case 19:
            fprintf(f, "%s = - %s\n", t->name, t->inputs[0]->name);
            break;
        case 16:
            fprintf(f, "%s = torch.exp(%s)\n", t->name, t->inputs[0]->name);
            break;
        case 15:
            fprintf(f, "%s = torch.log(%s)\n", t->name, t->inputs[0]->name);
            break;
        case 8:
            fprintf(f, "%s = %s @ %s\n", t->name, t->inputs[0]->name, t->inputs[1]->name);
            break;
        case 13:
            fprintf(f, "%s = torch.flatten(%s, start_dim=1)\n", t->name, t->inputs[0]->name);
            break;
        case 5:
            fprintf(f, "%s = torch.sum(%s)\n", t->name, t->inputs[0]->name);
            break;
        case 17:
            fprintf(f, "%s = torch.sum(%s, axis=1, keepdim=True)\n", t->name, t->inputs[0]->name);
            break;
        case 21:
            fprintf(f, "%s = torch.max(%s)[0]\n", t->name, t->inputs[0]->name);
            break;
        case 22:
            fprintf(f, "%s = torch.max(%s, dim=1, keepdim=True)[0]\n", t->name, t->inputs[0]->name);
            break;
        // todo: for case 9 and 10, use STRIDE_CONV
        case 9:
            // need squeeze because F.conv2d expects bias of shape (F, ), but bc tiny-torch doesn't support 1d tensors its shape is (F, 1)
            fprintf(f, "%s = F.conv2d(%s, %s, bias=%s.squeeze(-1), stride=1, padding=0)\n", t->name, t->inputs[0]->name, t->inputs[1]->name, t->inputs[2]->name);
            break;
        case 10:
            fprintf(f, "%s = F.conv2d(%s, %s, bias=%s.squeeze(-1), stride=1, padding=0)\n", t->name, t->inputs[0]->name, t->inputs[1]->name, t->inputs[2]->name);
            break;
        // todo: for case 11 and 12, use STRIDE_MAXPOOL
        case 11:
            fprintf(f, "%s = F.max_pool2d(%s, kernel_size=2, stride=2, padding=0)\n", t->name, t->inputs[0]->name);
            break;
        case 12:
            fprintf(f, "%s = F.max_pool2d(%s, kernel_size=2, stride=2, padding=0)\n", t->name, t->inputs[0]->name);
            break;

        default:
            printf("[codegen] unexpected op_type");
            exit(1);
    }
    fclose(f);
}

void codegen_tensor(tensor* t){
    FILE *f = fopen("./generated/test.py", "a");

    fprintf(f, "_%s = np.array([\n", t->name);
    // IS_CODEGEN = true;
    cprint(t, f);
    // IS_CODEGEN = false;
    fprintf(f, "])\n");
    fprintf(f, "%s = torch.Tensor(_%s.reshape%s)\n", t->name, t->name, str_shape(t));
    fprintf(f, "%s.requires_grad = True\n\n", t->name);
    fclose(f);
}

// todo-low: assert shapes are same
void codegen_assert_close(tensor* t){
    FILE *f = fopen("./generated/test.py", "a");

    // CODEGEN TENSOR
    fprintf(f, "_tiny_torch_%s = np.array([\n", t->name);
    cprint(t, f);
    fprintf(f, "])\n");
    fprintf(f, "_tiny_torch_%s = torch.Tensor(_tiny_torch_%s.reshape%s)\n", t->name, t->name, str_shape(t));
    // it's convenient to see how difference changes throughout the graph (later in the graph vs in the beginning)
    fprintf(f, "print('%s abs diff: ', torch.sum(torch.abs(%s) - torch.abs(_tiny_torch_%s)).item())\n", t->name, t->name, t->name);
    // todo-low: use "np.testing.assert_allclose" ?
    fprintf(f, "assert torch.allclose(%s, _tiny_torch_%s, atol=1e-4)\n\n", t->name, t->name);
    fclose(f);

}

void codegen_assert_grad_close(tensor* t){
    FILE *f = fopen("./generated/test.py", "a");

    // CODEGEN TENSOR
    fprintf(f, "_tiny_torch_%s_grad = np.array([\n", t->name);
    cprint(t->grad, f);
    fprintf(f, "])\n");
    fprintf(f, "_tiny_torch_%s_grad = torch.Tensor(_tiny_torch_%s_grad.reshape%s)\n", t->name, t->name, str_shape(t));
    fprintf(f, "print('%s grad abs diff: ', torch.sum(torch.abs(%s.grad) - torch.abs(_tiny_torch_%s_grad)).item())\n", t->name, t->name, t->name);
    fprintf(f, "assert torch.allclose(%s.grad, _tiny_torch_%s_grad, atol=1e-4)\n\n", t->name, t->name);

    // sanity check -- abs diff of grad with itself
    // fprintf(f, "print('[SANITY CHECK] %s grad abs diff: ', torch.sum(torch.abs(%s.grad) - torch.abs(%s.grad)).item())\n\n", t->name, t->name, t->name);
    fclose(f);

}




void codegen_imports(void){
    // clear contents of the file
    fclose(fopen("./generated/test.py", "w"));

    FILE *f = fopen("./generated/test.py", "a");
    fprintf(f, "# --------------------------\n# ATTENTION:\n# THIS FILE IS AUTOGENERATED\n# DO NOT MODIFY BY HAND.\n# --------------------------\n\n\n\n");

    fprintf(f, "import numpy as np\n");
    fprintf(f, "import torch\n");
    fprintf(f, "import torch.nn.functional as F\n");
    fprintf(f, "torch.set_printoptions(precision=8, sci_mode=False, threshold=10_000, edgeitems=100, linewidth=1000)\n");
    // use fclose to that it appears in the beginning of the file
    fclose(f);
}

// need this fn bc in tiny torch call to AG is not an op, so can't just add another case to the switch statement (to represent the backward call)
void codegen_backward_call(tensor* t){
    FILE *f = fopen("./generated/test.py", "a");
    fprintf(f, "%s.backward(torch.ones_like(%s))\n", t->name, t->name);
    fclose(f);
}




// comment:
// split the three passes of recursive_traverse  (where in the 1st pass I generate leaf tensors;
// in the 2nd pass op calls; and in the 3rd pass asserts for intermediate tensors) -- this way the
// generated code is more readable (all tensors are grouped together and declared above the ops,
// followed by the ops, which are followed by the asserts)

void codegen_all_leafs(tensor* t){
    if (t->num_uses != 0){
        return;
    }

    for (int i=0; i<t->num_inputs; i++){
        tensor* inp = t->inputs[i];
        // printf("[codegen_all_leafs] %s\n", inp->name);
        inp->num_uses--;
        codegen_all_leafs(inp);
    }

    if (t->is_leaf){
        // todo-low: don't write "input" and "label" tensors ?
        codegen_tensor(t);
    }
}

void codegen_all_ops(tensor* t){
    if (t->num_uses != 0){
        return;
    }

    for (int i=0; i<t->num_inputs; i++){
        tensor* inp = t->inputs[i];
        // printf("[codegen_all_ops] %s\n", inp->name);
        inp->num_uses--;
        codegen_all_ops(inp);
    }

    // todo: && "t->num_uses == 0"? bc num uses might have changed while processing the inputs
    if (!t->is_leaf){
        // note codegen_op_call is after the recursive call, so
        // codegen_op_call will be called when the recursive calls tack unwinds
        // which will generate code in the correct (reverse) order
        codegen_op_call(t);
    }
}

void codegen_all_asserts(tensor* t){
    if (t->num_uses != 0){
        return;
    }

    for (int i=0; i<t->num_inputs; i++){
        tensor* inp = t->inputs[i];
        // printf("[codegen_all_asserts] %s\n", inp->name);
        inp->num_uses--;
        codegen_all_asserts(inp);
    }

    // moved codegen_assert_close into the recursive function -- to avoid adding it here after each instruction
    if (!t->is_leaf){
        codegen_assert_close(t);
    }
}




// todo-now: can I run recursive_traverse and AG wt destroying t->num_inputs?
//   - at the moment I can run either AG or codegen -- to assert grads, have to be able to run both
// using these helper fns for now
void _save_num_uses(tensor* t){
    for (int i=0; i<t->num_inputs; i++){
        tensor* inp = t->inputs[i];
        inp->_num_uses = inp->num_uses;
        _save_num_uses(inp);
    }
}

void save_num_uses(tensor* t){
    t->num_uses = 0;
    _save_num_uses(t);
}

void rest_num_uses(tensor* t){
    for (int i=0; i<t->num_inputs; i++){
        tensor* inp = t->inputs[i];
        inp->num_uses = inp->_num_uses;
        rest_num_uses(inp);
    }
}

// this fn should be called on a final output of the graph (e.g. loss)
void generate_test(tensor* loss){
    // using t->num_uses in my codegen, and bc AG destroys these need to reset them here
    rest_num_uses(loss);

    codegen_imports();

    FILE *f = fopen("./generated/test.py", "a");
    fprintf(f, "\n\n\n# ~~~~~~~~~~ leafs ~~~~~~~~~~\n\n\n\n");
    fclose(f);
    codegen_all_leafs(loss);
    rest_num_uses(loss);

    f = fopen("./generated/test.py", "a");
    fprintf(f, "\n\n\n# ~~~~~~~~~~ ops ~~~~~~~~~~\n\n\n\n");
    fclose(f);
    codegen_all_ops(loss);
    rest_num_uses(loss);

    codegen_op_call(loss);
    codegen_backward_call(loss);

    f = fopen("./generated/test.py", "a");
    fprintf(f, "\n\n\n# ~~~~~~~~~~ intermediate tensors asserts ~~~~~~~~~~\n\n\n\n");
    fclose(f);
    codegen_all_asserts(loss);
    rest_num_uses(loss);

    f = fopen("./generated/test.py", "a");
    fprintf(f, "\n\n\n# ~~~~~~~~~~ grad asserts ~~~~~~~~~~\n\n\n\n");
    fclose(f);

    // // param_head is a global variable
    // extern param* param_head;
    param* temp = param_head;
    while (temp){
        // printf("[codegen_assert_grad_close] %s\n", temp->value->name);
        codegen_assert_grad_close(temp->value);
        temp = temp->next;
    }

    f = fopen("./generated/test.py", "a");
    fprintf(f, "print('--------------------------\\n TEST PASSED!\\n--------------------------')");
    fclose(f);
}




/*
parts of AG relevant to recursive_traverse:

void backward(tensor* loss){

        for (int i=0; i<t->num_inputs; i++){
            tensor* inp = t->inputs[i];

            // leaf tensors have no grad_fn, so don't push them on the queue
            // bc for each value pop'ed from the queue at later iterations,
            // this value's grad_fn will be called
            if (!inp->is_leaf && !is_pushed) {
                ready.push_front(inp);
            }

            // bc just called grad_fn of one of the outputs (t) of this tensor (inp)
            inp->num_uses--;

        }
    }
}
*/
