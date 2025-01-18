#include "nn.h"

#define PARAM_DEBUG false

// HEAD
param* param_head = NULL;

void log_params(void){
    param* temp = param_head;
    while (temp){
        tensor* w = temp->value;
        lprint(w);
        temp = temp->next;
    }
}

void print_num_params(void){
    param* temp = param_head;
    int num_params = 0;
    while (temp){
        tensor* w = temp->value;
        num_params += w->size;
        temp = temp->next;
    }
    printf("\nNum trainable params: %i\n", num_params);
}

int count_params(void){
    param* temp = param_head;
    int num_params = 0;
    while (temp){
        num_params += 1;
        temp = temp->next;
    }
    return num_params;
}

void add_param(tensor* t){
    param* new_param = (param*)checkMallocErrors(malloc(sizeof(param)));
    new_param->value = t;
    new_param->next = NULL;
    // todo: to reduce memory usage, use "velocity" field on the param
    //  struct to store "first_moments" for adam -- bc when adam is used
    //  currently "velocity" serves no purpose but does take up memory.
    // Even if SGD's "velocity" is slightly semantically different from Adam's
    //  "first moments", in both cases it's just a tensor with same shape as w
    //  -- can use it for both
    new_param->velocity = TensorLikeFill(t, 0.0);
    // used by adam:
    new_param->t = 0;
    new_param->beta1 = 0.9;
    new_param->beta2 = 0.999;
    new_param->epsilon = 1e-8;
    new_param->first_moment = TensorLikeFill(t, 0.0);
    new_param->second_moment = TensorLikeFill(t, 0.0);

    // append to the beginning of the linked list
    new_param->next = param_head;
    param_head = new_param;
}

// expects "name" be a NULL terminated string
tensor* get_param(const char* name){
    if (!param_head){
        printf("[get_param] linked list of params is empty\n");
        exit(1);
    }

    param* temp = param_head;
    while (temp){
        if (PARAM_DEBUG){
            printf("[get_param] iterating over %s\n", temp->value->name);
        }
        if (strcmp(name, temp->value->name) == 0){
            return temp->value;
        }
        temp = temp->next;
    }

    printf("[get_param] couldn't find %s in the linked list of params\n", name);
    exit(1);
}
