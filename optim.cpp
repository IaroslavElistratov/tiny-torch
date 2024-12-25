
// todo: to avoid data transfer impl sgd as composite_op: sub_(w->grad, mul(w->grad, TensorLikeFill(w, 0.2)), w->grad)
void sgd(float learning_rate) {
    // // param_head is a global variable
    // extern param* param_head;
    param* temp = param_head;
    // iterate over the linked list of params
    while (temp){
        // printf("[sgd] %s\n", temp->tensor->name);
        tensor* w = temp->tensor;
        tensor* w_local = COPY_FROM_DEVICE(w);
        tensor* w_grad_local = COPY_FROM_DEVICE(w->grad);
        for (int i=0; i<w->size; i++){
            w_local->data[i] -= w_grad_local->data[i] * learning_rate;
        }
        COPY_TO_DEVICE(w_local);
        // todo: memory leak
        w->data = w_local->data;

        temp = temp->next;
    }
}

void zero_grads() {
    // // param_head is a global variable
    // extern param* param_head;
    param* temp = param_head;
    while (temp){
        // printf("[zero_grads] %s\n", temp->tensor->name);
        temp->tensor->grad = NULL;
        temp = temp->next;
    }
}
