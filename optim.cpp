
// todo: mv to composite_ops
void sgd(float learning_rate) {
    // // param_head is a global variable
    // extern param* param_head;
    param* temp = param_head;
    // iterate over the linked list of params
    while (temp){
        // printf("[sgd] %s\n", temp->tensor->name);
        tensor* w = temp->tensor;

        // more convenient way (than the approach below) is to not
        // even create out tensor (new weight) in the first place
        sub_k_(w, mul_k(w->grad, TensorLikeFill(w->grad, learning_rate)), w);

        // // assign to the "w->data" because I want the weight to be persistent --
        // //  my first N tensors (weights, dataset) in the GC list are never removed from the GC;
        // //  but if I were to just replace the entire old weight tensor with the new weight tensor,
        // //  the new weight tensor would be cleaned up next epoch (bc it's in the end of the
        // //  GC list -- not one of the first N elements)
        // cudaFree(w->data);
        // tensor* new_w = sub_k(w, mul_k(w->grad, TensorLikeFill(w->grad, learning_rate)));
        // GC_IDX--; // don't track "new_w"
        // w->data = new_w->data;

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
