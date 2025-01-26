void adam(float learning_rate) {
    using namespace tiny_torch;

    param* temp = param_head;

    while (temp){
        tensor* w = temp->value;

        int t = temp->t;
        tensor* beta1 = TensorLikeFill(w, temp->beta1);
        tensor* beta2 = TensorLikeFill(w, temp->beta2);
        tensor* epsilon = TensorLikeFill(w, temp->epsilon);
        tensor* first_moment = temp->first_moment;
        tensor* second_moment = temp->second_moment;

        t++;

        // first_moment = beta1 * first_moment + (1 - beta1) * dw
        add_k_(
            mul_k(beta1, first_moment),
            mul_k(sub(TensorLikeFill(beta1, 1.), beta1), w->grad),
            first_moment
        );

        // second_moment = beta2 * second_moment + (1 - beta2) * dw**2
        add_k_(
            mul_k(beta2, second_moment),
            mul_k(sub_k(TensorLikeFill(beta2, 1.), beta2), pow_k(w->grad, 2)),
            second_moment
        );

        // first_unbias = first_moment / (1 - beta1**t)
        tensor* first_unbias = div_k(
            first_moment,
            sub_k(
                TensorLikeFill(beta1, 1.),
                pow_k(beta1, t)
            )
        );

        // second_unbias = second_moment / (1 - beta2**t)
        tensor* second_unbias = div_k(
            second_moment,
            sub_k(
                TensorLikeFill(beta2, 1.),
                pow_k(beta2, t)
            )
        );

        // w -= lr * first_unbias / (np.sqrt(second_unbias) + epsilon)
        sub_k_(
            w,
            div_k(
                mul_k(
                    TensorLikeFill(first_unbias, learning_rate),
                    first_unbias
                ),
                add_k(
                    sqrt_k(second_unbias),
                    epsilon
                )
            ),
            w
        );

        temp = temp->next;
    }
}


// todo: mv to composite_ops
void sgd(float learning_rate, float momentum) {
    // // param_head is a global variable
    // extern param* param_head;
    param* temp = param_head;
    // iterate over the linked list of params
    while (temp){
        // printf("[sgd] %s\n", temp->value->name);
        tensor* w = temp->value;

        if (momentum == 0.){
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
        } else {
            /*
            https://github.com/keras-team/keras/blob/e0108291a2c7a91271cb774bb130a4b8c576fb20/keras/src/optimizers/sgd.py#L19
            velocity = momentum * velocity - learning_rate * g
            w = w + velocity
            */
            tensor* velocity = temp->velocity;

            // note: using add_k_ here to avoid erasing in free_all_tensors
            sub_k_(
                mul_k(TensorLikeFill(velocity, momentum), velocity),
                mul_k(TensorLikeFill(w->grad, learning_rate), w->grad),
                velocity
            );
            add_k_(w, velocity, w);
        }

        temp = temp->next;
    }
}

void zero_grads(void) {
    // // param_head is a global variable
    // extern param* param_head;
    param* temp = param_head;
    while (temp){
        // printf("[zero_grads] %s\n", temp->value->name);
        temp->value->grad = NULL;
        temp = temp->next;
    }
}
